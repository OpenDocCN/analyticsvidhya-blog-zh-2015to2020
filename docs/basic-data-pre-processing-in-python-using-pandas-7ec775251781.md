# 使用 pandas 在 Python 中进行基础数据预处理

> 原文：<https://medium.com/analytics-vidhya/basic-data-pre-processing-in-python-using-pandas-7ec775251781?source=collection_archive---------0----------------------->

数据科学家需要执行几个数据预处理步骤。今天，我在博客中列出了一些常见的步骤。这个博客的 Jupyter 笔记本可以从[这里](https://github.com/vikeshsingh37/PythonTutorials/blob/master/DataProcessingInPython.ipynb)获得

我们开始吧。我从[犯罪 2018](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2) 中获取了这个故事的数据。数据存储在 csv 文件中。

# **负载数据**

第一步是读取 csv 文件。如果文件不是很大，我们可以一次性读取文件。

```
import pandas as pd
dat = pd.read_csv("Crimes2018.csv")
```

但是，如果文件很大，我们可以使用 pd.read_csv()中的 chunksize 以小数据块的形式读取文件。chunksize 是每次迭代中读取的行数。

```
for chunk in pd.read_csv("Crimes2018.csv", chunksize = 10000):
     dat = chunk
     ## Do what ever operation you want 
     print(chunk.shape)
```

# **检查加载的数据**

让我们假设文件不是很大，数据是一次性读取的。现在，让我们看看前 5 行数据。为此，

```
dat.head()
```

如果想打印更多的行，将行数作为参数传递给 head。

```
dat.head(10)
```

要查看数据框中有多少行和列

```
dat.shape
```

很好，我们已经看到了如何读取 csv 文件，检查数据的前几行以及数据中的行数和列数。

接下来是如何获得数据帧的摘要。这里有数字列和分类列。因此，在描述函数中使用了“include = all”。否则，将只显示数字列摘要

```
dat.describe(include = 'all')
```

# **查找分类和数字列**

很多时候需要识别分类列，例如对于一个热编码。下面的代码行将告诉我们哪些列是分类的，哪些是其他的。

```
cat_cols = dat.columns[(dat.dtypes == "object").tolist()].tolist()
num_cols = dat.columns[(dat.dtypes != 'object').tolist()].tolist()
```

# **查找缺少值的列和行**

现在是缺失值列。下面将显示 dat 的列中缺少的行数。

```
missing_vals = dat.isnull().sum()
missing_vals
```

要获取缺少值的列的列表:

```
missing_vals_cols = missing_vals[missing_vals > 0].axes[0].tolist()
missing_vals_cols
```

我们还可以看到缺少值的数据行。比方说，我们想检查哪些行缺少“位置描述”

```
dat.loc[dat['Location Description'].isnull()]
```

# **估算缺失值**

处理缺失值需要大量的研究和专业知识。我在这里只是告诉如何填充 NA 值，而不是插补策略。假设我们想用一个常数值(比如 10)估算“病房”列中的 NA 值。

```
dat['Ward'].fillna(10)
```

但这不会替换 dat 数据框中的值。为此，需要使用 inplace = True

```
dat['Ward'].fillna(10, inplace = True)
```

而不是 10，如果有人想用“Ward”的中值代替 NA 值

```
dat['Ward'].fillna(dat['Ward'].median(skipna = True), inplace = True)
```

上述方法适用于数字列。对于分类列，可以使用模式或字符“NotAvailable”。

```
# Replace NA by NotAvailable
dat['Location Description'].fillna("NotAvailable", inplace= True)# Replace NA by Mode
dat['Location Description'].fillna(dat['Location Description'].mode().tolist()[0], inplace= True)
```

# **删除缺少值的行**

有几行数据不能用任何方法估算。需要移除这些行，而机器学习算法无法处理这些行。缺失值的行数可由下式得出:

```
num_missing_rows = dat.isnull().any(axis =1).sum()
```

缺少值的行可以通过以下方式删除:

```
dat.dropna(axis =0, inplace = True)
dat.reset_index(inplace = True)
```

# **删除列**

可能需要删除一些列。假设我们想要删除“纬度”和“经度”列。

```
dat.drop(['Latitude','Longitude'],axis = 1, inplace = True)
```

# **对列中的值进行排序**

比方说，我们希望按照日期升序和 IUCR 降序对数据框进行排序。

```
dat.sort_values(["Date","IUCR"],ascending=[True, False], inplace = True)
dat.reset_index(inplace = True)
```

# **将一列拆分成多列**

可能需要用分隔符拆分列中的值，并创建两个新列。假设我们想要将“位置”列拆分为纬度和经度。数据中位置列的一个例子是(41.881892729，-87.38515564)

```
pattern = '|'.join(['\\(','\\)'])
dat = pd.concat([dat,dat["Location"].str.replace(pat = pattern,repl = "").str.split(",",expand = True)], axis = 1)# Rename columns
dat.rename(index =str, columns = {0:'Latitude',1:'Longitude'}, inplace=True)# Convert all columns to lowercase
dat.rename(str.lower, axis = 1, inplace= True)
```

# **从分类列创建虚拟列**

分类列中的值有时需要作为列存在。一种方法是添加虚拟变量。下面创建了“主要类型”列的虚拟变量，并将其添加到数据框中。

```
dat = pd.concat([dat,pd.get_dummies(dat["primary type"])], axis = 1)
dat.drop("primary type", axis = 1, inplace = True)
dat.rename(str.lower, axis = "columns", inplace = True)
dat.reset_index(drop=True,inplace=True)
```

# 按操作分组

很多时候需要通过操作来做一组。下面的例子给出了按“日期”和“位置描述”分组的“盗窃”的总和

```
pd.DataFrame(dat.groupby(["date","location description"])["theft"].sum()).reset_index()
```

# 筛选列中的行值

可以过滤列中的行值。位置描述为公寓或街道且盗窃为 0 的示例过滤数据。

```
dat[dat["location description"].isin(["APARTMENT","STREET"]) & (dat["theft"] == 0)]
```

# 应用功能

有时需要对某些列的所有行应用一个函数。假设我们想要通过将每个值除以它们的标准 z 分数来规范化“警察巡逻”和“盗窃”列。

```
app_dat = dat[["police beats","theft"]].apply(lambda x: (x-np.mean(x))/np.std(x), axis = 0)
```

对于“轴= 1”的每一列，可以沿着行执行类似的操作

# **获取相关矩阵**

```
num_cols = [i.lower() for i in num_cols]
dat[num_cols].corr()
```

# 更改列类型

你也可以改变熊猫的列类型。在本例中，列“逮捕”是布尔型的。比如说，我们想把它改成 object。

```
dat['arrest'] = dat['arrest'].astype('object')
```

# 按列透视

在几个操作中需要通过索引列上的列进行透视。下面的例子将枢纽列“逮捕”超过“索引”列，这必须是唯一的。新列中的值取自“抢劫”列。这些可以引入由-1 填充的 NaN。然后将透视数据帧添加回原始数据帧。

```
pivoted_dat = dat.pivot(index= "index", columns= "arrest", values="robbery").reset_index(drop = True)
pivoted_dat.fillna(-1, inplace=True)
dat = pd.concat([dat,pivoted_dat], axis=1)
```

# 逆透视数据(融化)

可能需要对列进行逆透视，以制作长格式的数据帧。这可以通过熔化来完成。在下面的示例中，列“索引”、“案例号”和“日期”将保持在原始数据框中。“描述”和“块”列中的值将作为行添加。这将导致数据框的行数比原始数据框多。

```
melted_dat = dat.melt(id_vars=["index","case number","date"], value_vars= ["description","block"])
```

# 连接

在许多情况下，可能需要连接两个数据帧。这里，我采用了两个不同于博客其他部分的数据框架。这些数据框创建为:

```
df1 = pd.DataFrame({'col1':[1,2,3,4,5], 'col2' : [12,43,10,20,2],'col3':['A','B','C','X','Y']})
df2 = pd.DataFrame({'col1':[1,2,3,3,4], 'col4' : [12,43,10,20,2],'col5':['AA','BB','CC','XX','YY']})
```

下面的例子展示了如何在 pandas 中进行所有类型的连接。

```
# Inner Join
df1.join(df2.set_index(['col1','col4']), on = ['col1','col2'], how = "inner")# Left Join
df1.join(df2.set_index(['col1','col4']), on = ['col1','col2'], how = "left")# Right Join
df1.join(df2.set_index(['col1','col4']), on = ['col1','col2'], how = "right")
```

我没有在这个博客中讨论过策划。稍后我可能会谈到这一点。希望你喜欢这个博客。如果有，请鼓掌。