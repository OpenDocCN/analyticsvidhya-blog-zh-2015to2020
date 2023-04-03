# 成人收入数据集的探索性和预测性数据分析

> 原文：<https://medium.com/analytics-vidhya/exploratory-and-predictive-data-analysis-of-the-adult-income-data-set-58a4d3abc04d?source=collection_archive---------24----------------------->

这个[数据集](https://archive.ics.uci.edu/ml/datasets/Adult)在 UCI ML 知识库中，有大约 45000 个人的人口统计细节，如年龄、性别、种族等。我将分享我预测个人收入是否超过 50K 的方法。

对于这个项目，我使用了 python 和 Jupyter Notebook。一开始，我导入了相关的库(在上帝创造天地之前)。除此之外，根据你的选择设置可视化的大小。。我用了一种概率方法来处理分类变量。这个项目的全部代码可以在[这里](https://github.com/codebat137/JupyterNoteboooks/blob/master/adult_income.ipynb)找到。

```
**import** **pandas** **as** **pd**
**import** **numpy** **as** **np**
**import** **sklearn** **as** **sk**
**import** **seaborn** **as** **sns**
**import** **matplotlib.pyplot** **as** **plt**
%matplotlib inline
plt.rcParams["figure.figsize"] = (20,10)
```

让我们导入训练数据集并开始吧。我们必须手动添加列标签，因为该数据集没有列标签。

```
data = pd.read_csv('adult.data')
data.info()
data.columns = ['age','work-class','fnlwgt','education','edu-num','marital',            'occup','relatnip','race','sex','gain','loss','hours','citizenship','>50k']
```

上述单元格的输出。由此可以推断，我们不必太担心缺失值。

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32560 entries, 0 to 32559
Data columns (total 15 columns):
39                32560 non-null int64
 State-gov        32560 non-null object
 77516            32560 non-null int64
 Bachelors        32560 non-null object
 13               32560 non-null int64
 Never-married    32560 non-null object
 Adm-clerical     32560 non-null object
 Not-in-family    32560 non-null object
 White            32560 non-null object
 Male             32560 non-null object
 2174             32560 non-null int64
 0                32560 non-null int64
 40               32560 non-null int64
 United-States    32560 non-null object
 <=50K            32560 non-null object
dtypes: int64(6), object(9)
memory usage: 3.7+ MB
```

由于最后一列有一个对象类型，我们需要找出使用的唯一标签，然后将其映射到整数类型。

```
data['>50k'].unique()
```

相应的输出:

```
array([' <=50K', ' >50K'], dtype=object)
```

现在我们将收入小于或等于 50k 对应的标签映射到 0，另一个映射到 1。我们还将再次获取信息，并检查列的数据类型是否已经更改为 int。

```
data['>50k'] = data['>50k'].map({' <=50K':0,' >50K':1})
data.info()
```

正如您可以从输出中推断的那样，该列的数据类型已经更改为 64 位整数类型。

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32560 entries, 0 to 32559
Data columns (total 15 columns):
age            32560 non-null int64
work-class     32560 non-null object
fnlwgt         32560 non-null int64
education      32560 non-null object
edu-num        32560 non-null int64
marital        32560 non-null object
occup          32560 non-null object
relatnip       32560 non-null object
race           32560 non-null object
sex            32560 non-null object
gain           32560 non-null int64
loss           32560 non-null int64
hours          32560 non-null int64
citizenship    32560 non-null object
>50k           32560 non-null int64
dtypes: int64(7), object(8)
memory usage: 3.7+ MB
```

现在，我写了一个函数，目的是根据分类变量的任何值，得到收入超过 50K 的概率。例如，每种性别、公民身份等的收入超过 50K 的概率是多少？为了实现这个目标，我计算了 50k '列的平均值。因为唯一可能的值是 0 和 1，所以平均值将对应于得到 1 的概率。

```
**def** rep_mean(var,data):   
   l = list(data[var].unique())
   d = dict()     
   **for** obj **in** l:      
      d[obj] = data[data[var]==obj]['>50k'].mean()
   **return** d
```

让我们在 work-class 列上检查这个函数，因为它恰好是分类函数。

```
temp1 = rep_mean('work-class',data)
temp1
```

输出返回一个字典。

```
{' Self-emp-not-inc': 0.2849271940181031,
 ' Private': 0.21867289390200917,
 ' State-gov': 0.27216653816499614,
 ' Federal-gov': 0.38645833333333335,
 ' Local-gov': 0.29479216435738176,
 ' ?': 0.10403050108932461,
 ' Self-emp-inc': 0.557347670250896,
 ' Without-pay': 0.0,
 ' Never-worked': 0.0}
```

这个字典可以映射到 work-class 列。

```
data['work-class'] = data['work-class'].map(temp1)
data.info()
```

正如您可以从输出中推断的那样，它的数据类型已经更改为 64 位浮点数。

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32560 entries, 0 to 32559
Data columns (total 15 columns):
age            32560 non-null int64
work-class     32560 non-null float64
fnlwgt         32560 non-null int64
education      32560 non-null object
edu-num        32560 non-null int64
marital        32560 non-null object
occup          32560 non-null object
relatnip       32560 non-null object
race           32560 non-null object
sex            32560 non-null object
gain           32560 non-null int64
loss           32560 non-null int64
hours          32560 non-null int64
citizenship    32560 non-null object
>50k           32560 non-null int64
dtypes: float64(1), int64(7), object(7)
memory usage: 3.7+ MB
```

我对其他分类列，也就是‘object’列做了同样的处理。整个实现可以在[这里](https://github.com/codebat137/JupyterNoteboooks/blob/master/adult_income.ipynb)找到。我通过将几列的值相加，并对概率和相关系数的乘积求和，创建了几个虚拟变量。

```
data['net'] = data['gain']+data['loss']
x = data.corr() 
x
```

我选择不在这里显示相关表，因为它没有清晰地呈现。我把这些变量放在一起，因为它们彼此之间有很强的相关性。

```
data['thresh1'] = 1000*(0.368866*data['education']+0.351885*data['occup'])data['thresh2'] = 1000*(0.447396*data['marital']+0.453578*data['relatnip'])
```

我用了很多算法测试。最后我选择了梯度提升，尽管它在训练数据上表现不佳。然而，与决策树和随机森林不同，它没有过度拟合的问题。完整的代码和输出可以在 Github 的[这里](https://github.com/codebat137/JupyterNoteboooks/blob/master/adult_income.ipynb)找到。