# Python 中的 ETL 管道

> 原文：<https://medium.com/analytics-vidhya/design-an-etl-pipeline-using-pandas-fd42adcb21f7?source=collection_archive---------2----------------------->

在数据世界中，ETL 代表提取、转换和加载。几乎在每个数据管道或工作流中，我们通常从各种来源(结构化、半结构化或非结构化格式)提取数据，转换或清理数据，并将其加载到 ODS(运营数据源)或 DWH(数据仓库)中，这最后一步基本上将数据整理成统一的格式，供各种下游应用程序使用。

执行 ETL 有多种方式，可以是普通的 SQL(结构化查询语言)、ETL 工具或任何编程语言。这篇博客的主要重点是设计一个非常基本的 ETL 管道，在这里我们将学习从数据库(比如 Oracle)中提取数据，使用各种 Pandas 方法转换或清理数据，并将转换或清理的数据加载到目标系统(CSV 或 DB)中。这篇博客的内容非常基础，可以帮助你理解 python 中 ETL 程序的流程。

![](img/a5cce58022fec5e8ca693407f757cf2e.png)

ETL 过程

我们有一个 XYZ 组织的员工数据和他们各自部门的信息，我们将尝试执行一些工资操作和其他东西，随着我们的进展，我们会知道。

现在从这里开始，我们将用编码语言交谈。好了，让我们直接跳到代码部分。

1.  **连接到源系统**

```
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine(‘oracle://scott:scott@orcl’, echo=False)
```

SQLAlchemy 是一个 Python SQL 工具包，它为我们提供了连接各种关系数据库的灵活性，在我们的例子中是 Oracle。

create_engine 是在 SQLAlchemy 下定义的一个方法，它使用参数(连接字符串)来连接数据库。

**2。从数据源提取数据**

提取在 XYZ 组织工作的雇员的数据，并执行各种转换操作来操作数据。

```
emp_df=pd.read_sql_query(‘select * from emp’,engine)
emp_df.head(10)
```

![](img/75b44f4cb0dff410955aefb1c9442a9d.png)

员工数据

提取员工各自的部门信息。

```
dept_df=pd.read_sql_query(‘select * from dept’,engine)
dept_df.head(10)
```

![](img/18611a450d941d7951303a70131ea703.png)

部门数据

read_sql_query 是一个连接数据库的 pandas 方法，该方法以**查询**和**连接字符串**作为输入参数，在数据库上触发查询，并以 pandas 数据帧格式给出结果。

**3。数据转换和清理:**我们将把这个部分分成子部分。

**第一部分。**编写一个函数，根据税板计算雇员工资的税。我们将通过各种可能的方法来解决这个问题。

**场景 1:使用 map()(T13)定义一个函数来计算薪水系列的税额**

```
def cal_taxes(sal):
	tax=0
	if sal >500 and sal <=1250:
 		tax=sal*.125
	elif sal>1250 and sal<=1700:
		tax=sal*.175
	elif sal>1700 and sal<=2500:
		tax=sal*.225
	elif sal>2500:
		tax=sal*.275
	else:
		tax=0
	return tax
```

使用 Pandas map()在系列上实现定义的功能。

```
emp_df[‘Tax’]=emp_df[‘sal’].map(cal_taxes)
emp_df.head(10)
```

![](img/a7962187a4a1ec2c6bdd636e9ec88f01.png)

带映射的函数

**场景 2:在 map()中使用 lambda 表达式**

```
#lambda expression with nested if-else
emp_df[‘tax’]=emp_df[‘sal’].map(lambda x: x*.125 if x>500 and x<=1250 else x*.175 if x>1250 and x<=1750
 	else x*.225 if x>1750 and x<2250
	 else x*.275 )
emp_df.head(10)
```

![](img/c4094ce26d17eb8687baef6ed0d27fb4.png)

使用 lambda 映射

**场景 3:通过 apply()使用 lambda 表达式**

```
#default axis of apply is axis=0 and with this argument it works exactly like map
emp_df[‘app_tax’]=emp_df[‘sal’].apply(lambda x: x*.125 if x>500 and x<=1250 
	else x*.175 if x>1250 and x<=1750 
	else x*.225 if x>1750 and x<2250
	else x*.275 )
emp_df.head(10)
```

![](img/2e5b960019ad0b127b1c66c0d43fd78f.png)

应用 lambda

**第二部分。**计算佣金/销售百分比%

首先删除 tax 列，因为我们只需要其中一项。

```
#drop syntax to drop single or multiple columns
emp_df.drop(columns=[‘tax’,’app_tax’],inplace=True)
emp_df.head(10)
```

![](img/41c9279cd9fcf37d07abc76d0d6d54b8.png)

删除列

在计算 comm/sal 百分比之前，似乎需要对数据进行一些清理，因为我们可以在 comm 列中看到 NaN 或 Null 和零，这些不需要的数据会影响计算。让我们清理数据。

```
#replace Nan or nulls or 0 in comm with their respective salary values
#means replace null in comm with value in salary of that recordemp_df[‘comm’]=emp_df[[‘sal’,’comm’]].apply(lambda x: x[0] 
	if (pd.isnull(x[1]) or int(x[1])==0) else x[1], axis=1)
```

![](img/ae7758f2538110649254275aa8b6a638.png)

零处理

现在，我们的数据被清理，让我们计算佣金/销售%

```
emp_df[‘comm_%’]=(emp_df[‘comm’]/emp_df[‘sal’])*100
emp_df.head(10)
```

![](img/1c6d0bb791637bd60b5ac08c51d11d66.png)

佣金/销售百分比

第三部分。我们可以看到，许多记录的 comm 值都是 100%,这显然不是因为性能，而是因为数据处理。在下一个需求中，我们必须将数据标记为**有效**和**无效**，并创建单独的数据集。

```
emp_df[‘Comm_Flag’]=emp_df[[‘sal’,’comm’]].apply(lambda x: ‘Invalid’ 
	 if int(x[0])==int(x[1]) else ‘Valid’,axis=1)
emp_df.head(10)
```

![](img/7870f33471bc56eb5dfb8681dc0c9369.png)

标记记录

现在，过滤数据并创建单独的数据框

```
#valid data
comm_valid=emp_df[emp_df[‘Comm_Flag’]==’Valid’]
comm_valid.head(5)
```

![](img/d6fef439af5e6a88e411965075ad3bb7.png)

有效数据集

```
#invalid values
comm_invalid=emp_df[emp_df[‘Comm_Flag’]==’Invalid’]
comm_invalid.head(10)
```

![](img/6102d6426060b39efcd2dacc747a4590.png)

无效的数据集

第四部分。分组数据

```
#calculate department wise average salary 
agg_sal=emp_df.groupby([‘deptno’])[‘sal’].mean()
print(agg_sal)
```

![](img/b5c9b77fdd521c60f0483be7858bbac3.png)

平均 sal

**第五部分。**将分组数据集与原始数据帧合并

```
#pandas merge() to combine data sets
joined_df=pd.merge(emp_df,agg_sal,on=’deptno’)
joined_df
```

![](img/1e31a3f74a73b6c86c7c281ba4adcef2.png)

组合数据集

列名看起来不酷，所以让我们重命名列以便更好地识别。

```
#rename columns to make data more meaningful
joined_df.rename(columns={‘sal_x’:’sal’,’sal_y’:’avg_sal’},inplace=True)
joined_df.drop(columns=’Comm_Flag’,inplace=True)
joined_df.sort_values(‘deptno’)
```

![](img/70f1b8f0cc23bde735df47fcb8ec6044.png)

**第六部分。**数据操作。

让我们假设我们有一个来自下游应用程序的特定要求，他们有特定的工作代码，他们希望数据是这种格式。因此，首先根据需求定义一个映射并实现代码。

```
#mapping of job codes defined below
job_map={‘MANAGER’:’MGR’,’PRESIDENT’:’Country_Head’,’CLERK’:’CLK’,
 ‘ANALYST’:’SDE2',’SALESMAN’:’Sales&Marketing’}#create a new dataframe with update job values
df=joined_df.replace({‘job’:job_map})
df
```

![](img/26377f43e22a9caf6b7d5749dbbcf181.png)

数据操作

**第七部分。**合并员工及其各自部门的数据集。

```
final=pd.merge(df,dept_df[[‘deptno’,’dname’,’loc’]],on=’deptno’,how=’inner’)
final
```

![](img/47688b9542075cd9eb27f92070c25fe1.png)

合并数据集

```
#manipulate dept names, just to get a more cleanliness
dname_map={‘RESEARCH’:’R&D’,’SALES’:’SALES’,’ACCOUNTING’:’ACCT’}
final=final.replace({‘dname’:dname_map})
final
```

![](img/ab1541bed360b2718cb20800b79a394a.png)

部门名称操作

**第八部分。**创建干净的最终数据集。

```
cleaned_df=final[[‘empno’,’ename’,’job’,’hiredate’,’sal’,’Tax’,’avg_sal’,’dname’,’loc’]]
cleaned_df
```

![](img/42163e3648bdc928fdb741005a82733b.png)

最终数据集

**4。加载到目标**

这是数据加载过程的最后一步，我们已经提取和处理了数据集，现在是时候将数据加载到 DWH/ODS，以使数据可供下游使用。

```
cleaned_df.to_sql(‘emp_dept’,con=engine,if_exists=’replace’,index=False)
```

pandas **to_sql()** 方法将数据转储到目标系统中。最后查询表并检查数据。

![](img/aef1f7c32ff9c29645927cb3b0fe000c.png)

数据加载到目标中

# **概要:**

使用 pandas 设计 ETL 管道的简单方法

了解用于建立数据库连接的 **SQLAlchemy** 包

熊猫 **read_sql_query()** 和 **to_sql()** 方法

熊猫 **map()** ， **apply()** 和 **lambda** 表达式与**嵌套 if-else**

熊猫**分组方式()**方法

使用 pandas **merge()** 合并数据集

Pandas **replace()** 方法来操作序列中的值

使用 **to_sql()** 方法将数据加载到目标

感谢所有人阅读我的博客，如果你喜欢我的内容和解释，请在媒体上关注我并分享你的反馈，这将永远帮助我们所有人提高我们的知识。

谢谢

Vivek Chaudhary