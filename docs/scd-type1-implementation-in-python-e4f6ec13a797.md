# Python 中的 SCD Type1 实现

> 原文：<https://medium.com/analytics-vidhya/scd-type1-implementation-in-python-e4f6ec13a797?source=collection_archive---------1----------------------->

本文的目标是理解使用 Python 数据处理库 Pandas 实现 SCD Type1。

在数据库世界中，有不同类型的数据建模:用于交易系统或 OLTP 系统的**交易建模** (TM)或**关系建模** (RM)，以及用于 OLAP 或 DWH 系统及其他几个系统的**维度建模**。除了处理关系实体的方式之外，这两种建模几乎相同，例如，TM/RM 通过主键和外键建立表关系，这使得这种设计适合 OLTP 系统，而在 OLAP 系统中，键最少，这使得系统适合更快地查询 BI 和报告。所以记住这篇文章的范围，我不打算深入数据建模的细节。

以下是 OLAP 维度模型数据库架构的外观:

![](img/7730e57c3314c33849a4818b09909ede.png)

OLAP 图式

中心表是**事实表**，它包含所有的事实或可测量的或数字的实体，而周围的表是**维度表**，其中包含支持数据，从而为事实表的数字数据提供意义。

**什么是 SCD？**

在实施 SCD 之前，了解 SCD 是非常重要的。 **SCD** 代表**缓变尺寸**。SCD 是数据仓库(DWH)操作中最常见和最完整的概念之一。

缓变维度是数据缓慢变化的维度，而不是按时间规律变化的维度。

**什么是 SCD Type1？**

在不需要在维度表中存储历史数据的情况下，实施 SCD 类型 1 方法。此方法用新数据覆盖维度表中的旧数据。

所以为了更好地理解这个过程，我必须把这篇文章分成两部分。然而，实际上只需编写一个代码就可以同时处理这两种情况。

**第 1 部分:**第一次加载数据，我们有一个源表和一个空的目标表，所有的数据都是新的，需要加载到 OLAP 目标。

**第 2 部分:**迎合现有数据和新记录的变化。例如，如果源中的实体发生变化，并且目标中存在特定的记录，则观察到变化并覆盖到目标(更新操作)，并且如果有任何新的记录符合插入条件，则进行插入。简单来说**三角负荷**。

第 1 部分和第 2 部分共同定义了 SCD 类型 1。

**使用案例简介:**ABC 公司有一个年中考核流程，一些最优秀的员工获得了加薪，一些新员工也加入其中。因此，对于加薪的现有员工，在不维护历史数据的情况下更新他们的工资，对于新员工，插入新记录。

很多功能方面，对吗？现在是时候为我们理解的功能编码了。好了，让我们跳到**“编码”**部分。

**第一部分。初始数据加载，目标表中没有记录**

第一步。导入所需的库和数据提取。

```
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine(‘oracle://scott:scott@orcl’, echo=False)
emp_df=pd.read_sql_query(‘select * from emp_src’,engine)
emp_df.head(20)
```

![](img/040f522f553419e8b26f7af7c8bc13b0.png)

源数据

```
# querying target table dataemp_delta_df=pd.read_sql_query(‘select * from emp_scd1’,engine)
emp_delta_df.head(20)
```

> 这是第一次加载，所以到目前为止目标表数据是空的。

```
e_src=emp_df[[‘empno’, ‘ename’, ‘sal’, ‘comm’, ‘deptno’]]#rename columns of src dataset
e_src.rename(columns={‘empno’:’EMPNO_SRC’, ‘ename’:’ENAME’, ‘sal’:’SAL_SRC’, ‘comm’:’COMM’, ‘deptno’:’DEPTNO’},inplace=True)
e_src.head(10)
```

![](img/1b8ac8e972707287b8157f3ceef8abae.png)

源数据

```
#rename columns of target empty dataframe
e_tgt=emp_delta_df[[‘empno’, ‘ename’, ‘sal’, ‘deptno’]]
e_tgt.rename(columns={‘empno’:’EMPNO_TGT’, ‘ename’:’ENAME’, ‘sal’:’SAL_TGT’, 
 ‘comm’:’COMM’, ‘deptno’:’DEPTNO’},inplace=True) 
```

第二步。合并源数据集和目标数据集

数据集与左外部连接连接，其中源充当左外部连接，原因是我们还必须获取新记录。

```
#merge datasets
joined_df=pd.merge(e_src,e_tgt,left_on=’EMPNO_SRC’,right_on=’EMPNO_TGT’,how=’left’)
joined_df
```

![](img/3c512746621209cb457b0b298a2f1f8a.png)

合并数据集

第三步。为插入和更新标记记录

比较来自源和目标数据框的 empno 值。如果 empno 值相等，则丢弃这些记录以插入到最终表中。
首先，我们将标记插入记录。

```
joined_df[‘I_U_FLAG’]=joined_df[[‘EMPNO_SRC’,’EMPNO_TGT’]].apply(lambda x: ‘Y’ 
 if (x[0]!=x[1] or pd.isnull(x[1])) else ‘N’,axis=1)
joined_df.head(20)
```

![](img/5883fa4aeea0fef0ce2788fd85fbc741.png)

记录标记

第四步。准备最终数据。

```
final_df=joined_df[[‘EMPNO_SRC’,’ENAME_x’,’SAL_SRC’,’DEPTNO_x’]]final_df=final_df.rename(columns={'EMPNO_SRC':'EMPNO','ENAME_x':'ENAME','SAL_SRC':'SAL','DEPTNO_x':'DEPTNO'})final_df.head(20)
```

![](img/9bf0ad7375dd3a943d9840d15e0df9a9.png)

最终数据

将数据插入目标表 emp_scd1

```
final_df.to_sql(‘emp_scd1’,con=engine,if_exists=’append’,index=False)
```

![](img/6a286d5737906c912bc635af43884b5e.png)

EMP_SCD1 表格数据

SCD 类型 1 的第 1 部分按预期工作，目标表的初始加载成功完成。

在故事的下一部分，我们将理解如何执行 Insert else update 操作并编写代码。

第二部分。如果存在，插入 Else 更新

第一步。源和目标数据提取

一个警告:我已经更新了一些记录，并从后端向源表中插入了新记录，所以请注意这些新手。

```
emp1=pd.read_sql_query(‘select * from emp_src’,engine)
emp1.head(20)
```

![](img/cf6354735df8a09baf59b8c380c3cafe.png)

更新的源表

```
emp1_df=emp1[[‘empno’,’ename’,’sal’,’deptno’]]
emp1_df.rename(columns={‘empno’:’empno_src’,’sal’:’sal_src’},inplace=True)
emp1_df.head(20)
```

![](img/e6fd105ca770f138c7449e0828184884.png)

重命名的列

```
#load data from source table
emp2=pd.read_sql_query(‘select * from emp_scd1’,engine)
emp2.rename(columns={'empno':'empno_tgt','sal':'sal_tgt'},inplace=True)
emp2.head(10)
```

![](img/916edf85fa89d2447263846d3aa6e94f.png)

目标表数据

第二步。合并源和目标

同样，该连接将是左外连接，以从源表中获取匹配和不匹配的数据。

```
join_df=pd.merge(emp1_df,emp2,left_on=’empno_src’,right_on=’empno_tgt’,how=’left’)
join_df.head(20)
```

![](img/1df09a7d3a08c2a65b3e1e02a02d8115.png)

合并数据集

第三步。为插入和更新标记记录

```
#Insert Flag
join_df[‘INS_FLAG’]=join_df[[‘empno_src’,’empno_tgt’,’sal_src’,’sal_tgt’]].apply(lambda x:
 ‘I’ if pd.isnull(x[1]) else ’N’, axis=1)#Update Flag
join_df[‘UPD_FLAG’]=join_df[[‘empno_src’,’empno_tgt’,’sal_src’,’sal_tgt’]].apply(lambda x:
 ‘U’ if (x[0]==x[1] and x[2]!=x[3])
 else ‘N’,axis=1)join_df.head(20)
```

![](img/d40088ae0ccca86ca88b22a7e2e8b577.png)

标记为插入/更新的记录

第四步。准备插入和更新数据集

```
#create seperate dataset for New recordsins_rec=join_df[join_df[‘INS_FLAG’]==’I’]
ins_upd=ins_rec[[‘empno_src’,’ename_x’,’sal_src’,’deptno_x’]]
ins_upd.rename(columns={‘empno_src’:’empno’,’ename_x’:’ename’,’sal_src’:’sal’,’deptno_x’:’deptno’},inplace=True)
```

![](img/ee1c05020d7eb645a62892d9247eff32.png)

插入记录

```
#Create seperate dataset for Updated recordsupd_rec=join_df[join_df[‘UPD_FLAG’]==’U’]
upd_df=upd_rec[[‘empno_src’,’ename_x’,’sal_src’,’deptno_x’]]
upd_df.rename(columns={‘empno_src’:’empno’,’ename_x’:’ename’,’sal_src’:’sal’,’deptno_x’:’deptno’},inplace=True)
upd_df.head(5)
```

![](img/29d171359aac0da0b1fa2b717e072f86.png)

更新记录

向目标表插入新记录:

```
ins_upd.to_sql(‘emp_scd1’,engine,if_exists=’append’,index=False)
```

将新记录更新到目标中:

这一步不像简单的插入那样简单。让我们看看如何更新记录。

```
#update records in target tableimport sqlalchemy as sqla
metadata = sqla.MetaData(bind=engine)
datatable = sqla.Table(‘emp_scd1’, metadata, autoload=True)#loop over the dataframe items to update values in target
for ind, row in upd_df.iterrows():
	upd=sqla.sql.update(datatable)\
		.values({‘sal’:row.sal})\
		.where (sqla.and_(datatable.c.empno==row.empno))
	session.execute(upd)
session.flush()
session.commit()
```

预期产出:

员工编号 **7369 和 7900** 工资在目标中更新为 **1190 和 1350** 。Empno **7743、7745 和 7744** 被标记为插入。

**元数据**:是表对象及其相关模式结构的集合。SQL Alchemy 中的数据库表链接到一个元数据对象。该表会将其自身添加到元数据中。表对象本身存储在 Metadata.tables 字典中。

让我们查询该表并检查输出:

![](img/b6229f2b43ea3396899a0386777646dc.png)

按预期在目标中插入 else 更新记录

这是用 Python 实现的 SCD Type1，为了更好地理解流程和过程，它被分成了两个部分。

# **总结:**

对 OLAP、维度建模和图式有基本的了解

理解 SCD 及其类型 Type1

用例实现分为两个部分

所需的库和数据提取

初始数据加载(满载)

如果存在实施，则插入 Else 更新

使用 sql_alchemy 库插入和更新记录

感谢所有人阅读我的博客，如果你喜欢我的内容和解释，请在媒体上关注我并分享你的反馈，这将永远帮助我们所有人提高我们的知识。

谢谢

Vivek Chaudhary