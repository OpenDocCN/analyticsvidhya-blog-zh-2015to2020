# Python 中的 SCD Type2 实现

> 原文：<https://medium.com/analytics-vidhya/scd-type2-implementation-in-python-95bb08878ce2?source=collection_archive---------0----------------------->

本文的目标是理解使用 Python 数据处理库 Pandas 实现 SCD Type2。

**什么是 SCD？**

在实施 SCD 之前，了解 SCD 是非常重要的。 **SCD** 代表**缓变尺寸**。SCD 是数据仓库(DWH)操作中最常见和最完整的概念之一。

缓变维度是数据缓慢变化的维度，而不是按时间规律变化的维度。

![](img/7730e57c3314c33849a4818b09909ede.png)

DWH 建筑图

**什么是 SCD Type2？**

实施 SCD 类型 2 方法，其中在维度表中维护历史数据。这种方法不会用新数据覆盖维度表中的旧数据，也许它会使用标志或时间戳保持以前的数据和新数据的正确版本。

所以为了更好地理解这个过程，我必须把这篇文章分成两部分。然而，实际上只需编写一个代码就可以同时处理这两种情况。

**第 1 部分:**第一次数据加载，我们有一个源表和一个空的目标表，所有的数据都是新的，需要加载到 OLAP 目标。

**第 2 部分:**满足现有数据和新记录的变化。例如，如果源中的实体发生变化，并且目标中存在特定记录，则观察到变化，并且将这些变化插入到具有更新版本的目标中，并且先前记录的版本发生变化，这被认为是历史记录。

第 1 部分和第 2 部分共同定义了 **SCD 类型 2** 。

**使用案例简介:**ABC 公司有一个年中评估流程，一些最优秀的员工获得了加薪，一些新员工也加入了进来。因此，对于加薪的现有员工，在维护历史数据的同时更新他们的工资，对于新员工，插入新记录。

很多功能方面，对吗？现在是时候为我们理解的功能编码了。好了，让我们跳到**“编码”**部分。

**第一部分。初始数据加载，目标表中没有记录**

第一步。导入所需的库和数据提取。

```
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine(‘oracle://scott:scott@orcl’, echo=False)emp_df=pd.read_sql_query(‘select * from emp_src’,engine)
emp_df.head(20)
```

![](img/50d650bf8111b8e8530ff83feadb9739.png)

源记录

```
#query target table
emp_delta_df=pd.read_sql_query(‘select * from emp_scd2’,engine)
emp_delta_df.head(20)
```

> 这是第一次加载，所以到目前为止目标表数据是空的。

```
e_fload_df=emp_df[[‘empno’,’ename’,’sal’,’deptno’]]#for first time load versioning is 1 to identify records as latest
e_fload_df[‘flag’]=1e_fload_df.head(10)
```

![](img/40ffda46eb0b5ae2efd595220236e5e7.png)

满载数据集

代理键在数据库级别处理。

```
#sequence is created in Oracle Database
create sequence seq_eid;#to insert value of sequence key, before row trigger is created
create or replace trigger trig_emp_seq
before insert on emp_scd2
for each row
begin
 if :new.EID is null
 then
 select seq_id_scd2 into :new.EID from dual;
 end if;
end;
/#insert records into database 
e_fload_df.to_sql(‘emp_scd2’,con=engine,if_exists=’append’,index=False)
```

![](img/e4ee0d75e7ab74263b7c2281b10f1eb7.png)

EMP_SCD2 表格数据

SCD 类型 2 的第 1 部分按预期工作，目标表的初始加载成功完成。

在故事的下一部分，我们将理解如何执行插入作为更新和更新操作并编写代码。

第二部分。如果存在，作为更新插入

第一步。源和目标数据提取

一个警告:我已经更新了一些记录，并从后端向源表中插入了新记录，所以请注意这些新手。

```
emp_src=pd.read_sql_query(‘select * from emp_src’,engine)
emp_src.head(20)
```

![](img/b315ec4ddfe1c44167f2bb36c9c91583.png)

更新的源表

```
emp_scd=pd.read_sql_query(‘select * from emp_scd2’,engine)
emp_scd.head(20)
```

![](img/2695e1d7776f35aa2ad17db5bb300663.png)

EMP_SCD2 目标数据

```
#rename columns is source and target for identificationemp_src=emp_src[[‘empno’,’ename’,’sal’,’deptno’]]
emp_src.rename(columns={‘empno’:’empno_src’,’sal’:’sal_src’},inplace=True)
emp_scd.rename(columns={‘empno’:’empno_tgt’,’sal’:’sal_tgt’},inplace=True) #left join both the source and target dataframes to identify new records
join_df=pd.merge(emp_src,emp_scd,left_on=’empno_src’,right_on=’empno_tgt’,how=’left’)
join_df.head(20)
```

![](img/15fa31287ddca9de5296ff6f5d877486.png)

连接的数据框架

识别新记录并为插入记录创建单独的数据框。

```
join_df[‘INS_FLAG’]=join_df[[‘empno_src’,’empno_tgt’]].apply(lambda x:’I’ if pd.isnull(x[1]) else ’N’, axis=1)ins_rec=join_df[join_df['INS_FLAG']=='I']
ins_rec=ins_rec[['empno_src','ename_x','sal_src','deptno_x']]
ins_rec.rename(columns={'empno_src':'empno','ename_x':'ename','sal_src':'sal','deptno_x':'deptno'},inplace=True)ins_df=ins_rec[['empno','ename','sal','deptno']]#flag=1 for versioning of new records
ins_df['flag']=1
ins_df.head(5)
```

![](img/c39bf934b49488a9efb361aefd3434be.png)

插入记录

```
ins_df.to_sql(‘emp_scd2’,con=engine,if_exists=’append’,index=False)
```

![](img/4f449e49e0bbaf7d5978b3a68b8e17cf.png)

EMP_SCD2 新记录

确定薪资发生变化的记录，更新其先前的标志并插入新记录。

```
join_df[‘INS_UPD_FLAG’]=join_df[[‘empno_src’,’empno_tgt’,
 ‘sal_src’,’sal_tgt’]].apply(lambda x:’UI’ if x[0]==x[1] and x[2]!=x[3] else ’N’, axis=1)ins_upd_rec=join_df[join_df[‘INS_UPD_FLAG’]==’UI’]
ins_upd_rec=ins_upd_rec[[‘empno_src’,’ename_x’,’sal_src’,’deptno_x’]]
ins_upd_rec[‘flag’]=1
ins_upd_rec.rename(columns={‘empno_src’:’empno’,’ename_x’:’ename’,’sal_src’:’sal’,’deptno_x’:’deptno’},inplace=True)
ins_upd_rec
```

![](img/71fbf8b943c9b1c3944214b1f2b03682.png)

检测到薪金变化

现在，是时候更新检测到工资变化的记录的标志了，之后将插入新的记录。

```
import sqlalchemy as sqla
from sqlalchemy.orm import sessionmaker
metadata = sqla.MetaData(bind=engine)
datatable = sqla.Table(‘emp_scd2’, metadata, autoload=True)
Session = sessionmaker(bind=engine)
session = Session()#Flag updation process to mark them as previous records
for ind, row in ins_upd_rec.iterrows():
    upd=sqla.sql.update(datatable)\
        .values({'flag':0})\
        .where (sqla.and_(datatable.c.empno==row.empno))
    session.execute(upd)
session.flush()
session.commit()
```

![](img/f091ea6d691e260003ddb110b8898f18.png)

EMP_SCD2 标志已更新

```
ins_upd_rec.to_sql(‘emp_scd2’,con=engine,if_exists=’append’,index=False)
```

![](img/78020b7dec25087b790a108c5f1d5c9d.png)

EMP_SCD2 最终数据

这是用 Python 实现的 SCD Type2，为了更好地理解流程和过程，它被分成了两个部分。

# 总结:

理解 SCD 类型 2

用例实现分为两个部分

所需的库和数据提取

初始数据加载(满载)

数据库序列和触发器

如果存在实施，则作为更新插入

使用 sql_alchemy 库插入和更新记录

感谢所有人阅读我的博客，如果你喜欢我的内容和解释，请在媒体上关注我并分享你的反馈，这将永远帮助我们所有人提高我们的知识。

谢谢

Vivek Chaudhary