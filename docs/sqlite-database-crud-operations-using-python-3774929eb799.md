# 使用 Python 的 SQLite 数据库“CRUD 操作”。

> 原文：<https://medium.com/analytics-vidhya/sqlite-database-crud-operations-using-python-3774929eb799?source=collection_archive---------3----------------------->

![](img/f581dad96479b96e8f1f265250fc1de5.png)

写这篇文章的目的是为了对后端开发好奇的初学者，或者想用服务器端编程语言学习数据库技术的前端开发人员。

众所周知，像 python、java 和其他许多服务器端语言对于后台开发人员来说是不够的，甚至后端开发人员也需要更多的数据库技术知识。

所以我们先从 python 的 SQLite 数据库基础说起。

我选择 SQLite 数据库只是一个选项，人们可以将相同的知识应用于任何其他数据库，如 Mysql 和 Oracle 或任何其他数据库。数据库技术最好的部分是所有的数据库都非常相似，因为 SQL 概念接受很少的新数据库。

> 我们没有使用任何 python 包来简化我们学习这四个基本操作的工作。

## **积垢**

*   c:创建
*   r:阅读
*   u:更新
*   删除

## 创建:

在表中插入或创建新记录。因此，让我们在 Sqlite 数据库中创建一个示例表。

```
# Creating table into database!!!import sqlite3# Connect to sqlite database
conn = sqlite3.connect(**'students.db'**)
# cursor object
cursor = conn.cursor()
# drop query
cursor.execute(**"DROP TABLE IF EXISTS STUDENT"**)# create query
query = **"""CREATE TABLE STUDENT(
        ID INT PRIMARY KEY NOT NULL,
        NAME CHAR(20) NOT NULL, 
        ROLL CHAR(20), 
        ADDRESS CHAR(50), 
        CLASS CHAR(20) )"""** cursor.execute(query)# commit and close
conn.commit()
conn.close()
```

***conn = sqlite3 . connect(' students . DB ')***是连接方法，对于 SQLite DB 来说非常简单，但对于不同的数据库会有所不同。

***cursor.execute()*** 方法执行 sqlite 查询。

```
**"""
CREATE TABLE table_name (
   column name datatype properity,
   ...
   ...
);
"""**
```

以上语法可以与查询进行映射，创建查询有三个主要属性"**列名数据类型属性**"。

在每个数据库操作之后，我们应该添加一个 commit 和 close DB 操作。

## 插入:

```
import sqlite3conn = sqlite3.connect(**'students.db'**)conn.execute(**"INSERT INTO STUDENT (ID,NAME,ROLL,ADDRESS,CLASS) "
             "VALUES (1, 'John', '001', 'Bangalore', '10th')"**)conn.execute(**"INSERT INTO STUDENT (ID,NAME,ROLL,ADDRESS,CLASS) "
             "VALUES (2, 'Naren', '002', 'Hyd', '12th')"**)conn.commit()
conn.close()
```

上面的查询语法是硬编码数据插入，但是当我们有来自外部输入的数据时，我们可以这样修改语法。

```
query = (**'INSERT INTO STUDENT (ID,NAME,ROLL,ADDRESS,CLASS) '
         'VALUES (:ID, :NAME, :ROLL, :ADDRESS, :CLASS);'**)params = {
        **'ID'**: 3,
        **'NAME'**: **'Jax'**,
        **'ROLL'**: **'003'**,
        **'ADDRESS'**: **'Delhi'**,
        **'CLASS'**: **'9th'** }conn.execute(query, params)
```

![](img/139567e96413a79a2507921649c9453a.png)

[https://sqliteonline.com/](https://sqliteonline.com/)是一个很好的在线平台，可以用来执行数据库操作，不需要安装任何额外的软件。

## 阅读:

这是一个重要的操作，因为这属于选择查询，在从数据库获取记录时有更多的真实性。对于多表数据库，有时这种操作会非常棘手，这里有几个选择操作的例子。

```
import sqlite3conn = sqlite3.connect(**'students.db'**)
cursor = conn.execute(**"SELECT * from STUDENT"**)
print(cursor.fetchall())conn.close()
```

获取所有数据的最简单方法是“从表名中选择”

```
**SELECT column1, column2, columnN FROM table_name;**
```

我们可以只提到那些需要的列名，如果不需要提取所有数据，那么提到列名总是一个好的做法。

```
**SELECT column1, column2, columnN FROM table_name WHERE column_name = value;**
```

where 子句返回记录将与值匹配的特定行。

## 更新:

更新是改变现有的记录，更新的简单规则是使用最佳方法到达记录并改变它。

```
import sqlite3conn = sqlite3.connect(**'students.db'**)
conn.execute(**"UPDATE STUDENT set ROLL = 005 where ID = 1"**)
conn.commit()cursor = conn.execute(**"SELECT * from STUDENT"**)
print(cursor.fetchall())conn.close()
```

> 注意:为什么我说使用最佳方法，因为不仅在更新的情况下，在所有的数据库操作中，我们应该避免读取或获取不必要的记录，这将节省查询时间，并将提高任何应用程序的整体系统性能。

## 删除:

从表中删除任何记录都是一个删除操作，下面的代码显示了删除查询示例。

```
import sqlite3conn = sqlite3.connect(**'students.db'**)conn.execute(**"DELETE from STUDENT where ID = 2;"**)
conn.commit()cursor = conn.execute(**"SELECT * from STUDENT"**)
print(cursor.fetchall())conn.close()
```

这些是使用 python 对 SQLite 数据库进行的基本 CRUD 操作。

源代码可以在 GitHub 上找到[点击这里。](https://gitlab.com/NarendraH/narendrablogssource/-/tree/master/Python_Sqlite_CRUD_Operations)

> 直接开始开发应用程序总是好的，我们应该学习这些应用程序所需的基础主干技术。
> 
> 如果您对 CRUD 操作的数据库操作有基本的了解，那么使用任何 python SQL 库都可以轻松实现。

谢谢大家！！！快乐编码。