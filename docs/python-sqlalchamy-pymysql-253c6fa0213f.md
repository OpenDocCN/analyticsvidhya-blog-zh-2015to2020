# Python — SQLAlchamy — PyMySql

> 原文：<https://medium.com/analytics-vidhya/python-sqlalchamy-pymysql-253c6fa0213f?source=collection_archive---------4----------------------->

SQLAlchemy 支持使用方言/引擎的多个数据库客户机

支持方言的完整列表[点击此处](https://docs.sqlalchemy.org/en/13/dialects/)

`create_engine`接受数据源名称(dsn)作为参数

# dsn 的组成部分

`scheme://username:password@host/database?params`

由于我们使用的 MySQL 方案是`mysql`

**注意:-** 在我们的例子中，主机是`mysql`，因为它运行在 docker 中，通常是 127.0.0.1 或 localhost

*engine = create _ engine(' MySQL+pymysql://root:password @ MySQL/stack overflow _ survey _ 1 ')*

# 获得连接

conn = engine.connect()

# 将示例查询转换为数据框架

我们可以使用熊猫`read_sql`函数将结果直接转换成数据帧，它接受 2 个强制参数

1.  询问
2.  数据库连接对象

```
query **=** 'SELECT * FROM responses limit 5;';pd.read_sql(query, conn,index_col**=**'id')
```

# ORM

对象关系映射是一种使用面向对象编程语言在不兼容类型系统之间转换数据的技术。

在数据库管理系统中，它将查询结果转换成对象

SQLAlchemy 支持开箱即用的 ORM

# 资料组

我们使用 2018 年至 2020 年堆栈溢出调查的数据集。

# 数据库模式

![](img/3f0d34736c42ae516e1b4189a47b0ae1.png)

# 将架构转换为 ORM

> *Base = declarative _ Base()
> Base . metadata . bind = engine
> Base . metadata . create _ all()
> Session = Session maker(bind = engine)
> ses = Session()*

我们在 python 中创建的用于访问模式的类

> *##响应表
> 类响应(基本):
> _ _ tablename _ _ = " Responses "
> _ _ table _ args _ _ = { ' extend _ existing ':True }*
> 
> *Id = Column(Integer，primary _ key = True)
> Age = Column(Integer)
> Hobbyist = Column(Boolean)
> Country = Column(String)
> Gender = Column(String)
> Education = Column(String)
> Employment = Column(String)
> years code = Column(Integer)
> YearsCode pro = Column(Integer)
> CreatedAt = Column(DateTime)
> update dat = Column(dat*

**在不编写查询的情况下获取 5 条记录的示例查询**

> *query = ses.query(Response)。limit(5)
> print("query:"，query，" \ n \ n ")
> for response in query . all():
> print(response。Id，回复。年龄，反应。爱好者，回应。国家，回应。性别)*

**这会自己创建查询，并在我们打印时从数据库返回数据**

```
query :  SELECT responses.`Id` AS `responses_Id`, responses.`Age` AS `responses_Age`, responses.`Hobbyist` AS `responses_Hobbyist`, responses.`Country` AS `responses_Country`, responses.`Gender` AS `responses_Gender`, responses.`Education` AS `responses_Education`, responses.`Employment` AS `responses_Employment`, responses.`YearsCode` AS `responses_YearsCode`, responses.`YearsCodePro` AS `responses_YearsCodePro`, responses.`CreatedAt` AS `responses_CreatedAt`, responses.`UpdatedAt` AS `responses_UpdatedAt` 
FROM responses 
 LIMIT %(param_1)s 

1 None True germany male
2 None False united kingdom none
3 None True russian federation none
4 25 True albania male
5 31 True united states male
```

**选择国家/地区，按国家/地区分组计数，按计数顺序取前 5 个结果**

> *query = ses.query(响应。Country，func . count(" * ")\
> 。group_by(响应。国家)\
> 。order_by(func.count('* ')。desc())。限制(5)
> print("query:"，query，" \ n \ n ")
> for RES in query . all():
> print(RES)*

**这会自己创建查询，并在我们打印时从数据库返回数据**

```
query :  SELECT responses.`Country` AS `responses_Country`, count(%(count_2)s) AS count_1 
FROM responses GROUP BY responses.`Country` ORDER BY count(%(count_3)s) DESC 
 LIMIT %(param_1)s 

('united states', 53727)
('india', 31185)
('germany', 16215)
('united kingdom', 15854)
('canada', 8979) 
```

我们可以用 orm 做更多的事情，比如定义关系和自动获取关系等等。

在数据科学中，因为我们需要使用复杂的查询，所以 ORM 可能不是非常有用

# 使用 ORM 的优势

1.  它是数据库不可知的，为支持的方言提供编程接口。
2.  它防止 SQL 注入
3.  易于可视化和维护代码

# 不足之处

1.  相对来说，它比原始 SQL 要慢
2.  难以编写复杂的查询
3.  需要为一些聚合和分析函数编写原始 SQL