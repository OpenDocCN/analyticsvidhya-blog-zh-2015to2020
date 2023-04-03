# rails 6 中有趣的活动记录查询方法

> 原文：<https://medium.com/analytics-vidhya/fascinating-active-records-query-methods-in-rails-6-63d431dbfdd6?source=collection_archive---------18----------------------->

你好，

我最近偶然发现了一些迷人的 rails 魔术(方法),我想和大家分享一下..

![](img/3b92dc5a543a6e9cdb54db94176a7509.png)

查询方法

1.  **注解:**

好的。让我们从注释 rails 6 中新引入的活动记录方法开始。

![](img/c384379583404c6fcedcd9ba3957d2f4.png)

给…作注解

有没有想过给 SQL 查询加注释。这样一个新手可以很容易地理解你所做的复杂查询。

那么注释就是做到这一点的魔法。

**例句:**

```
[User](https://apidock.com/rails/User)**.**[annotate](https://apidock.com/rails/ActiveRecord/QueryMethods/annotate)**("**selecting user names**").**[select](https://apidock.com/rails/ActiveRecord/QueryMethods/select)**(**:name**)**
*# SELECT "users"."name" FROM "users" /* selecting user names */*

[User](https://apidock.com/rails/User)**.**[annotate](https://apidock.com/rails/ActiveRecord/QueryMethods/annotate)**("**selecting**",** **"**user**",** **"**names**").**[select](https://apidock.com/rails/ActiveRecord/QueryMethods/select)**(**:name**)**
*# SELECT "users"."name" FROM "users" /* selecting */ /* user */ /* names */*
```

SQL 块注释分隔符“/*”和“*/”将自动添加。

**2。创建方式:**

它的工作是设置从一个关系对象创建新记录时使用的属性。

![](img/569b66d2485e20b6d85c998c831aeeae.png)

创造

**例如:**

```
users = User.where(name: 'Oscar')
users.new.name *# => 'Oscar'*

users = users.create_with(name: 'DHH')
users.new.name *# => 'DHH'*
```

还可以将 nil 传递给 create_with 来重置属性。

**3。扩展:**

用于通过提供的模块或块用附加方法扩展范围。

返回的对象是一个关系，可以进一步扩展。

> 语法:扩展(*模块和块)

![](img/89b3afcc18c37858f4641482987aa4ee.png)

延伸

**例如:**

创建一个名为分页的模块。

```
**module Pagination**
  **def** **page**(number)
    *# pagination code goes here*
  **end**
**end**
```

scope = model . all . extending(Pagination)
scope . page(params[:page])

使用块:

```
scope = Model.all.extending **do**
  **def** **page**(number)
    *# pagination code goes here*
  **end**
**end**
scope.page(params[:page])
```

**4。提取关联:**

这是 rails 6 中新引入的方法。它的任务是从关系中提取一个名为`association`的。首先预加载命名关联，然后从关系中收集单个关联记录。

![](img/1ca5678487543759c5f49b703c73f043.png)

提取

**例如:**

```
account.memberships.extract_associated(:user)
*# => Returns collection of User records*
```

这也可以写成这样:

```
account**.memberships.preload**(:user)**.collect**(&:user)
```

**5。优化器提示:**

如果想在数据库查询中使用提示，rails 6 提供了一个名为 optimizer_hints 的新方法。

例如:

指定要在 SELECT 语句中使用的优化程序提示。

示例(对于 MySQL):

```
Topic.optimizer_hints("MAX_EXECUTION_TIME(50000)", "NO_INDEX_MERGE(topics)")
*# SELECT /*+ MAX_EXECUTION_TIME(50000) NO_INDEX_MERGE(topics) */ `topics`.* FROM `topics`*
```

示例(对于带有 pg_hint_plan 的 PostgreSQL):

```
Topic.optimizer_hints("SeqScan(topics)", "Parallel(topics 8)")
*# SELECT /*+ SeqScan(topics) Parallel(topics 8) */ "topics".* FROM "topics"*
```