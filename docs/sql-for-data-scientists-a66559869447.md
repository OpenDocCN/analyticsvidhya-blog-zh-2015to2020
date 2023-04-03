# 面向数据科学家的 SQL

> 原文：<https://medium.com/analytics-vidhya/sql-for-data-scientists-a66559869447?source=collection_archive---------30----------------------->

在本文中，我将通过解决问题来介绍几个重要的 SQL 函数——希望本文能够帮助初学者，或者作为准备 SQL 筛选的指南。这里解决的问题是基本的，这里描述的每个问题都将包含模拟数据，供您创建表和运行/处理查询。我使用 DB Fiddle MySQL v8.0 进行除 pivot 之外的所有查询，pivot 需要 MS SQL，您可以使用 SQLLite online 和 MSSQL editor 来处理 pivot 问题，或者使用类似的在线平台来创建表格和运行查询，这样您将有机会了解查询的确切情况，并准备替代解决方案，还可以编辑数据来测试不同的用例。

这里提出的每一个问题都可以用几种不同的方法来解决，我鼓励你用不同的方法来解决它，并且寻求最优化。这里的目标是让读者在提供模拟数据的同时了解问题。

**问题:**给定销售表，其中有多行客户的购买明细。编写一个查询，返回一周中每一天每位客户的购物总额。**用过:**枢

**数据:**

**查询:**

```
select Customer, Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday from
(
 select Customer,Units, PurchaseDay from PivotsTable
 ) as src
 pivot
 (
 Count(Units) for PurchaseDay in ([Monday],[Tuesday],[Wednesday],[Thursday],[Friday],[Saturday],[Sunday])
 ) as p
```

通常，只包括您需要出现在最终结果中的列:

```
Select pivoted data from
( select column you need to be pivoted + columns you need in the output) as source
pivot
(units to be displayed in the pivoted c) as p
```

如果你运行下面的查询，会有什么结果呢？

```
select Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday from
(
 select Units, PurchaseDay from PivotsTable
 ) as src
 pivot
 (
 Sum(Units) for PurchaseDay in ([Monday],[Tuesday],[Wednesday],[Thursday],[Friday],[Saturday],[Sunday])
 ) as p
```

**问题**:给定数据对，找出唯一的数据对？下表中的每一行都可以被认为是社交媒体连接(follower，following)，问题是显示连接存在的唯一对。同样的问题可以被框架化为寻找唯一的飞行路线-具有 3 行 MSP - > ATL 和 ATL - > MSP，MSP - > EWR 应该导致 MSP - > ATL，MSP - > EWR。**已用**:存在于何处

**数据:**

**查询**:

```
select Follower, Following from Symmetric S1 WHERE EXISTS 
(SELECT * FROM Symmetric S2 where S1.Follower = S2.Following and S2.Follower = S1.Following and S1.Follower > S1.Following)
UNION
select Follower, Following from Symmetric S1 WHERE NOT EXISTS 
(SELECT * FROM Symmetric S2 where S1.Follower = S2.Following and S2.Follower = S1.Following)
```

第一部分将返回存在双向关系的唯一对，而第二部分将给出只有单向关系的对，然后使用 UNION 将它们连接起来。

**问题**:求各系学生的最高分。每一行对应于每个系的几个学生的分数。**使用:**视窗

**数据:**

**查询:**

```
WITH CTE as(
select Department, Student, Score, dense_rank() over(Partition by Department Order by Score desc) as RankNum from College)
select Department, Student, Score from CTE where RankNum = 1
```

如何找到每个系排名第二的学生，以及从上面的查询中删除按系划分的结果是什么？还有，探究一下 RANK，DENSE_RANK，ROW_NUMBER 的区别。

```
WITH CTE as(
select Department, Student, Score, dense_rank() over(Order by Score desc) as RankNum from College)select Department, Student, Score from CTE where RankNum = 1
```

**问题**:交换行。每行都有一个 Id 和名称。**使用:**格当，内联接

**数据:**

**查询:**

```
select S1.* from School S1 inner join(
SELECT 
    CASE 
        WHEN((SELECT MAX(Id) FROM School)%2 = 1) AND id = (SELECT MAX(Id) FROM School) THEN Id
        WHEN id%2 = 1 THEN id + 1
        ELSE id - 1
    END AS Ordered_Id, row_number() over() as rn
FROM School) S2
on S1.Id = S2.Ordered_Id 
order by S2.rn
```

如果您删除 row_number()并运行下面的查询，您认为结果为什么会发生变化？

```
select S1.* from School S1 inner join(
SELECT 
 CASE 
 WHEN((SELECT MAX(Id) FROM School)%2 = 1) AND id = (SELECT MAX(Id) FROM School) THEN Id
 WHEN id%2 = 1 THEN id + 1
 ELSE id — 1
 END AS Ordered_Id
FROM School) S2
on S1.Id = S2.Ordered_Id
```

**问题**:查找在特定日期/日子登录的用户。每一行都有 Id、登录日期，并且表可以有重复的行。**用过:**在

**数据:**

**查询:**

```
select id from logins where LoginDate = '2020-07-09' and id in (Select id from logins where LoginDate = '2020-07-08')
```

找到今天登录而不是昨天登录的 id 列表怎么样——您可以探索不同的日期函数来解决与日期相关的问题。

**问题**:查找连续三天登录的用户**使用:** CTE，Windows

```
with NoDups as (select distinct * from Logins)select id from 
(select id, row_number() over(partition by Id) as rn from NoDups) A
where A.rn = 3
```

**问题**:给定两个表，一个是用户订单，另一个是类别信息。编写一个查询来返回每个类别中的订单，包括类别，即使没有订单。**已用:**连接

**数据:**

**查询:**

```
select c.Name as Category, coalesce(sum(o.Units),0) as TotalUnits from Categories c left join Orders o
on o.CategoryId = c.Id
group by c.name
```

当您使用内部连接运行相同的查询时，请检查不同之处:

```
select c.Name as Category, sum(o.Units) as TotalUnits from Orders o inner join Categories c
on o.CategoryId = c.Id
group by c.name
```

**问题**:求每月登录次数的百分比变化。同样的问题可以归结为保留率。**常用:**滞后，超前

**数据:**

## 查询:

```
with CTE AS (select *, lag(LoginCounts, 1) over() as PrevMonthCount from Logins)
SELECT MonthName, (LoginCounts — PrevMonthCount) * 100/ PrevMonthCount as RetentionPercent FROM CTE
```

感谢您的阅读。请报告任何错误/建议。