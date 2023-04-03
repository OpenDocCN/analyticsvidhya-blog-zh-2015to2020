# 代码糖果:优化 SQL 的策略

> 原文：<https://medium.com/analytics-vidhya/code-candy-strategies-to-optimize-sql-f0c907e3ca7b?source=collection_archive---------23----------------------->

![](img/3595c4e2ba6607b4ffca73c9d428c4a5.png)

选择正确的策略来构建 SQL 查询

数据是新千年的石油。幸运的是，对我们程序员来说，数据的提取不需要建立巨大的石油钻井和雇佣高薪专家，而是使用一些技术工具和可以通过实践获得的正确技能。

大多数大型 IT 系统，如银行、医疗保健和医院，都拥有海量数据，并使用关系数据库进行日常运营。在使用昂贵的数据智能系统提供的方便的数据提取特性之前，任何使用关系数据库的人肯定会选择 SQL 作为从这些数据库中提取数据的第一个工具。

出生于 1945 年的 boomer SQL 仍然被认为是 2019 年千禧年世界的五大数据科学技能之一。如今，一个 SQL 分析师专家一天可以赚 1000 美元，然而，SQL 很容易掌握，几乎所有受过训练的程序员都时不时地尝试一下。

虽然像“永远不要使用 select *”和“尽可能使用 group by”这样的谚语并不是没有听说过，但是下面这些直观的、相对来说没有记录的技巧对于编写快速和可伸缩的查询来说还是很有用的。

## Case 语句，一个方便的数据分组工具

在查询中使用 CASE 语句是在简单易读的查询中对数据进行分组的好方法。下面的查询是使用案例查询查找每个季度总业务成本的便捷方法。

```
SELECT SUM(CASE WHEN season = ‘Winter’ THEN total_cost end) as Winter_total,
SUM(CASE WHEN season = ‘Summer’ THEN total_cost end) as Summer_total,
SUM(CASE WHEN season = ‘Spring’ THEN total_cost end) as Spring_total,
SUM(CASE WHEN season = ‘Fall’ THEN total_cost end) as Spring_total,
SUM(CASE WHEN season = ‘All Year’ THEN total_cost end) as Spring_total
FROM (
select season, sum(supply * cost_per_unit) total_cost
from fruit_imports
group by season
 ) a
```

## **UNION 代替 OR**

在一个 SQL 查询中使用 OR 来选择列被证明是低效的，因为这个查询没有使用表索引，并且需要很长时间来执行。使用 UNION 组合两个 select 语句的结果，使用索引并减少查询提取时间。

`SELECT * FROM TABLE WHERE COLUMN_1 = 'value'` `OR COLUMN_2 = 'value'`

另一方面，下面的查询运行得更快。

`SELECT * FROM TABLE WHERE COLUMN_1 = 'value'`

`UNION`

`SELECT * FROM TABLE WHERE COLUMN_2 = 'value'`

## 相关子查询

SQL 语句通常试图找到常见英语问题的答案。"你能找出哪些部门的收入高于平均水平吗？"“今年有哪个部门的收入甚至低于最低收入数字吗？

当我们阅读这些问题时，它们看起来就像是在公园里散步，但是将它们转化为 SQL 是另一回事。

让我们从一个简单的查询开始，查找工资高于平均水平的雇员。

下面的 SQL 使用一个子查询将问题分成两部分，第一个查询挑选出雇员的姓名和工资，第二个查询挑选出一组工资高于雇员平均工资的雇员。这种编写查询的方式不仅使它更接近英语，而且实际上比使用连接为中等大小的表构建 SQL 查询要快得多。

`SELECT Name,Salary FROM Employees WHERE Salary > (Select(AVG(Salary)FROM Employees`

现在是一个新的查询，我们正在寻找雇员超过 40 人的部门

`SELECT department FROM employees WHERE 40 < (select (count(emp_id) from employees e WHERE e.department = d.department Group by department`

这难道不是编写查询的好方法吗？编写子查询有助于将问题分解成更短的块，并使查询可读，尽管需要记住，在上面的两个示例中，查询是针对数据库中的每个记录执行的，可能不适合巨大的数据块。

希望这一分钟的阅读能让您对编写更好的 SQL 查询有所了解。请记住这些代码，如果您对改进 SQL 查询有任何建议，请告诉我。