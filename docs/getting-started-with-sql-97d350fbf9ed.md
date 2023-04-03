# SQL 入门

> 原文：<https://medium.com/analytics-vidhya/getting-started-with-sql-97d350fbf9ed?source=collection_archive---------11----------------------->

如何使用基本命令

如果数据科学家真的是世界上最性感的工作，那么 SQL 是绝对不性感的工具，你可以用它来完成你的大部分工作。根据工作描述，SQL 甚至比 Python 和 Java 等其他流行的编程语言更受欢迎。

**TL，DR** —转到最后的基本命令汇总表

![](img/61aa5fbab773b46daa092e9193891119.png)

什么？由[里斯·肯特什](https://unsplash.com/@rhyskentish?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

## 什么是 SQL

SQL 是与数据库交互最常用的语言。为什么？它相对容易理解，并允许我们直接访问数据的存储位置。

SQL 有两种不同的发音方式。你可以单独说出字母“S”，“Q”，“L”，或者像单词“Sequel”。无论哪种方式都可以，尽管要注意…你可能会遇到一些人，他们固执地坚持自己的发音是正确的。即使是大型科技公司似乎也不同意这一点:微软和甲骨文数据库使用“Sequel”，而 IBM 和大多数开源数据库使用“S.Q.L”

SQL 最初是由 IBM 的研究人员在 70 年代开发的，最初被称为 SEQUEL。然而，一家飞机公司已经持有该商标，所以他们去掉了元音字母，并将其重命名为 SQL。SQL 有时也被称为结构化查询语言……但那更拗口。

有不同类型的 SQL 数据库。一些最流行的是 Oracle 数据库、SQL Server、DB2、PostgreSQL 和 MySQL。这些都使用 SQL，但是它们之间有一些细微的变化。

更多关于历史的[这里](https://vertabelo.com/blog/sql-or-sequel/)

## 什么是数据、数据库、表格？

**数据**是与对象相关的事实的集合。数据可以是数字、文字、尺寸，甚至是图像、声音或视频。

数据库是一个有组织的数据集合。这是一种存储信息的方式，以便以后可以访问。有许多不同类型的数据库。一些例子是关系数据库、面向对象数据库、NoSQL 数据库、图表数据库……在这篇博客中，我们将讨论关系数据库——数据存储在彼此相关的表中。

什么是桌子？这是数据按行和列分组的地方。如果你熟悉 excel 或 Google sheets，表格也是类似的。一列中的所有数据必须是同一类型的数据。数据类型包括字符串(一系列字符)、数字、日期和布尔值(表示真或假)。事情远不止如此。例如，您可能想要指定您的数据是否有小数(整数对浮点)，或者它的大小(微小对中等对大)。查看更多[点击这里](https://dev.mysql.com/doc/refman/8.0/en/data-types.html)

![](img/84b7631062d7f141623eeeef1bffe425.png)

由 [Abel Y Costa](https://unsplash.com/@abelycosta?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 制作的表格(不是数据表)

## 我们为什么要用桌子？

将数据存储在表中使我们能够高效地存储数据并轻松地访问数据。

想象一下，我们想要跟踪一个篮球赛季的球员。我们可以尝试构建一个大表，不仅包含每场比赛中的每场比赛，还包含所有球员信息和球队信息。这将很快变得非常复杂。我们会有很多重复的信息。每次勒布朗·詹姆斯打球，我们都要列出他的位置、身高、体重。想象一下如果事情发生了变化。如果一个玩家长高了，增加了[的体重](https://twitter.com/espn/status/1076269609156837377/)，甚至[改名为](https://www.cbssports.com/nba/news/ron-artest-changes-name-again-this-time-to-metta-ford-artest/)，那么我们将不得不多次修改我们的数据

相反，如果我们有一个包含每次比赛统计数据的游戏表，一个包含赛季记录和球队花名册的球队表，一个包含姓名和个人统计数据的球员表，那么跟踪我们的数据并做出更改将会容易得多。

## **如何使用 SQL**

SQL 是用语句编写的。我们将使用的语句是 SELECT —这允许我们访问和显示数据。SELECT 语句称为查询。SQL 不区分大小写，但最好将 SQL 关键字大写，以便它们在查询的其他部分中脱颖而出。

我们也想让我们的代码尽可能的易读。这通常意味着每个关键字都要换一行，或者为了防止代码变得太长。我们还使用缩进和空白来表明代码与关键字相关。

用分号结束 SQL 语句也是一个好习惯。分号是语句结束的标志。在一些 SQL 数据库中，需要使用这种方法。

![](img/a1f4a7eb9a4d043ec26ae433505e82a2.png)

第一步照片由[克里斯蒂安·陈](https://unsplash.com/@christianchen?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

## **第一步**

要查看一个表中的所有数据，我们应该编写:

**SELECT * FROM table；**

**选择**是我们选择想要显示的列的地方。这里我们用逗号分隔列名，或者在上面的例子中我们用“*”来显示所有的列

**FROM** 是我们选择包含所需列的表的地方。

就这样？只需三个词，我们就能看到整个表格。SQL 很简单。

没那么快。如果我们只有一个表，并且只有很少的几行，这就很好了。事实上，数据总是更复杂…所以我们还需要学习一些技巧。尽管有一些共同点。除了少数例外，每个查询总是有一个 SELECT 和一个 FROM 语句。

有些数据库非常大，有几百万甚至几十亿行。检索所有这些数据需要很长时间。幸运的是，如果我们只想看看表中的数据是什么样子，我们不必检索所有内容。我们可以使用 LIMIT 命令只返回我们指定的行数。

```
SELECT *
  FROM table
 LIMIT 10;
```

也许我们想看看谁是比赛中得分最高的球员。我们可以在想要排序的列上使用 **ORDER BY** 子句对数据进行排序。为了将最高分排在顶部，我们将在列名后面加上 **DESC** 来降序排列。在我们的 SQL 查询中，ORDER BY 出现在我们应用了过滤器之后——我们将在接下来了解这一点。

![](img/95f585e90252311d6852e2bd650c78a6.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由[马克·巴宾](https://unsplash.com/@marcbabin?utm_source=medium&utm_medium=referral)过滤

# **过滤我们的结果**

对数据进行排序是一回事，但是如果我们有特定的东西要搜索呢——例如回到我们的篮球示例，我们希望找到在一场比赛中得分超过 30 分的所有球员。为此，我们可以使用 **WHERE** 子句过滤数据。其中给出了满足指定条件的数据子集。我们可以使用数字过滤器，如等于、不等于、大于、小于；或者非数字过滤器—比如，IN，NOT …

## 喜欢

**LIKE** 允许我们过滤相似的值，而不是精确的值。LIKE 的特别强大之处在于，我们可以用通配符代替一个字符或一系列字符。通配符“%”代表任何字符序列。它可以在我们筛选的值之前或之后应用。通配符“_”是单个字符的替代。我们也可以使用 ILIKE，它的工作方式与 LIKE 相同，但忽略了字符的大小写。

```
SELECT last_name, height
  FROM players
 WHERE last_name ILIKE ‘jam%’;
```

## 在…里

中的**允许我们过滤想要包含的值列表。在下面的例子中**

```
SELECT *
  FROM games
 WHERE month IN (‘May’, ‘June’);
```

## 不

我们将**而不是**放在条件语句之前，以选择语句为假的行。NOT 常与 LIKE 和 IN 连用。

```
SELECT *
  FROM games
 WHERE month NOT IN (‘May’, ‘June’);
```

## 和

**和**用于过滤满足两个或更多条件的行。并且由于数据必须满足更多的条件，使得过滤器更加严格。这样，它的工作方式与英语句子相反。例如，如果我们想要勒布朗詹姆斯**和**安东尼·戴维斯的数据，我们不能应用以下过滤器:

```
WHERE last_name is ‘James’ AND ‘Davis’
```

因为没有同时姓詹姆斯和戴维斯的球员。

```
SELECT team,
  FROM games
 WHERE month NOT IN (‘May’, ‘June’)
   AND home_score <= 100;
```

## 运筹学

或用于筛选满足任何条件的行。或者使过滤器更加膨胀。在前面的例子中，为了得到勒布朗·詹姆斯和安东尼·戴维斯，我们可以使用 OR。我们经常一起使用 AND 和 OR。

```
SELECT *
  FROM players
 WHERE team LIKE ‘%Lakers’
   AND (last_name = ‘Davis’ OR last_name = ‘James');
```

## 在...之间

这用于筛选列中的值在特定范围内的行。虽然我们可以使用 AND / OR 来复制相同的东西，但是在之间使用**更加直观。请注意，BETWEEN 是包含性的，它包括我们指定的值。**

```
SELECT *
  FROM games
 WHERE date BETWEEN ‘2020–01–01’ AND ‘2020–04–01’;
```

## 空

我们可以使用 IS **NULL** 或来标识没有数据的行。这是检查缺失数据的有用步骤。注意我们不能使用= NULL。

```
SELECT *
  FROM players
 WHERE last_name IS NULL;
```

这涵盖了在一个表中处理数据的基本步骤。我们的下一步是学习连接表，但这是另一个博客。

![](img/4eb707765eb724f91bcdc92640137474.png)

由[亚伦负担](https://unsplash.com/@aaronburden?utm_source=medium&utm_medium=referral)对 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 进行总结

# 摘要

**关键字** : *代码示例*(如何使用)

**选择** : *选择 column1，column2* (选择需要分析的列)

**FROM:***FROM table*(选择包含所需列的表格)

**限制:** *限制 5* (限制您指定的数据返回行数)

**排序依据:** *排序依据列 1* (使用指定的列对数据进行排序)

**WHERE:***WHERE column 1>3*(将结果过滤为大于指定值)

## **逻辑运算符**

**LIKE:** *其中 first _ name LIKE“% Bron %”*(当列包含指定文本“Bron”时返回行)

**IN:***WHERE month IN(' Aug '，' Sep')* (指定要包含的值列表，在本例中仅包含' Aug '和' Sep '的月份)

**不:** *其中月份不在('八月'，'九月')*(不颠倒条件句)

**和** *其中点> 20 和反弹> 10* (过滤所有指定条件均为真的行)

**或:** *其中指向> 10 或辅助> 10* (至少有一个条件为真时过滤)

**BETWEEN:***WHERE minutes _ played BETWEEN EN 20 和 30* (选择一个范围内的值)

**为空:** *其中姓氏为空*(过滤该列中没有值的行)

## 资源

有关 SQL 的更多信息，请点击此处:

[](https://mode.com/sql-tutorial/introduction-to-sql/) [## 数据分析 SQL 教程|基本 SQL 模式分析

### 本教程是为想用数据回答问题的人设计的。对许多人来说，SQL 是“肉和土豆”…

mode.com](https://mode.com/sql-tutorial/introduction-to-sql/) 

这篇博客只以篮球为例。如果你真的想探索这一点，这里有一些很好的资源:

[](https://www.basketball-reference.com/) [## 篮球统计和历史| Basketball-Reference.com

### 纪念汤姆·海因森(1934-2020)凯尔特人队的传奇人物，名人堂成员，8 次 NBA 冠军球员，2 次 NBA 冠军主教练…

www.basketball-reference.com](https://www.basketball-reference.com/) [](https://medium.com/hardwood-convergence) [## 硬木会聚

### 篮球和机器学习相遇的地方

medium.com](https://medium.com/hardwood-convergence) [](https://www.nba.com/stats/) [## NBA 数据

### NBA 高级统计的家-NBA 官方统计和高级分析。

www.nba.com](https://www.nba.com/stats/)