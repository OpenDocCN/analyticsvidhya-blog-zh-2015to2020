# MySQL 中的触发器

> 原文：<https://medium.com/analytics-vidhya/triggers-in-mysql-8bda5ba2025c?source=collection_archive---------21----------------------->

你好。今天，让我们来看看 MySQL 中一个有趣的话题，触发器。

对于那些不了解 SQL 的人来说，结构化查询语言(SQL)是一种用于关系数据库管理和数据操作的标准计算机语言。MySQL 是数据库管理系统的一个例子。嘶…它很有名。

![](img/acc5807fb63a96ad8b7187e2895cbf33.png)

卡斯帕·卡米尔·鲁宾在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

**触发器**帮助我们在检测到特定原因时实例化一个动作。简而言之，当表发生特定变化时，执行特定的预定义操作。

让我们看一个 MySQL 中触发器命令的例子，稍后我们可以讨论它的功能。

```
DELIMITER $$
CREATE
   TRIGGER my_trigger BEFORE INSERT
   ON EMPLOYEE
   FOR EACH ROW BEGIN
      INSERT INTO trigger_test VALUES('add new emp');
END $$
DELIMITER;
```

## 说明

上面的代码创建了一个触发器，它在用户输入每条记录之前，在“EMPLOYEE”数据库的“trigger_test”表中插入一条带有文本“add new emp”的记录。因此，例如，如果用户输入了 2 条记录，他/她将按以下顺序看到这些记录:

添加新员工'，'记录 1 '，'添加新员工'，'记录 2 '

> “my_trigger”是我们刚刚创建的触发器的名称。

您可以将语法更改为 AFTER INSERT，以便在用户将每一行添加到表中后添加“添加新员工”记录，如下所示:

```
DELIMITER $$
CREATE
   TRIGGER my_trigger AFTER INSERT
   ON EMPLOYEE
   FOR EACH ROW BEGIN
      INSERT INTO trigger_test VALUES('add new emp');
END $$
DELIMITER;
```

您可以类似地执行更新和删除操作。

> 如果您对我们为什么使用“分隔符$$”感到好奇，它只是暗示代码的结尾现在将由双美元符号(“$$”)来表示，而不是默认的分号。这样做是为了区分触发器代码和普通的 MySQL 代码。

**现在**，让我们为一个触发器编写代码，如果记录的“年龄”列中的值低于 18，该触发器将拒绝允许向表中输入记录。

```
DELIMITER $$
CREATE
   TRIGGER is_adult BEFORE INSERT
   ON PEOPLE                        
   FOR EACH ROW BEGIN
      IF NEW.age < 18
      THEN
          SIGNAL SQLSTATE '45000'
          SET MESSAGE_TEXT = 'Only adults are allowed.';
      END IF;
END $$
DELIMITER;
```

上面的代码遵循与上一个代码相同的规则，唯一的变化是使用了 NEW.age，它引用了根据条件检查的新记录(试图插入的记录)。

> 信号 SQLSTATE '45000 '允许开发人员编写自定义错误消息。

因此，使用触发器，您可以在数据库表发生变化时采取自己选择的操作。

感谢阅读！