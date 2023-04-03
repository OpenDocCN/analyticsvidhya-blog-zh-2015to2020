# DB::交易！！拉勒维尔甜蜜黑客#00…

> 原文：<https://medium.com/analytics-vidhya/db-transaction-the-laravel-sweet-hack-00-ded131b71ec9?source=collection_archive---------13----------------------->

# 固执己见的观点

对我来说，Laravel 是最好的 php 开发框架之一，除了对原始 php 的了解。我只使用过 code igniter，slim，你可以支持我的观点，这是一个固执的观点，但我手头有事实(*改天再讲*)

# 少数黑客

**黑客！！！**是的，hack，根据我最小的语言用法库，hack 是以简单的方式完成复杂事情的整洁方式。或者我们可以称之为让困难的事情看起来简单的好方法。

如果我可以通过对我进行非常广泛的查询来获得一个关系的数据，这是一种黑客行为，这就是 laravel 为您提供的模型中属性的使用，惰性集合的使用以及它提供的所有其他相关 ORM 功能。

**阴谋黑客*****数据库事务*** 赋予我们更安全地执行数据多次修改操作的能力，如 ***删除、更新和插入*** 数据到我们的数据库。

DB::transaction 有一个 **DB::commit()** 和 **DB::rollback()** ，它们是基于正在执行的查询批的状态使用的。

# 引人深思的事

给出了一个任务，要求做以下事情:您需要解析多个 **CSV 文件，每个文件有大约 50k 条记录**，其中它应该首先删除所有记录，然后在数据库中插入所有记录。什么方法是可行的？？！！当你使用事务时，任务会做得很好。

***伪代码*** *为此会看起来像这样的东西*

**DB::transaction()**/*开始事务*/

*试试{*

while(($ data = fgetcsv($ exchange symbol))！= = false)/*读取 csv 记录*/

if(count($ row data)= = 10000){ DB::table(' all _ securities ')-> insert($ row data)；unset($ row data)；} /*获取前 10000 条记录并尝试插入它们*/

**DB::commit()**；/* Commit Transaction */}*catch(\ Exception $ e){*//回滚事务 DB::roll back()；} fclose($ exchange symbol)；

这是一种更简单的看待问题的观点。您可能会注意到，我们插入了 10，000 条记录。我们可以根据我们的需要/数据库服务器来改变这个限制。

因此，在我们的示例中，如果在 while 循环中的删除操作或插入操作引发了任何异常，事务将在 catch 块中回滚，并且不会发生任何数据库更改。如果一切顺利，我们提交交易。

**谢谢 DB::transaction()**

如果它值得别人欣赏，请鼓掌欢迎。干杯

# laravel # laravelhacks # simpler code