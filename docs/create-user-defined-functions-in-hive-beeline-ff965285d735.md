# 在 Hive / Beeline 中创建用户定义的函数

> 原文：<https://medium.com/analytics-vidhya/create-user-defined-functions-in-hive-beeline-ff965285d735?source=collection_archive---------10----------------------->

Hive / Beeline 支持创建宏，这些宏非常方便且易于创建。

![](img/4117ef729e5295b9fee1d06996b144a5.png)

国际管理集团:unsplash.com

```
**Quick syntax** CREATE TEMPORARY MACRO <macroname> (optional params)
<body of the macro>
```

创建临时宏使用给定的可选列列表作为表达式的输入来创建宏。顾名思义，临时宏存在于…