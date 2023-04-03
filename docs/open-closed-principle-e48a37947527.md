# 开闭原理

> 原文：<https://medium.com/analytics-vidhya/open-closed-principle-e48a37947527?source=collection_archive---------23----------------------->

![](img/a301b6acf57bd465dc54b4157cb7f8b6.png)

> 软件实体(类、模块、函数等。)应该对扩展开放，但对修改关闭

这个原则建议我们重构系统，以便通过添加新代码来实现进一步的更改，并且因为它不会更改旧代码，所以不会导致更多的修改。符合 OCP 的模块有两个主要属性。

1.  它们可以延期。的行为…