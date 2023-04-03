# 利斯科夫替代原理

> 原文：<https://medium.com/analytics-vidhya/liskov-substitution-principle-fb36c7a38874?source=collection_archive---------20----------------------->

![](img/78b021246059dddb8055e44d4e15d696.png)

> 子类型必须可以替换它们的基本类型。

芭芭拉·利斯科夫在 1987 年提出了这个原则。它通过关注超类的行为及其子类型扩展了开闭原则。当我们考虑违反它的后果时，它的重要性就变得很明显了。考虑一个使用以下类的应用程序。

```
public class Rectangle 
{ 
  private double width…
```