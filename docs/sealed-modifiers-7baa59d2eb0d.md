# 密封改性剂

> 原文：<https://medium.com/analytics-vidhya/sealed-modifiers-7baa59d2eb0d?source=collection_archive---------15----------------------->

![](img/90013dcb622af64d4e6361a5fab7691e.png)

在本文中，我将讨论什么是密封修饰符，如何使用它，以及它对应用程序性能的影响。

首先，我们先来一个定义；sealed 是一个修饰符，如果它被应用于一个类，那么它就**不可继承**，如果它被应用于虚拟方法或属性，那么它就**不可验证**。

```
public sealed class A { ... }
public class B 
{
    ...
    public sealed string…
```