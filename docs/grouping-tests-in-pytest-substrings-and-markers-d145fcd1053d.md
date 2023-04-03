# PyTest 中的分组测试-子字符串和标记

> 原文：<https://medium.com/analytics-vidhya/grouping-tests-in-pytest-substrings-and-markers-d145fcd1053d?source=collection_archive---------8----------------------->

正如我之前的[博客](https://blog.usejournal.com/getting-started-with-pytest-c927946b686)所承诺的，在这里我将讨论 PyTest 中分组测试的强大特性。测试分组是执行测试的有效方法，因为它帮助我们根据标准执行或跳过某些测试。如果你是 PyTest 新手，那么在此之前请参考[PyTest 入门](https://blog.usejournal.com/getting-started-with-pytest-c927946b686)。

![](img/9497c58f76684c6d01c364510ff2fdc1.png)

照片由[马丁·桑切斯](https://unsplash.com/@martinsanchez?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

> 在潜入之前，快速环境检查 **Python 版本** …