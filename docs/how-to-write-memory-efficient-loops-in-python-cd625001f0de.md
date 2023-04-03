# 如何用 Python 编写高效内存循环

> 原文：<https://medium.com/analytics-vidhya/how-to-write-memory-efficient-loops-in-python-cd625001f0de?source=collection_archive---------1----------------------->

*生成器的可视化指南和实现它们的三种方式*

![](img/6134fc10757457c0978ce351b31f0b6c.png)

照片由[Tine ivani](https://unsplash.com/@tine999?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

在 python 中，当你构建一个数字、图像、文件或任何其他你想要迭代的对象的列表时，你实际上是在堆积内存，因为你在列表中添加了新的条目，也就是说，每次你做*your _ list . append(new _ item)*你的列表消耗的内存块等于 *sys.getsizeof(new_item)* 。这里的问题是…