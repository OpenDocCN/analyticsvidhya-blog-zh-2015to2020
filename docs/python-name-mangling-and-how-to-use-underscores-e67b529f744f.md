# Python 名称混乱以及如何使用下划线

> 原文：<https://medium.com/analytics-vidhya/python-name-mangling-and-how-to-use-underscores-e67b529f744f?source=collection_archive---------5----------------------->

## 为什么 python 中的 __var 可能不像您想象的那样工作

![](img/480dfed390f119c081d0e1453487b289.png)

基于[大卫·克洛德](https://unsplash.com/@davidclode?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/python?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

## _ 单下划线前缀

如果你在代码中使用了下划线，你最有可能使用的就是这些。单个前导下划线被用作方法和数据属性的弱“内部使用”或“私有”指示符。这些…