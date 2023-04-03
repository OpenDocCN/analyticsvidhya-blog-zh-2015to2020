# Python 中的浅拷贝 vs 深拷贝 vs 赋值

> 原文：<https://medium.com/analytics-vidhya/shallow-copy-vs-deep-copy-vs-assignment-in-python-921d7e413a3a?source=collection_archive---------6----------------------->

## python 中浅拷贝、深拷贝和赋值的快速概述。

![](img/63df176248641a3b79fa1a34888322d8.png)

照片由[丹尼尔·奥伯格](https://unsplash.com/@artic_studios?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

**在 Python 中可以通过三种方式复制对象:**

1.  **赋值操作**
    `**a=b**`
2.  **浅抄** 抄()
    `**a=copy(b)**`
3.  **deepcopy** `**a=deepcopy(b)**`

## 赋值操作: