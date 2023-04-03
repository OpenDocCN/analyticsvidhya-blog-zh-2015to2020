# Python 中可变对象的浅拷贝、深拷贝和无拷贝

> 原文：<https://medium.com/analytics-vidhya/shallow-copy-deep-copy-and-no-copy-of-immutable-objects-in-python-b6e673e23c01?source=collection_archive---------13----------------------->

Python 有两种类型的拷贝——深层拷贝和浅层拷贝。在本文档中，我们将讨论在可变对象的上下文中这些是什么。但首先，我们将看到我所说的*无副本*是什么意思。让我们开始吧。

![](img/60818253b38baf63597d6c4b73bb5fab.png)

照片由[路易斯·汉瑟@shotsoflouis](https://unsplash.com/@louishansel?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# 没有副本

嗯，没有副本——顾名思义——不是副本。只是一个新手程序员分配一个…