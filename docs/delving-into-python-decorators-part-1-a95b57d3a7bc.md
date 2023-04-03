# 深入研究 Python Decorators:第 1 部分

> 原文：<https://medium.com/analytics-vidhya/delving-into-python-decorators-part-1-a95b57d3a7bc?source=collection_archive---------28----------------------->

![](img/0e9f0e2fb0cd35db746f14aa4a19ed92.png)

这是关于 Python 中装饰者的四部分系列的第一部分。

出于多种原因，Decorators 是一个强大的 Python 特性。它们允许我们扩展一个*可调用的* [ [1](https://www.geeksforgeeks.org/callable-in-python/) ]的行为，而无需永久修改它们。它们提高了可读性、可维护性，从而提高了生产率。除此之外，装饰者在现代程序员的工具箱中有更多的实际用途。你可以用它们来…