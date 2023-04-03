# 使用 PInject 的 Pythonic 依赖注入

> 原文：<https://medium.com/analytics-vidhya/pythonic-dependency-injection-with-pinject-3e9de4886517?source=collection_archive---------6----------------------->

本文简要介绍了 PInject，这是一个由 Google 编写的 Pythonic 依赖注入框架。

# 依赖注入框架

![](img/52e8d306e249bfa8fb3e7aa7408c7efe.png)

克里斯里德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

依赖注入是一个简单的概念:

1.  创建对象依赖关系图。
2.  当你要求一个对象时，你得到那个对象及其所有的…