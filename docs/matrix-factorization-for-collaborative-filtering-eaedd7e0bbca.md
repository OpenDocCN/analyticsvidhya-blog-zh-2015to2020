# 用于协同过滤的矩阵分解

> 原文：<https://medium.com/analytics-vidhya/matrix-factorization-for-collaborative-filtering-eaedd7e0bbca?source=collection_archive---------15----------------------->

![](img/864bff8f0921b0e2b800a757df842bfd.png)

图片来源— [网飞](https://www.netflix.com/browse)

在这篇博文中，我们试图理解在推荐系统中使用矩阵分解进行协同过滤背后的基本直觉。

协同过滤背后的核心思想是，过去同意的用户将来也会同意。

让我们通过一个例子来理解这一点。设用户用向量 U 表示，并且…