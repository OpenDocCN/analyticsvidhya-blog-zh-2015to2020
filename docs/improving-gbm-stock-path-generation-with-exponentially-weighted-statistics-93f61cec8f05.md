# 利用指数加权统计改进 GBM 股票路径生成

> 原文：<https://medium.com/analytics-vidhya/improving-gbm-stock-path-generation-with-exponentially-weighted-statistics-93f61cec8f05?source=collection_archive---------11----------------------->

![](img/d4fed1eb0b09b08b90479f0946787c17.png)

马库斯·温克勒在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

[在最近的一篇文章](/analytics-vidhya/simulating-stock-prices-using-geometric-brownian-motion-f400bb6af6dd)中，我写了使用蒙特卡罗模拟来确定股票期权盈利的可能性，方法是使用几何布朗运动(GBM)生成多条路径，并计算这些路径的一些统计数据。这个项目可以在我的[个人项目网站](http://projects.anthonymorast.com/buyingoptions/)找到。完成那个项目后不久，我在看一个 YouTube 视频…