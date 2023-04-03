# 一种实用的无监督异常检测方法

> 原文：<https://medium.com/analytics-vidhya/a-practical-approach-to-unsupervised-anomaly-detection-c19849bcce03?source=collection_archive---------30----------------------->

![](img/8993d53e3761eb79fca858337534c710.png)

在这篇文章中，我们将使用隔离森林和 k-means 聚类在一个真实的例子上执行无监督的异常检测。

**背景** —有一家自行车共享初创公司，根据行驶的距离向客户收取每次骑行的费用。距离的计算有三个来源——通过自行车 GPS，通过安装在自行车上的物联网盒子…