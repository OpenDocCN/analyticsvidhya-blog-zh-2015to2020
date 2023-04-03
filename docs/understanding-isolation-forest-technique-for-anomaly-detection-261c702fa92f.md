# 了解用于异常检测的隔离林技术

> 原文：<https://medium.com/analytics-vidhya/understanding-isolation-forest-technique-for-anomaly-detection-261c702fa92f?source=collection_archive---------18----------------------->

![](img/e79ce6b838af1c82cbc0f57a621e5611.png)

**什么是异常？**
在数据集中，异常是一组在特征上不同于正常数据点的数据点。

**隔离森林背后的理念—** 由于异常很少，并且与正常数据点不同，如果我们对数据集进行随机分区，异常点将有可能…