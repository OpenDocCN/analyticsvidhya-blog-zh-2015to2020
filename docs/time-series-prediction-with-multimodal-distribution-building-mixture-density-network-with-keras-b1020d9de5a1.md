# 多峰分布时间序列预测——用 Keras 和张量流概率建立混合密度网络

> 原文：<https://medium.com/analytics-vidhya/time-series-prediction-with-multimodal-distribution-building-mixture-density-network-with-keras-b1020d9de5a1?source=collection_archive---------2----------------------->

## 探索均值是不好的估计值的数据。

![](img/fe12e5960aba1ae85e462c0dc288ee9e.png)

image:[https://commons . wikimedia . org/wiki/File:Guentersberg _ wasserkupe _ Tree _ Road _ fork . png](https://commons.wikimedia.org/wiki/File:Guentersberg_Wasserkuppe_Tree_Road_Fork.png)

当我们进行回归时，我们希望估计最可能的值。但是根据我们的模型和数据，我们通常只能得到一些最小化均方误差的数字…