# 使用随机投影节省聚类的计算时间

> 原文：<https://medium.com/analytics-vidhya/save-computation-time-for-clustering-using-random-projections-bf8441486b29?source=collection_archive---------4----------------------->

## 加速 k-means 聚类等算法的简单方法

![](img/d862cc90293d8dcd33ffecb3aad96b25.png)

梅尔·普尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

F 有时，你不得不处理具有大量特征的数据集。通常，这是一件好事，因为我们可以利用的特征越多，我们的分类和回归模型的预测就越准确，至少如果…