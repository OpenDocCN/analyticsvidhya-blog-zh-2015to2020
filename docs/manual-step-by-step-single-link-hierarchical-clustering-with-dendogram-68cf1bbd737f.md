# 使用树状图的手动逐步单链接层次聚类。

> 原文：<https://medium.com/analytics-vidhya/manual-step-by-step-single-link-hierarchical-clustering-with-dendogram-68cf1bbd737f?source=collection_archive---------3----------------------->

您来这里是因为，您对层次聚类有所了解，并且想知道单链路聚类如何工作以及如何绘制树状图。

**层次聚类**:其慢::复杂::可重复::不适合大数据集。

让我们取 6 个简单的向量。

使用欧几里得距离让我们计算距离矩阵。
`Euclidean Distance = sqrt( (x2 -x1)**2 + (y2-y1)**2 )`