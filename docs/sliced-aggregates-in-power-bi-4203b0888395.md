# Power BI 中的切片聚合

> 原文：<https://medium.com/analytics-vidhya/sliced-aggregates-in-power-bi-4203b0888395?source=collection_archive---------24----------------------->

## 利用数据建模和 DAX 加速数据集

不同的建模方法可以提高 Power BI 数据集的性能，尽管会带来额外的复杂性。

一种情况是，用户需要访问非常大的数据集，但通常应用相同的过滤器集，从而产生满足大多数查询需求的一小部分行。切片聚合方法(在数据模型+ DAX 中实现)可用于使您的模型在需要的地方发挥性能，但仍允许全方位的…