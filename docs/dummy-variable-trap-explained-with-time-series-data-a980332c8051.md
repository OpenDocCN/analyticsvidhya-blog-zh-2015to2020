# 用时间序列数据解释虚拟变量陷阱

> 原文：<https://medium.com/analytics-vidhya/dummy-variable-trap-explained-with-time-series-data-a980332c8051?source=collection_archive---------16----------------------->

> "知道陷阱在哪里——这是避开它的第一步。"

![](img/80c553c7ab4a626bc99b951b87eb5de8.png)

我们遇到的许多数据集都有连续变量和分类变量的组合。很少有像决策树这样的算法可以直接处理分类数据，但是许多其他的 ML 算法不能处理标签数据，我们需要将所有的变量转换成数字。