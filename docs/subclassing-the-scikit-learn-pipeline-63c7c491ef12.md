# 子类化 Scikit-Learn 管道

> 原文：<https://medium.com/analytics-vidhya/subclassing-the-scikit-learn-pipeline-63c7c491ef12?source=collection_archive---------20----------------------->

![](img/4b09219e3705e8d61316f92bfac04cbb.png)

这是一条华丽的管道，与我们发现自己陷入的糟糕的意大利面条式代码一点也不相似。迈克·本纳在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

如果您访问 Scikit-Learn 开发人员指南，您可以很容易地找到他们希望您定制的对象的分类。它包括估计器、预测器、转换器和模型类，还有一个很好的向导[带你了解它们的 API 的来龙去脉。](https://scikit-learn.org/stable/developers/develop.html)

但是如果由于某种(可能被误导的)原因，您已经决定实现您自己的…