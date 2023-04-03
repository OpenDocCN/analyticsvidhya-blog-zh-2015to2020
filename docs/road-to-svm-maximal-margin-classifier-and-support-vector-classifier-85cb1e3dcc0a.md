# 通往 SVM 之路:最大间隔分类器和支持向量分类器

> 原文：<https://medium.com/analytics-vidhya/road-to-svm-maximal-margin-classifier-and-support-vector-classifier-85cb1e3dcc0a?source=collection_archive---------3----------------------->

[支持向量机](/swlh/support-vector-machine-from-scratch-ce095a47dc5c)是一种流行的用于分类任务的机器学习算法，尤其是对非线性可分数据的适应性(这要归功于所谓的核技巧)。然而，在我们今天使用之前，开发了几个具有相同底层结构的模型。在这篇文章中，我将向你展示其中两个项目背后的直觉，它们的逐步实施带来了现代的 SVM。

它们是最大间隔分类器和支持向量分类器。然而…