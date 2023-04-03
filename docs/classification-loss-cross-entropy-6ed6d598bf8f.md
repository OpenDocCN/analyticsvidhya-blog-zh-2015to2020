# 分类损失:交叉熵

> 原文：<https://medium.com/analytics-vidhya/classification-loss-cross-entropy-6ed6d598bf8f?source=collection_archive---------33----------------------->

我最近从事计算机视觉项目的分类任务。论文和教程提到**交叉熵**作为最常用的损失函数来衡量预测和标签之间的差异。现在，问题是我们用交叉熵做什么，怎么用，为什么用？

本文基于 Tensorflow 使用 MNIST 数据集的 [**对服装图像**](https://www.tensorflow.org/tutorials/keras/classification) 进行分类教程。

![](img/43ee057fc9501403bab5e72a13ca3792.png)

MNIST 数据集的服装图像