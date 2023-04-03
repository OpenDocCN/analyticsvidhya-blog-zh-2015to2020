# VGGNet 架构解释

> 原文：<https://medium.com/analytics-vidhya/vggnet-architecture-explained-e5c7318aa5b6?source=collection_archive---------1----------------------->

VGGNet 是由牛津大学的卡伦·西蒙扬和安德鲁·齐泽曼在 2014 年提出的卷积神经网络架构。本文主要研究卷积神经网络的深度对其精度的影响。你可以找到 VGGNet 的原始论文，题目是[用于大规模图像识别的甚深卷积网络](https://arxiv.org/abs/1409.1556)。

**建筑**

基于 VGG 的 convNet 的输入是 224*224 RGB 图像。预处理层采用像素值在 0–255 范围内的 RGB 图像，并减去图像平均值，即…