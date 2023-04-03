# 使用 Tensorflow2.0 从头开始构建和训练 U-Net

> 原文：<https://medium.com/analytics-vidhya/training-u-net-from-scratch-using-tensorflow2-0-fad541e2eaf1?source=collection_archive---------0----------------------->

作为一名机器学习工程师，我一直试图理解已发表的研究论文，但理解这些论文总是很难，复制这些结果更难。

一篇这样的论文是 [U-Net](https://arxiv.org/pdf/1505.04597.pdf) :生物医学图像分割的卷积网络。我已经看到了相当多的 U-Net 实现，但是它们都没有实现论文中解释的确切架构。

在代码和文章的实现过程中，我确实参考了 TensorFlow 提供的[教程](https://www.tensorflow.org/tutorials/images/segmentation)，它…