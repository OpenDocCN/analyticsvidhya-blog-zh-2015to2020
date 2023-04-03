# 超图像超分辨率的损失函数(SISR)

> 原文：<https://medium.com/analytics-vidhya/loss-functions-for-image-super-resolution-sisr-8a65644fbd85?source=collection_archive---------12----------------------->

在 SISR，Autoencoder 和 U-Net 被大量使用；然而，众所周知，他们很难训练到收敛。损失函数的选择对指导模型走向最优起着重要的作用。今天，我介绍两个单幅图像超分辨率的损失函数。

陆正阳和陈莹发表了一个具有创新的损失函数的 U-Net 模型，用于单幅图像的超分辨率。他们的工作引入了两个损失函数:像素比较的均方误差(MSE)和边缘比较的平均梯度误差(MGrE)。在…