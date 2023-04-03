# 深度学习基础——重量衰减

> 原文：<https://medium.com/analytics-vidhya/deep-learning-basics-weight-decay-3c68eb4344e9?source=collection_archive---------1----------------------->

# 什么是体重衰减？

权重衰减是一种正则化技术，通过向损失函数添加小的惩罚，通常是权重(模型的所有权重)的 L2 范数。

损耗=损耗+重量衰减参数*重量的 L2 范数

有些人喜欢只对权重而不是偏差应用权重衰减。PyTorch 将权重衰减应用于权重和偏移。

# **我们为什么要用重量衰减？**

*   以防止过度拟合。