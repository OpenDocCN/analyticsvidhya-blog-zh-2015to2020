# 反向传播背后的直觉

> 原文：<https://medium.com/analytics-vidhya/bacpropagation-6ff75e72d06a?source=collection_archive---------23----------------------->

## 反向传播的数学

## 问题定式化

考虑下面的计算图，它基本上显示了深度神经网络:

其中 **x** 是网络的输入， **y** 是网络的输出， *f* 是一个线性层，也可以定义为:

其中 **W⁽ⁱ⁾** 和 **b⁽ⁱ⁾** 分别是层 *i* 对应的权重偏差。 *g* 为非线性层(函数)，如`relu`或`sigmoid`。