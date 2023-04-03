# 递归神经网络

> 原文：<https://medium.com/analytics-vidhya/recurrent-neural-networks-97f3b034e70?source=collection_archive---------4----------------------->

神经网络是深度学习任务中采用的典型算法。在我的[上一篇](https://towardsdatascience.com/neural-networks-parameters-hyperparameters-and-optimization-strategies-3f0842fac0a5)中，我一直在谈论神经网络的基本结构以及在建立深度学习模型之前应该知道的元素(参数、超参数和策略)。

在这里，我将深入研究 RNN 领域。

RNN 背后的思想是，给定一个变量及其相应的目标(我们想要预测的值)，今天的产出可能会影响明天的产出。对于熟悉时间序列分析的人来说，这听起来可能…