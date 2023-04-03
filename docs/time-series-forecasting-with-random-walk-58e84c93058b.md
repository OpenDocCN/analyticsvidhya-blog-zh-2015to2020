# 随机游走时间序列预测

> 原文：<https://medium.com/analytics-vidhya/time-series-forecasting-with-random-walk-58e84c93058b?source=collection_archive---------1----------------------->

## 我可以用随机游走过程来模拟我的时间序列吗？

## 摘要

本文的目的是测试时间序列是否可以通过模拟一个[随机行走过程](https://en.wikipedia.org/wiki/Random_walk)来复制。

随机游走的结构很简单，下一个观察值等于上一个观察值加上一个随机噪声:

> y[t+1] = y[t] + wn~(0，σ)

所以用机器学习的话来说，我们的任务是建立一个学习标准的随机行走…