# 孟乔森强化学习

> 原文：<https://medium.com/analytics-vidhya/munchausen-reinforcement-learning-9876efc829de?source=collection_archive---------3----------------------->

![](img/2a9f443d97d693649fb5373eb17aed13.png)

一种改进的 DQN 网络，其性能优于 [**【彩虹】**](https://arxiv.org/abs/1710.02298) ，而没有利用**优先化经验重放**缓冲器或 **n 步引导**的进步。

论文的作者 [**Munchausen 强化学习**](https://arxiv.org/abs/2007.14430)**【M-RL】**通过一个非常简单的想法实现了这些令人印象深刻的结果:将比例对数策略添加到即时奖励中。