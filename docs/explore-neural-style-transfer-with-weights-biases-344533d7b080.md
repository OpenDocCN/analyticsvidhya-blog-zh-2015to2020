# 探索神经类型转移

> 原文：<https://medium.com/analytics-vidhya/explore-neural-style-transfer-with-weights-biases-344533d7b080?source=collection_archive---------11----------------------->

研究超参数对艺术风格的影响

![](img/b71877d3a6a3a51a851ca6041c732720.png)

来源-[https://www . wandb . com/articles/neural-style-transfer-with-weights-bias](https://www.wandb.com/articles/neural-style-transfer-with-weights-biases)

# 介绍

在本教程中，我们将通过 Gatys 的神经风格转移算法，实现它，并使用 W&B 库跟踪它。让我们假设我们正在为生产构建一个风格转换应用程序。我们需要比较改变各种参数所产生的结果。这需要主观…