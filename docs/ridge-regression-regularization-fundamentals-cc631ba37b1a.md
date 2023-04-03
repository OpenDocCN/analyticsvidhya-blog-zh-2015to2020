# 岭回归:正则化基础

> 原文：<https://medium.com/analytics-vidhya/ridge-regression-regularization-fundamentals-cc631ba37b1a?source=collection_archive---------6----------------------->

正则化是一种用于减少机器学习模型的方差的方法；换句话说，它用于减少过度拟合。当机器学习模型在训练样本上表现良好，但未能对未经训练的数据产生准确预测时，就会发生过度拟合。

理论上，有两种主要的方法来建立能够很好地对看不见的数据进行归纳的机器学习模型:

1.  为我们的目的训练最简单的模型(根据奥卡姆剃刀)。