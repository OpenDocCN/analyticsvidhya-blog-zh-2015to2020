# 迁移学习—概念 3 卷积网络

> 原文：<https://medium.com/analytics-vidhya/inceptionv3-convolutional-network-88a925d43eae?source=collection_archive---------15----------------------->

使用 InceptionV3 架构深入探讨迁移学习的应用

![](img/a0d5804d80081e4074d5bc20da95d065.png)

图片来源:[迁移学习—作者 Niklas Donges，Experfy，2019 年 2 月 15 日](https://machinelearningblogcom.files.wordpress.com/2018/04/1_dvc-cgzanelnsi4w_z-eva.png?w=736)

# 摘要

本文旨在通过 CIFAR-10 数据集的分类模型提供对 Inception V3 卷积网络应用的洞察和清晰性，每个训练类包含约 6000 幅图像。为了调整模型，我们需要对数据进行预处理，并添加要训练的层，从而转移对架构的学习。

# 介绍

使用卷积网络创建和训练图像识别模型被用于使用给定的一组参数提取相关的方面和特征。问题是在试图使模型尽可能低的偏差时，体系结构的复杂性增加，从而使其容易过拟合，并且训练成本高。作为一个解决方案，我们使用了一种方法，这种方法可以通过使用预先训练过的架构和应用这些方法来防止过度拟合。

# 材料和方法

*   **学习率衰减**:需要避免偏离通过减小每一步而达到的最小值，因此学习率必须降低
*   **提前停止**:需要防止模型在长时间的训练后变得更差，在这种情况下，有必要在模型开始沿着这条路走之前停止模型
*   迁移学习:这个想法让我们有能力使用从一个问题的解决过程中获得的知识，并将其应用到一个相关的问题中。在这种情况下，我们能够使用先前训练的网络，同时保持良好的准确性。通过采用 InceptionV3 架构，我们替换了最后的 1000 个类，并添加了另外两层，其中一层用于对数据集中的 10 个类进行分类
*   **上采样**:这是将我们的图像传递给模型所必需的，因为 CIFAR-10 数据集包含的图像的最小尺寸低于 InceptionV3 的模型所接受的尺寸(75，75，3)。如果一个子集的数据显示了一个独立的特征或行为集，而不是另一个子集或大多数的特征或行为集，那么这个过程通常用于处理数据不平衡
*   **Adam Optimizer** :该模型是一种自适应学习率优化算法，设计用于训练深度神经网络。这是用来编译我们的模型
*   Jupyter Notebook:使用 Jupyter Notebook 使用托管运行时进行最终测试，因为它明显更快。

*示例:学习率衰减函数的代码样本，用于元组验证模型和提前停止的条件:*

![](img/c39647070de50f416099b9e068974e40.png)

# 结果

虽然训练最初计划经历算法的 30 次迭代(历元)，但是如上所述，通过提前停止来停止模型，并且我们能够在验证数据集中获得 87%的准确性，在训练数据集中获得 98%的准确性。

![](img/6be98341e8d0f6b436e1c1c314b6e7f1.png)

# 讨论

在此过程中需要注意的一些事情包括使用较低的学习率来利用预训练模型的权重的重要性。这在选择超参数时发挥作用。我们使用了 Adam 优化器，但是其他优化器也可以工作得很好，比如 SGD，它们都会影响获得一个充分训练的模型所需的时间(算法的迭代次数)。

另一个值得注意的观察结果是第一次验证的准确率实际上要高于训练准确率，分别为 72%和 59%。这很可能是因为 Keras dropout 使用在训练集和测试集中显示了不同的行为。最终的速率是由于训练中连续模型变化的调整，因为在测试过程中关闭了漏失。此外，时段 5 中的验证准确度高于时段 6 中的验证准确度，除了训练集的分层和复杂性之外，这再次与退出使用有关。

冻结应用程序层也是优先的，否则计算时间会超出图表，特别是对于这种大小的数据集，并且因为层总是产生相同的输出。可以根据需要添加层来配置对模型的任何调整。

# 引用的文献

*   *深度学习迁移学习的温和介绍*。(2019 年 9 月 16 日)。机器学习精通。[https://machine learning mastery . com/transfer-learning-for-deep-learning/](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)
*   布沙耶夫诉(2018 年 10 月 24 日)。*亚当——深度学习优化的最新趋势*。中等。[https://towards data science . com/Adam-latest-trends-in-deep-learning-optimization-6be 9a 291375 c](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)
*   亚历克斯·克里日夫斯基。(2012).从微小图像中学习多层特征。多伦多大学。
*   *CIFAR-10 和 CIFAR-100 数据集*。(未注明)。多伦多大学计算机科学系。[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
*   Szegedy，c .，Vanhoucke，v .，Ioffe，s .，Shlens，j .，& Wojna，Z. (2016)。重新思考计算机视觉的初始架构。 *2016 年 IEEE 计算机视觉与模式识别大会(CVPR)* 。[https://doi.org/10.1109/cvpr.2016.308](https://doi.org/10.1109/cvpr.2016.308)