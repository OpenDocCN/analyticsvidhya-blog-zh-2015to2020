# 让它在 Kaggle 的 MNIST 数字识别器中从零开始达到最高的 6 %- 2。

> 原文：<https://medium.com/analytics-vidhya/getting-it-to-top-6-in-kaggles-mnist-digit-recognizer-from-scratch-2-815869d643a2?source=collection_archive---------38----------------------->

# 第 2 部分:卷积神经网络

![](img/ab0cd0dac7ebad3f4b041dac0760cc89.png)

演职员表:[塞尚·卡马乔](https://cezannec.github.io/)

谈到图像数据集，CNN 是最好的选择。我不想详细介绍 CNN 是如何工作的，你可以在这篇文章中读到更多:[https://adeshpande 3 . github . io/A-初学者% 27s-理解指南-卷积神经网络/](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/)

让我们在 TensorFlow 中创建一个小型 CNN，并与我们简单的全连接神经网络进行比较，看看它是如何工作的。

我们可以保持神经网络的基本结构(1.1 部分)不变，在此基础上，我们可以继续添加卷积和池层的组合。你可以看到，一个图像通过模型的旅程使用了*。*

*使用上面的简单 CNN 模型，训练精度达到 0.9924，而 Kaggle 测试得分为 0.98857，排名为 925，这是从第 1 部分开始通过我们的 NN 的几乎 400–500 的跳跃。*

*随着模型复杂性的增加，我们甚至将模型的精度提高到 0.9901，等级大约为 800。我们现在领先于一半以上的竞争对手，但我们还有很多可以改进的地方。你可以玩层数，内核大小，过滤器，看看你是否能找到运气。*

> *在接下来的[****部分(系列之三)****](/@rushikesh0203/getting-it-to-top-6-in-kaggles-mnist-digit-recognizer-from-scratch-3-8b11b79958a2) *中，我们将尝试从排名 800 达到 220，即**前 10%** ，然后一路做到**前 6%** 。**