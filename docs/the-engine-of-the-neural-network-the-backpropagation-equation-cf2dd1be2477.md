# 神经网络的引擎:反向传播方程

> 原文：<https://medium.com/analytics-vidhya/the-engine-of-the-neural-network-the-backpropagation-equation-cf2dd1be2477?source=collection_archive---------3----------------------->

![](img/a604981b6b7ad231f82aa923b772fc99.png)

艾莉娜·格鲁布尼亚克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

虽然有了 TensorFlow 和 PyTorch 这样的库，程序员可以在不理解背后的数学知识的情况下创建一个强大的神经网络，但理解保持复杂的卷积神经网络和生成对抗网络运行的简单方程是重要的。该方程揭示了神经网络如何学习的本质。