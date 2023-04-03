# 在 Pytorch 中使用自动编码器对 MNIST 数据集进行维度操作

> 原文：<https://medium.com/analytics-vidhya/dimension-manipulation-using-autoencoder-in-pytorch-on-mnist-dataset-7454578b018?source=collection_archive---------4----------------------->

*让我们把理论和代码联系起来学习吧！*

现在，根据[深度学习书籍](https://www.deeplearningbook.org/contents/autoencoders.html)，自动编码器是一个神经网络，它被训练成旨在将其输入复制到其输出。在内部，它有一个隐藏层，描述用于表示输入的代码。该网络可以被视为由两部分组成:编码器函数“h=f(x)”和产生重构“r=g(h)”的解码器。