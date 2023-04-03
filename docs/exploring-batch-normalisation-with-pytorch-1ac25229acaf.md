# 用 PyTorch 探索批量标准化

> 原文：<https://medium.com/analytics-vidhya/exploring-batch-normalisation-with-pytorch-1ac25229acaf?source=collection_archive---------7----------------------->

继我之前的[帖子](/analytics-vidhya/transforming-data-in-pytorch-741fab9e008c)之后，在这篇帖子中，我们将讨论“批处理规范化”及其在 PyTorch 中的实现。

批量标准化是一种用于提高神经网络效率的机制。它的工作原理是稳定隐层输入的分布，从而提高训练速度。

![](img/9b478f36dc17bd2a1bbb393ca7dc0700.png)

# 1.批量标准化的本质