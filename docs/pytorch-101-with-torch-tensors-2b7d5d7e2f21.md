# PyTorch🔦101，第一部分:火炬张量

> 原文：<https://medium.com/analytics-vidhya/pytorch-101-with-torch-tensors-2b7d5d7e2f21?source=collection_archive---------27----------------------->

![](img/6f0fbaf454fb9ccd93f354766e7f0156.png)

PyTorch 是一个令人惊叹的 python 库，它在构建深度学习中发挥了重要作用。Pytorch 和 Tensorflow 一样有很高的社区支持。在这个脸书人工智能产品中有各种我想剪切的实用程序，

1.  这个库在语法上与 python 非常相似，因为它倾向于将 python 作为它的主要编程语言。您会发现与各种通用库(如-Numpy)在语法上有许多相似之处。现在，这一点把我们带到了下一个简单易学的点。
2.  **易学，**由于语法与 python 非常相似，开发者学习起来更容易。并且文档组织得当，很容易在那个[文档](https://pytorch.org/docs/stable/index.html)中找到你要找的东西，PyTorch 有一个巨大的[社区。](https://discuss.pytorch.org/)
3.  PyTorch 有一个非常有用的特性叫做[数据并行](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)。使用这个特性，PyTorch 可以在多个 CPU 或 GPU 核心之间分配计算工作。PyTorch 的这个特性允许我们使用`torch.nn.DataParallel`来包装任何模块，并帮助我们在批处理维度上进行并行处理。

这样的例子不胜枚举。

本教程基本上涵盖了不同的 PyTorch 张量函数，当您使用 Pytorch 执行不同的深度学习任务时，这些函数非常重要。实际上，任何 PyTorch 模型(如-ann、CNN)或操作都不能处理任何不是 Torch 张量的数组。因此，为了转换这些数组，特别是我正在考虑的 NumPy 数组，我们需要使用一些函数，还有一些奇妙的和 sssuuupppeeerr 简单的函数用于数学运算和一些深度学习实用程序。那么，说了这么多，让我们从目录开始吧

## 目录:

1.  火炬。张量()
2.  torch.ones()
3.  火炬.零点()
4.  torch.rand()
5.  火炬.手动 _ 种子()
6.  将火炬张量从 CPU 移动到 gup，反之亦然
7.  火炬张量的大小转换
8.  数学实现

## 导入所需的库:

# 火炬。张量()

这是一个 PyTorch 函数，它帮助我们创建 PyTorch 张量或数组。我们只需要在火炬内部传递一个 NumPy 数组或者一个 list。张量()然后嘣，你的 PyTorch 张量就好了。在这里，我首先创建了一个列表(6 号单元格)，然后创建了一个 NumPy 数组(7 号单元格)，之后，我将列表和混乱转换成火炬张量。

**torch.from_numpy()** 是另一种将 numpy 数组转换成 torch 张量的方法，我们只需要在那个函数中传递那个 NumPy 数组就可以了。

我们几乎涵盖了从 NumPy 数组和列表创建 torch 张量的所有方面，如果我们需要从 torch 张量创建一个 NumPy 数组呢？很简单，我们只需要加上**。numpy()** 同火炬张量。

# torch.ones()

这个函数创建一个 1 的数组，我们只需要提供数组大小(比如- 3x3 或 2x2)就可以创建一个数组。这个函数类似于我们在 NumPy 中的函数，即 **np.ones()** 。在 **torch.ones()** 的例子中可以看到语法也是类似的。

因此，在 PyTorch 中，我们有一个函数，它不仅在功能上相似，在语法上也相似，而且它不是 PyTorch 中唯一一个与 NumPy 函数相似的函数，还有其他函数。

# torch.zeros():

这个函数创建一个由零组成的数组，我们只需要传递数组的大小。这个函数也类似于 NumPy 函数 **np.zeros()** 。

# torch.rand():

这个函数创建了一个用随机数填充的数组，我们只需要传递数组的大小。它类似于 NumPy 函数 **np.random.rand()** ，也做类似的工作。

# torch.manual_seed():

种子是编码深度学习架构时非常重要的元素，改变种子可以改变深度神经网络的结果或输出。在 NumPy 中，我们使用 n **umpy.random.seed()** 进行播种，但是在 PyTorch 中，我们有 **torch.manual_seed()** 应用种子，这里我们只需要在 **torch.manual_seed()** 中传递任意随机数，这将防止一次又一次地改变输出。

注意，当我们每次传递相同的播种数并创建相同的数组时，我们每次都得到相同的输出(查看单元格 17 和 18)。但是，当我们改变播种的数量时，输出也会改变(单元格 19)。

# 参考:

1.  [https://pytorch.org/docs/stable/tensors.html](https://jovian.ml/outlink?url=https%3A%2F%2Fpytorch.org%2Fdocs%2Fstable%2Ftensors.html)
2.  [https://heart beat . fritz . ai/10-reasons-why-py torch-is-the-deep-learning-framework-of-future-6788 BD 6b 5 cc 2](https://heartbeat.fritz.ai/10-reasons-why-pytorch-is-the-deep-learning-framework-of-future-6788bd6b5cc2)
3.  笔记本链接—[https://jovian.ml/soumya997-sarkar/torch-basics](https://jovian.ml/soumya997-sarkar/torch-basics)

## 感谢您的阅读👩‍💻👨‍💻