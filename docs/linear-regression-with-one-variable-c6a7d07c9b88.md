# 一元线性回归

> 原文：<https://medium.com/analytics-vidhya/linear-regression-with-one-variable-c6a7d07c9b88?source=collection_archive---------25----------------------->

## **用梯度下降法和正规方程预测**

线性回归是一种**线性模型**，例如，假设输入变量(x)和单个输出变量(y)之间存在线性关系的模型。更具体地说，y 可以从输入变量(x)的线性组合中计算出来。

在本文中，我们将使用梯度下降来拟合数据集的线性回归参数θ。

数据集取自[这里](https://storage.googleapis.com/kaggle-data-sets/8834/12327/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20201031%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20201031T062320Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=2bdaab426616f3ca61ffe408b382f5bcabf0fd6ff4b9de913750c1087d7a98d5cdf6422361f0556c63fbe85ed27c02cd1efd00f9f411dfef9521ae992a3c8cda7c3e8033905484ad17bf346c6cffd074409dc832d957397190ce755d6556f9d239bed6687fbad708d81be9620e9c57af176e0a3395b108cf21dccef8516ffa3bc962f2b63e55e6eb24d58c3732474d5ed68d42a8ae1f45bf496c1d4175ee6d52e4b2c12b68684a2f23fb9fe96307d8b15acc1a6ee8f365f2ae290bf68a9d72f3ab26af5be627b8a7b4a6f9b93ba35133d0c2faa48857dc83ad50e2d58a48a8bc58f1ee35d2b1c5106f4a8bb47ab4c27d4d1706ea2e67bdbcd2b2ff08c92a7e72)

首先，让我们导入我们需要的库

![](img/e80e017e824aaa7696a93390b2fe334e.png)

我们来看看数据

![](img/dc9f7743847beef7104b7d649c12c628.png)

我们定义输入变量(X)和输出变量(y)

![](img/8f6553247f34841b85e53ae935ed90e9.png)

在开始任何任务之前，通过可视化来理解数据通常是有用的。对于此数据集，您可以使用散点图来可视化数据，因为它只有两个属性可以绘制(利润和人口)。

![](img/1ff02da568ea0153dfd132cba8d77f99.png)

# **梯度下降**

在某种程度上，我们将使用梯度下降将线性回归参数θ拟合到我们的数据集。

**更新方程式**

线性回归的目标是最小化成本函数

![](img/6cd479e90a2c73514d804f41b5dde976.png)

其中假设 hθ(x)由线性模型给出

![](img/1bca538bc4c88e0a15413b7b8d1b40a1.png)

模型的参数是θj 值。这些是您将调整以最小化成本 J(θ)的值。一种方法是使用批量梯度下降算法。在批量梯度下降中，每次迭代执行更新。

![](img/0c13ee926f76b58c04da1bc982a73e56.png)

**实施**

我们已经为线性回归建立了数据。在下面的单元格中，我们为数据增加了另一个维度，以适应θ0 截距项。不要多次执行此单元格。

![](img/db1d02ffcdfee7c6e63cc03fc7175643.png)

**计算成本 J(θ)**

当您执行梯度下降来学习最小化成本函数 J(θ)时，通过计算成本来监控收敛是有帮助的。在本节中，您将实现一个计算 J(θ)的函数，以便检查梯度下降实现的收敛性。

![](img/f1f6af5575a4d3aa478120e9e85b14ab.png)

**梯度下降**

接下来，您将实现梯度下降

![](img/b18d1faa7420d8bafb526a02133d1c80.png)

在您完成调用实现后，我们将θ参数初始化为 0，迭代初始化为 2000，学习速率α初始化为 0.5。

![](img/02642ecfc3d274c1944de5e9da3a9b43.png)

我们将使用最终值θ来预测身高 1.52 的体重

![](img/bd64d89617e1f02a18ba60fcd3e558e7.png)

# 正态方程

线性回归的封闭解是

![](img/d553958bc95792c8b14e9afc2e33a0b9.png)

使用这个公式，你将在一次计算中得到精确解:不存在像梯度下降中那样的“循环直到收敛”。

![](img/ca3a6052b0b7a2f685d5d6ff57e68b45.png)

我们计算θ

![](img/47431dd8d3a05e9e997460be0f13a0e4.png)

我们将使用最终值θ来预测身高 1.52 的体重

![](img/34da52b09ddf59830146ba1a7bbca33f.png)

# 与 Sklearn 线性回归比较

现在我们使用 sklearn 将其与线性回归进行比较

我们首先确定特征和目标

![](img/f5c30185960efff8ff3c441e630ccd6f.png)

然后我们使用 sklearn 进行线性回归

![](img/f3af8d3b87387321e4b95b5978a33808.png)

我们的截距和系数，我们将用它来预测身高 1.52 的体重

![](img/af4f2f1094797669223f155f55ce1ab2.png)

从上面的结果可以看出，对于截距和系数，正规方程和线性回归 sklearn 之间的计算结果是相同的。然而，使用梯度下降的计算给出不同的结果。

为了克服这一点，有必要将迭代和学习速率α设置为不同的值，以便可以获得与使用正规方程和 sklearn 线性回归的计算结果接近的θ值。

谢谢你