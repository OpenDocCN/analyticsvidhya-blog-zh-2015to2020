# 人工神经网络，第 3 部分—使用 Keras 和 Tensorflow 实现神经网络

> 原文：<https://medium.com/analytics-vidhya/artificial-neural-networks-part-3-implementing-neural-networks-with-keras-and-tensorflow-39cbf6fde7d?source=collection_archive---------21----------------------->

![](img/6551bf813279d98092475647d22475b3.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Max Duzij](https://unsplash.com/@max_duz?utm_source=medium&utm_medium=referral) 拍照

在本系列的前几篇文章中，我们讨论了以下主题

1.  [人工神经网络——人工神经网络背后的概念](/analytics-vidhya/artificial-neural-networks-part-1-d36fb7bce6bb?source=your_stories_page---------------------------)
2.  [梯度下降](/analytics-vidhya/understanding-gradient-descent-without-the-math-bc31a4781c88?source=your_stories_page---------------------------)

这篇文章将介绍使用 Tensorflow 和 Keras 实现一个基本的神经网络。

这个问题是基于 UCI 机器学习知识库上的自行车共享数据集，可以在这里找到—

[自行车共享数据集](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

zip 文件中提供了两个数据集， **daily.csv，**和 **hour.csv.**

在这个问题中，我使用了 daily.csv 文件，它是 hour.csv 文件的汇总。

这个问题包括预测出租自行车的数量，包括每天的数量。

让我们从实施步骤开始——

第一步从导入所需的库和读取数据集开始

![](img/e57b93b273311959dc8caceeea2700dd.png)

现在，我们已经将 CSV 文件作为 pandas 数据帧导入，让我们看看可用列的形状和细节。

![](img/1d41d194432a1daf887ff098d34ae52f.png)

有 731 条记录和 16 列，包括我们的目标变量“cnt ”,它是每日租赁自行车的计数。每列的详细信息可以在上面提供的 UCI 数据集链接中找到。

看看每日数据集的前几行——

![](img/a9c55bf8a4b76d7c1c7a37e974a17b90.png)

我们可以看到，列 season、mnth、weekday、work day 和 weathersit 的值是作为数字提供的，因此被视为数字列。如果我们将这些转换为用于 EDA 的一些可视化的分类，这将是有益的。

首先，我们将把相关性绘制成热图。

![](img/5e2da5044c81a8680bcc61c08664d9d1.png)

作为对值的第一印象，可以推断出变量与目标的相关性不是很高，除了变量‘registered’**的值为 0.95。**

对于 EDA，让我们从单变量可视化开始。

![](img/2272a3b5b603eb2322efecb6a2cca2a5.png)

目标变量不包含任何负值或任何异常值。

![](img/d20735085041b45afa3bdb22ec71c541.png)

转到独立变量，

1.  假日— 0(否)，1(是)

![](img/b8ae1c9539b00b3dc21ed94d670d1387.png)

2.工作日— 0(否)，1(是)

![](img/081b95532328cab2455f3739004e8602.png)

3.Weathersit —

![](img/fa2a1569f25f7641c29e6d00e28daf57.png)

用自变量绘制目标变量——

![](img/e7566c9037b7aba88a48ce5bea7e35fc.png)

平均使用率较高的月份是 6 月至 9 月，即 6 月至 9 月。

![](img/6f76b124a2fd944ea4fc6bc38862909d.png)

同样，更多的使用是在第二季和第三季，即春季和夏季。

根据目标确定数字列之间的关系和行为。

![](img/80229dd7109294364331a69d488a845d.png)

正如我们之前看到的，注册和总乘坐次数之间有很高的相关性。

![](img/2a8be62a61d7fd005c3f9e401bf7e86e.png)

在中温范围内使用更多

![](img/294a3f83fa7888c61392f0757e9388dc.png)

显然，atemp 和 temp 变量高度相关。我们可以去掉其中一个变量。

基于 EDA，我们将放弃以下变量—

dteday，instant，atemp，yr，已注册

![](img/7c72674493de130ca3c605e5ec37a827.png)

在 EDA 之后，我们将建立数据集来训练和测试模型。

![](img/3ce42109de910a1be71cf6f9da10f6e2.png)

编码分类数据—

我们也可以使用 sklearn 的 OneHotEncoder 库，但这里我使用 pandas getdummies()和一小段代码片段来保持编码后训练集和测试集中的列相同。

![](img/e8a9f02df05a0082b09cc751e00d7a83.png)

缩放值以保持在 0 和 1 的相同范围内—

![](img/3f50c3151a245eb936e56158ac0bc13e.png)

导入用于构建网络的库

![](img/b8e05a77a72f44f22215f3f79ff640dc.png)

我们正在导入顺序类来创建将被堆叠到层中的模型。

密集类将创建深度连接的神经网络层。这一层将对输入执行操作，应用激活函数，并将结果传递给下一层。

构建网络—

![](img/8616611d8ab0fa425d45ec1b99dca2e2.png)

第一层是具有 34 个结点的密集层，这基本上是输入层，值 34 来自处理后数据集中的要素数。

![](img/5cd7465a3779fbfb45de7372c1926029.png)

第二、第三和第四层是隐藏层，分别包含 34、34 和 10 个节点。最后，我们添加一个节点，这就是我们的输出。我们的输出是一个数值，否则，在分类的情况下，它将是目标中类的数量。

编译模型—

![](img/311508f2d8496e525ded1e9d825a1cab.png)

这里的损失函数是 MSE，均方误差。我们会尽量减少这种情况。这将是实现使用亚当优化。优化器致力于最小化损失函数的值。

最后，在我们的训练数据上拟合模型。我们还通过了作为验证集的测试数据。该过程将不断计算训练集和测试集的损失。纪元的数目是 1000，这是我任意选择的。我们可以利用像提前停止这样的方法，这将在使用损失函数的最小值时停止训练。考虑到我们的训练集和测试集中的记录数量较少，批大小为 64。

![](img/83eadc8305c9c1a20529cdefc47e83de.png)

可视化损失值—

![](img/870794a0d216e4afc662a668bc5d143f.png)

我们可以看到，验证损失和训练损失的值在第 400 个时期达到最小。我们仍然可以查看更多的历元数，看看验证损失何时开始增加，我们何时开始过度拟合。但是对于这个博客，我们将坚持 1000 个时代。

让我们快速进入预测和计算指标。

![](img/9a527ec6dfaf989ad0b403aea396ab03.png)

均方误差约为 70k，平均绝对误差为 703。我们的模型能够解释 83%的差异。这很好，但我们仍然可以改进并获得更低的 MSE。

让我们绘制预测值与实际值的对比图，看看它们相差有多远

![](img/69ed9b05a0f78e9b25c7c4eafce74db9.png)

不是很远。这看起来不错。

所以，这是一个关于我们如何用 Keras 和 Tensorflow 实现神经网络的快速帖子。

我希望这有所帮助，我将感谢任何关于如何改进模型的反馈和建议。