# 神经网络预测自行车共享骑行

> 原文：<https://medium.com/analytics-vidhya/neural-network-to-predict-bike-sharing-rides-397e0358ba45?source=collection_archive---------8----------------------->

![](img/5b9922824eeb7d38b4e05993f2e3c1ce.png)

在本文中，我们将讨论从头构建一个神经网络，使用真实数据集进行预测。这种方法将帮助我们更好地理解神经网络的概念。数据集可在以下位置找到:

[https://archive . ics . UCI . edu/ml/datasets/Bike+Sharing+Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

我们将创建两个文件:Jupyter Notebook 和 my_answers.py，它们将实际包含我们的神经网络和超参数。

1.1 我们开始用木星笔记本预测 _ 自行车 _ 共享 _ 数据。我们执行所有必要的进口:

![](img/6b7b863c0a2d43330af0885ddb379750.png)

第`*%matplotlib inline*`行将 matplotlib 图形放在代码旁边。

*下两行:*

```
%load_ext autoreload
%autoreload 2
```

每次执行新的一行之前，都会重新加载所有更改过的模块。

`%config InlineBackend.figure_format = ‘retina’` 的目的是增加我们绘图的分辨率。

`import numpy as np` 是用 Python 包导入的基础科学计算。

*import pandas as pd* 是 Python 库的导入，用于数据操作和分析。

`import matplotlib.pyplot as plt` 导入模块“matplotlib.pyplot”，pyplot 是 matplotlib 的绘图框架。

1.2 我们用`rides = pd.read_csv(data_path)`**读取数据集用 *rides.head()* 看数据帧的前 5 行**

**![](img/b1bdefab08bb3bba17c363bdebcfd7b1.png)**

**1.3 让我们来看一个图表，显示数据集中前 10 天左右骑自行车的人数。并不是每天 24 个条目。周末的客流量较低，而当人们骑自行车上下班时，客流量会出现高峰。**

**![](img/82d9c8924bce5610cdd4fd9bbddf2029.png)**

**1.4 我们用熊猫`get_dummies()`创建二进制虚拟变量**

**![](img/12f107640c4449c4b9e46163ac50ea41.png)**

**1.5 我们通过移动和缩放变量来标准化每个连续变量，使其平均值为零，标准差为 1。**

**![](img/45ef999882912d7dd4f60a957bf004c0.png)**

**1.6 我们将数据分为训练集、测试集和验证集。**

**![](img/4c8a98a2447dadc22686d7f05ec88c61.png)**

**2.1 现在我们要在 my_answers.py 文件中构建我们的网络。在我们的类 NeuralNetwork(object)中，我们定义了构造函数方法 __init__。我们设置输入层、隐层和输出层的节点数，初始化权重，将 self.activation_function 设置为 sigmoid 函数。**

**![](img/3b5d67cf2194fbeb26baae79d9869396.png)**

**2.2 我们在 forward_pass_train 中实现前向传递，当我们在网络的各层中工作时，计算每个神经元的输出。我们使用初始化的权重将信号从神经网络的输入层向前传播到输出层。输出层只有一个节点，它用于回归，因此节点的输出和节点的输入是相同的。**

**![](img/e2f755565ae79df80760ec03ce8a1567.png)**

**2.3 我们在反向传播中实现反向传播。我们还使用权重将误差从输出反向传播回网络，以更新我们的权重。**

**![](img/da48474bd7be6af21b1a800508acf669.png)**

**2.4 我们在 update_weights 中更新梯度下降步骤上的权重。**

**![](img/6d1172c10525baa0c9763d8cd7672e39.png)**

**2.5 我们在`run`方法中实现向前传递。**

**![](img/9fa00f51d742289c91603fab0770bb15.png)**

**2.6 我们定义了用于训练神经网络训练。**

**![](img/46af465e2b544b4f5ecd4653d4989673.png)**

**3.1 我们回到 Jupiter 笔记本 Predicting_bike_sharing_data，在这里导入我们的神经网络。**

**![](img/4e6215d100c845a5be0a64c88af565d3.png)**

**3.2 我们运行这些单元测试来检查我们的网络实现的正确性。**

**![](img/4f758b5dddb8b306c8a12d578761e059.png)****![](img/be21553782ef99a21841d6a8edd71c4f.png)**

**4.1 我们要训练我们的神经网络。我们在您的 myanswers.py 文件中设置超参数**

**![](img/263948be8e8fcbbe9d6fa42a19ec824f.png)**

**设定超参数时需要考虑的事情**

**![](img/fc1bf6d821875e448376f9bf21593eaf.png)**

**4.2 我们将它们导入我们的笔记本**

**![](img/d04825e3a896deef76793c2f98aa33c0.png)**

**4.3 我们训练我们的网络。**

**![](img/483fec0e5deff714f94c59be9fe446e7.png)****![](img/06f786da0f62357f50c64d5b0fd65949.png)**

**4.4 最后，我们检查我们的预测**

**![](img/8798dfd05943e170e98a994981641b1e.png)**

**您可以在这里找到完整的代码:**

**[](https://github.com/forfireonly/Bikesharing_neural_network) [## for fire only/bike sharing _ neural _ network

github.com](https://github.com/forfireonly/Bikesharing_neural_network) 

快乐编码我的朋友们！**