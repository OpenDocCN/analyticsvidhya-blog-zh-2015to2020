# Python 中的朴素贝叶斯分类器，从头开始

> 原文：<https://medium.com/analytics-vidhya/naive-bayes-classifier-in-python-from-scratch-eacf52a8d43f?source=collection_archive---------11----------------------->

![](img/7fb6ecf3687661cb1d0380a65cde0920.png)

马克·科赫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

在本文中，我们从头实现了朴素贝叶斯分类器，是的——直接从零开始，没有库。可能有许多库在一两行代码中实现了这一点，但这不是我们在这里寻找的。这篇文章是为了加强你对这个主题的理解，除了用泥和粘土做建筑，还有什么更好的方法呢？还有一件事，我不会在这里把重点放在理论上，你可以看看 [**这个**](/@srishtisawla/introduction-to-naive-bayes-for-classification-baefefb43a2d) 如果需要修改，会是一个很棒的阅读！

场景是这样的，我们得到了一个 csv 数据集，其中有 10 名怀疑感染了新冠肺炎病毒的患者。我们有五种健康状况，即。发烧了？，有咳嗽吗？有呼吸道不适？，流鼻涕？和喉咙痛——分别用 0 或 1 表示。最后一列是测试结果，也包含 0 表示阴性，1 表示阳性。这是数据，

![](img/ff56421abb3fc823bc91ef12f0017f63.png)

现在我们需要用朴素贝叶斯来训练算法，最后给出一个长度= 5 的元组(实际上是一个数组)，其中索引为的*中的值(本质上是 0 或 1)表示上述症状为*的*。我们将根据列车数据返回 0 或 1，符号带有通常的含义。*

![](img/e96571061b342e6223e1c1efca44b8cf.png)

这里，我们已经将 csv 加载到程序中，并转换为数据帧，进而转换为 2D 数组。

![](img/29043dc91ddd073e8f88a0bb6218a579.png)

这里我们计算正和/或负的总概率。

![](img/4e95d4295afad7f14d7b13e1cc840a9c.png)

给定测试的阳性和/或阴性，该块找出每个特征的单独概率。结果被汇总到一个名为 prob 的字典中，其中 pos 表示假设测试结果为正，相应特征为真的概率，反之亦然。

![](img/1641d4e79af03f0a0c73c204488fdb1b.png)

一旦确定了个体概率，就需要计算给定特征的正和/或负的总概率。这是一个遍历字典和增加特征的循环。

![](img/1f4fffedadfec6831ab5c7132a1b2d08.png)

给定手中特征的总概率，两个概率都需要除以 [**证据号**](https://www.geeksforgeeks.org/naive-bayes-classifiers/#:~:text=Bayes'%20Theorem,-Bayes'%20Theorem%20finds&text=Basically%2C%20we%20are%20trying%20to,%2C%20it%20is%20event%20B).) ，这是为了归一化概率。最后一件事是比较阳性和阴性结果的标准化概率，如果阳性>阴性，则返回阳性(在这种情况下为 1)，表明该人被预测诊断为疾病阳性，否则为阴性。

case = [1，0，0，1，1]的情况返回 1，意味着预测是肯定的。现在我已经在 Jupyter 笔记本上完成了，这里的每个部分都是编辑器的模块。这里有一个 [*链接*](https://gist.github.com/Hussain-Safwan/551cbc1c8e21c8b9934bbe38422f8e5e) 到完整的代码和数据集。

编码快乐！