# 超参数调整——超点贝叶斯优化(Xgboost 和神经网络)

> 原文：<https://medium.com/analytics-vidhya/hyperparameter-tuning-hyperopt-bayesian-optimization-for-xgboost-and-neural-network-8aedf278a1c9?source=collection_archive---------0----------------------->

![](img/64700a5ad720575cebbb98dbb453f9ad.png)

**超参数:**这些是确定算法学习过程的某些值/权重。

**机器学习模型的某些参数:**学习速率、alpha、最大深度、col-samples、权重、gamma 等。

**深度学习模型的某些参数**:单元(单元数)、层(层数)、辍学率、核正则化子、激活函数等。

**超参数优化**是为机器学习/深度学习算法选择最优或最佳参数。通常，我们最终用各种可能的参数范围手动调整或训练模型，直到获得最佳拟合模型。超参数调整有助于确定最佳调整参数并返回最佳拟合模型，这是构建 ML/DL 模型时要遵循的最佳实践。

在本节中，我们讨论一种最精确和最成功的超参数方法，即**超视。**

优化无非是找到一个最小的成本函数，这决定了一个模型在训练集和测试集的整体性能更好。

这是一个功能强大 python 库，可以搜索超参数值空间。它实现了三个函数来最小化成本函数，

1.  随机搜索
2.  树 Parzen 估计量
3.  适应性 TPE

**导入所需包:**

```
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
```

远视功能:

*   `hp.choice(label, options)` —返回选项之一，应为列表或元组。
*   `hp.randint(label, upper)` —返回范围[0，上限]之间的随机整数。
*   `hp.uniform(label, low, high)` —统一返回一个介于`low`和`high`之间的值。
*   `hp.quniform(label, low, high, q)` —返回值 round(uniform(low，high) / q) * q，即舍入小数值并返回整数
*   `hp.normal(label, mean, std)` —返回具有平均值和标准差σ的正态分布的实数值。

**机器学习算法 hyperopt-XGBOOST 中涉及的步骤:**

**步骤 1:** 初始化空间或一个所需的值范围:

**第二步:**定义目标函数:

**第三步:**运行远视功能:

这里，“最佳”为您提供最适合模型的最佳参数和更好的损失函数值。 **trials** ，它是一个包含或存储所有统计和诊断信息的对象，如超参数、模型已训练的每组参数的损失函数。 **fmin** ，它是一个最小化损失函数的优化函数，接受 4 个输入。使用的算法是' **tpe.suggest** ，其他可以使用的算法是' **tpe.rand.suggest** 。

> ***使用使用 hyperopt 获得的最佳参数重新训练模型算法，并根据测试集对其进行评估，或将其用于预测***

**深度学习算法/神经网络的 hyperopt 中涉及的步骤:**

**步骤 1:** 初始化空间或一个要求的数值范围:

**第二步:**定义目标函数:

**第三步:**运行远视功能:

> ***使用使用 hyperopt 获得的最佳参数重新训练模型算法，并根据测试集对其进行评估，或将其用于预测***

# **结论:**

我们已经讨论了如何使用 sklearn python 库“hyperopt ”,这是数据科学领域中广泛首选的库。超参数调整是建立学习算法模型的重要步骤，需要仔细检查。另一个用于神经网络超参数调整的不同 python 库是**‘hyperas’。关于这篇文章的任何问题，欢迎在下面评论。**