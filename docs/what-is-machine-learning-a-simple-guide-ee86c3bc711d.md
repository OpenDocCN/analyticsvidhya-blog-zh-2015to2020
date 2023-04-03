# 什么是机器学习？简单的指南

> 原文：<https://medium.com/analytics-vidhya/what-is-machine-learning-a-simple-guide-ee86c3bc711d?source=collection_archive---------29----------------------->

## 以及带有示例的算法概述。

![](img/6b7f2dd718a9aed4dd96bb2e11a9c4fd.png)

鸣谢: [@askkell](https://unsplash.com/@askkell) 来自 [Unsplash](https://unsplash.com/)

# 介绍

什么是机器学习:机器学习是教会计算机系统如何在输入数据时做出准确预测的过程。

*让我们用一个例子来思考这个定义。*
什么是机器学习？我们有这个世界，在这个世界里，我们有人类，我们有计算机，人类和计算机的主要区别之一是人类从过去的经验中学习，而计算机需要被告知做什么，它们需要被编程。现在的问题是，我们能让计算机也从经验中学习吗？答案是肯定的，我们可以。而这正是机器学习。

教计算机从过去的经验中学习执行任务。当然，对于计算机来说，过去的经验只是作为数据记录下来。

# 创建好的机器学习系统需要什么？

1)数据准备能力
2)算法—基本和高级
3)自动化和迭代过程
4)可扩展性
5)整体建模

# 机器学习的类型

机器学习算法的类型在它们的方法、它们输入和输出的数据的类型以及它们打算解决的任务或问题的类型方面不同。

# 监督学习

监督式学习顾名思义是指作为老师的监督人的存在。基本上，监督学习是一种学习，在这种学习中，我们使用标记良好的数据来教授或训练机器，这意味着一些数据已经标记了正确的答案。之后，向机器提供一组新的示例(数据)，以便监督学习算法分析训练数据(训练示例集)并从标记数据产生正确的结果。

现在你一定想知道，有监督学习下的算法类型。

下面提到一些主要的重要的和基本的算法！

1.  线性回归

```
>>> from sklearn.linear_model import LinearRegression
>>> model = LinearRegression()
>>> model.fit(x_values, y_values) 
```

2.感知器算法

```
>>> from sklearn.datasets import load_digits
>>> from sklearn.linear_model import Perceptron
>>> X, y = load_digits(return_X_y=True)
>>> clf = Perceptron(tol=1e-3, random_state=0)
>>> clf.fit(X, y)
```

3.决策树和随机森林

```
>>> from sklearn.tree import DecisionTreeClassifier
>>> model = DecisionTreeClassifier()
>>> model.fit(x_values, y_values)
```

4.朴素贝叶斯

```
>>> from sklearn.datasets import load_digits
>>> from sklearn.linear_model import Perceptron
>>> X, y = load_digits(return_X_y=True)
>>> clf = Perceptron(tol=1e-3, random_state=0)
>>> clf.fit(X, y)
```

5.SVM(支持向量机)

```
>>> from sklearn.svm import SVC
>>> model = SVC()
>>> model.fit(x_values, y_values)
```

6.集成方法(AdaBoost)

```
>>> from sklearn.ensemble import AdaBoostClassifier
>>> model = AdaBoostClassifier()
>>> model.fit(x_train, y_train)
>>> model.predict(x_test)
```

# 无监督学习

无监督学习是使用既未分类也未标记的信息训练机器，并允许算法在没有指导的情况下对该信息进行操作。在这里，机器的任务是根据相似性、模式和差异对未分类的信息进行分组，而无需任何事先的数据训练。

下面提到一些主要的重要的和基本的算法！

1.  **聚类:** k 均值，层次聚类分析

```
>>> from sklearn.cluster import KMeans
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [10, 2], [10, 4], [10, 0]])
>>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
>>> kmeans.labels_
array([1, 1, 1, 0, 0, 0], dtype=int32)
>>> kmeans.predict([[0, 0], [12, 3]])
array([1, 0], dtype=int32)
```

1.  **关联规则学习:** Eclat
2.  **可视化与降维:**核主成分分析，t 分布，主成分分析

# 应用程序

机器学习被用于许多很酷的事情，如图像或语音识别、面部识别、垃圾邮件检测、欺诈检测、股票市场、教计算机如何下棋或任何游戏、无人驾驶汽车、虚拟现实耳机，当然还有许多其他东西。

# 结论

让我们通过提及在线学习机器学习的最佳方式或者你可以说最佳资源来结束这篇文章。

[](https://towardsdatascience.com) [## 走向数据科学

### 共享概念、想法和代码的媒体出版物。

towardsdatascience.com](https://towardsdatascience.com) [](http://machinelearningmastery.com/) [## 机器学习掌握-机器学习掌握

### 让开发者在机器学习方面变得令人敬畏。

machinelearningmastery.com](http://machinelearningmastery.com/) [](https://medium.com/analytics-vidhya) [## 分析 Vidhya

### 分析 Vidhya 是一个由分析和数据科学专业人士组成的社区。我们正在构建下一代数据科学…

medium.com](https://medium.com/analytics-vidhya) 

**鼓励我在这个帖子上发表更多关于机器学习的东西。希望你觉得它很有见地。**

谢谢你。