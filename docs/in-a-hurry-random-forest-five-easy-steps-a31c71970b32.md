# 赶时间！随机森林:五个简单的步骤

> 原文：<https://medium.com/analytics-vidhya/in-a-hurry-random-forest-five-easy-steps-a31c71970b32?source=collection_archive---------28----------------------->

## 随机森林中发生了什么——以及建立模型的五个简单快捷的步骤。

![](img/7db89a1a084099bf49c90bf8b2db5b2d.png)

照片由[维姆·范因德](https://unsplash.com/@wimvanteinde?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

这个博客是我对监督机器学习的随机森林技术的理解的一部分。

随机森林是分类问题中最常用的技术之一。

决策树是随机森林的构建块。

在每一个节点上，决策树会问，什么样的特征可以让我分割出尽可能不同的结果组。例如蓝色和红色的 1 和 0，第一个节点基于颜色“它是红色的吗？”，那么如果进一步，我们想要分类 0 或 1，进一步的节点 spits 基于‘它是 0 吗？’

分割是根据基尼指数进行的。

随机森林由大量作为整体运行的决策树组成。每棵树代表他们的类，多数票成为 RF 的预测类。

这些个人决策树最好的部分是，它基于 bagging 概念工作，并且它们是低相关的，因此，每棵树都保护彼此免受各自错误的影响。

低相关性是可能的，因为-RF 模型选择随机观察作为每棵树的训练样本。此外，每棵树选择的特征是随机的。

下面是应用随机森林模型的几个快速检查点步骤，

**Step1:** 使用库(randomForest)构建 randomforest()模型。请确保目标要素是因子或将其转换为 as.factor()函数。

**步骤 2:** 运行模型并检查 OOB(出袋)误差。OOB 依赖于树的数量和每棵树选择的特征的数量。[也可以绘制 OOB 图表]

**第三步:**为了增加单个树的强度(或准确性),检查最低错误率，并相应地选择随机树的大小。

**步骤 4:** 使用 tuneRF()函数调优随机森林模型。

**步骤 5:** 对测试数据集进行预测，使用函数 predict()。

我借鉴了饶彤彤的《理解随机森林》。

对于实际应用以及如何对 Titanic 案例进行随机森林研究— [**点击此处。**](https://github.com/RutvijBhutaiya/The-Famous-Titanic-Study)

[](https://github.com/RutvijBhutaiya/The-Famous-Titanic-Study) [## rutvijbhutaya/著名的泰坦尼克号研究

### 展开步骤步骤 2:下载 Titanic 数据集步骤 3:设定研究目标步骤 4:多变量分析…

github.com](https://github.com/RutvijBhutaiya/The-Famous-Titanic-Study)