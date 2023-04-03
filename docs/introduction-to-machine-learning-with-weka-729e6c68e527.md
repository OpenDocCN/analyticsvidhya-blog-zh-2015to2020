# 用 WEKA 介绍机器学习！

> 原文：<https://medium.com/analytics-vidhya/introduction-to-machine-learning-with-weka-729e6c68e527?source=collection_archive---------22----------------------->

## 这是机器学习世界的一个良好开端

![](img/3824d3e5882718e2f1b06e5f11b09386.png)

Jukan Tateisi 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

eka 是一套全面的 Java 类库。Weka 包实现了许多最先进的机器学习和数据挖掘算法。

在本文中，我将讨论最重要的 weka 模块，如下所示:

# 探险家

Explorer 是一个使用 Weka 探索数据的环境。Explorer 是 Weka 的主要图形用户界面。Weka Explorer 包括主要的 Weka 包和一个可视化工具。Weka 的主要特性包括过滤器、分类器、聚类、关联和属性选择。

# 实验者

Weka Experimenter 是一个用于执行实验和在学习方案之间进行统计测试的环境。

# 知识流

Weka KnowledgeFlow 是一个支持与 Explorer 相同功能的环境，但包含一个拖放界面。

# 工作台

Weka Workbench 是一个一体化的应用程序，它在用户可选的透视图中结合了其他应用程序。

# 简单 CLI

Weka 团队建议使用 CLI 来深入了解 Weka 的用法。大多数关键功能都可以从 GUI 界面获得，但是 CLI 的一个优点是它需要的内存少得多。如果您发现自己遇到了 ***内存不足*** 错误，CLI 界面是一个可能的解决方案。

Weka 模块中有一些冗余。您将重点关注以下三个 Weka 模块，因为它们足以创建您的 Java 应用程序所需的模型，这三个模块是:

*   Weka 知识流
*   Weka Explorer
*   Weka 简单 CLI

我已经排除了实验者和工作台。您将使用知识流模块来比较不同算法的多个 ROC 曲线。ROC 曲线是说明二元分类器系统在其辨别阈值变化时的诊断能力的图表。

实验者也可以这样做，但是即使 Weka 没有最好的图形界面，我还是更喜欢知识流模块的图形方式。如果您正在为 Weka 模块寻找一个定制的透视图，您可以使用工作台模块。

Weka 团队确实以随每个版本分发的 PDF 文件的形式提供了官方文档，并且怀卡托大学为想要学习 Weka 的开发人员提供了许多视频和支持资源。Weka 手册有 340 多页，如果你想认真了解 Weka，这是必不可少的读物。

作为结论，我将展示来自 Weka 创建者的官方 Weka 文档:

*   Weka manual:当前版本的 Weka manual(如***【WekaManual-3–8–2.pdf】****【WekaManual-3–9–2.pdf】)*始终包含在发行版中。对于任何特定的 Weka 版本，手动文件名为***WekaManual.pdf***。
*   Weka book:Weka 团队已经出版了一本书， ***数据挖掘——实用的机器学习工具和技术*** ，作者是 Witten、Frank 和 Hall。这本书是一本非常好的 ML 参考书。虽然它没有详细介绍 Weka，但它确实涵盖了数据、算法和一般 ML 理论的许多方面。
*   YouTube:Weka YouTube 频道， ***WekaMOOC*** ，包含许多有用的 Weka 操作视频。

起源于[blog.selcote.com](http://blog.selcote.com/2020/02/14/introduction-to-machine-learning-with-weka/)