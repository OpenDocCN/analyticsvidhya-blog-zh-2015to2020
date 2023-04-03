# 不平衡数据情况下的三个步骤，并仔细查看拆分器类

> 原文：<https://medium.com/analytics-vidhya/three-steps-in-case-of-imbalanced-data-and-close-look-at-the-splitter-classes-8b73628a25e6?source=collection_archive---------4----------------------->

![](img/574f37f6408e242b9a3719f598a833fb.png)

当我们有一个**不平衡的**(比如标签中的%90 个 A、%10 个 B)数据集时，我们应该小心使用**“训练/测试分割”**步骤(以及交叉验证)

有 3 件事要做:

*   以这样的方式分割测试集，使您的**测试集**在整个测试集中具有相同的比例。否则，由纯粹的…