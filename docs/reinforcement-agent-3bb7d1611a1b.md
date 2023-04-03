# 增强剂

> 原文：<https://medium.com/analytics-vidhya/reinforcement-agent-3bb7d1611a1b?source=collection_archive---------23----------------------->

在我以前的文章中，我已经给出了关于强化学习及其一些算法的简要细节。

今天，我要告诉你一些简单的解释，关于强化代理如何在一个从未互动过的环境中工作。

![](img/33b9c702fc22f00412e12e6021d67aa4.png)

[张家瑜](https://unsplash.com/@danielkcheung?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral) 上拍照

强化智能体当在一个****环境*** 内行动时，它学习到*的最优行为来实现价值函数以获得奖励。代理的行为将由策略控制。首先，一个带有输入值的代理链接。然后它改变数据集的表示。这将有助于代理识别硬输入表示，并且这将有助于比初始输入值更进一步。**

**每个代理都有 ***工作记忆*** 。其中一些主要可以分为两个部分。这些记忆与记忆细胞一致。第一个木桶是 ***知觉缓冲*** 。它获得原始向量输入。 ***参加缓冲器*** 是第二个，开始时为空。最后，第三个缓冲器保存初始输入的副本，称为 ***忽略缓冲器*** 。**

**![](img/b5acf5ff2b895df088357641be10b80a.png)**

**[布雷特·乔丹](https://unsplash.com/@brett_jordan?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/robot?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片**

**代理以两种方式处理这些购物篮。为了获得输入的三个副本，代理复制了 bucket 2 中的初始输入。否则，桶被代理删除，因为它将能够得到非常清楚的两个桶。我上一篇文章讨论的 ***奖励函数*** ，不是类标签独立的。要学习价值函数，可以使用奖励函数。当考虑分类新的(看不见的)数据集时，在开始时，代理能够与它交互。然后它决定奖励的摄入是积极的还是消极的。通过与数据集交互，代理将了解模式的负值或正值。对于分类新的看不见的模式，可以使用这个值。**