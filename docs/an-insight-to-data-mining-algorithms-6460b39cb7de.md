# 数据挖掘算法透视

> 原文：<https://medium.com/analytics-vidhya/an-insight-to-data-mining-algorithms-6460b39cb7de?source=collection_archive---------21----------------------->

![](img/13e3f67d38a8178dc20c9e6ed098d2b6.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Rohan Makhecha](https://unsplash.com/@rohanmakhecha?utm_source=medium&utm_medium=referral) 拍摄的照片

O 最有启发性的经验之一是简单的想法往往非常有效，我强烈建议在分析实际数据集时采用简单第一的方法。

数据集可以展示许多不同种类的简单结构。

在一个数据集中，可能有一个属性完成所有工作，而其他属性可能不相关或多余。

# **推断基本规则**

无论如何，先尝试最简单的事情总是一个好计划。

想法是这样的:

> 我们制定测试单个属性的规则，并相应地进行分支。

每个分支对应于不同的属性值。

给每个分支的最佳分类是显而易见的:使用在训练数据中最常出现的类。

# **缺少值和数字属性**

尽管这是一种非常初级的学习方法，但 1R 确实可以处理缺失值和数字属性。

它以简单而有效的方式处理这些问题。

*缺失的*被视为另一个属性值。

因此，例如，如果天气数据包含 *outlook* 属性的缺失值，则在 *outlook* 上形成的规则集将指定四个可能的类值，分别用于*晴天、阴天、*和*雨天*以及第四个用于*缺失*。

# **统计建模**

1R 方法使用单一属性作为决策的基础，并选择最有效的一个。

另一个简单的技巧是使用所有属性，并允许它们对决策做出同等重要且相互独立的贡献。

# **构造决策树**

决策树算法基于对分类问题的分而治之的方法。

他们自上而下地工作，在每一个阶段寻找一个能最好地区分类别的属性；然后递归地处理分裂产生的子问题。

# 结论

这种策略会生成一个决策树，如果有必要，可以将其转换为一组分类规则——尽管如果要生成有效的规则，这种转换并不简单。

## 感谢阅读！

如果你喜欢我的工作，喜欢支持我…

**。s** 在我的 [**Youtube 频道**](https://www.youtube.com/channel/UCU_LhClyNOtEQw7R0q9ovoQ) 上订阅，我分享了很多精彩的内容，比如

这可是在视频里

。在 facebook 上关注我[这里](https://www.facebook.com/zelakioui)

。在 twitter 上关注我[这里](https://twitter.com/zelakioui)