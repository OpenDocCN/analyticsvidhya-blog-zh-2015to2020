# 实施 ML/DL/优化的人力资源数据

> 原文：<https://medium.com/analytics-vidhya/hr-data-for-practicing-ml-dl-5fba688bebc1?source=collection_archive---------13----------------------->

![](img/e46618524a805ecf94b4677f1a61e4d4.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Toa Heftiba](https://unsplash.com/@heftiba?utm_source=medium&utm_medium=referral) 拍摄的照片

# 介绍

人物分析是一个新兴的领域，不管是好是坏，它还没有像它的另一个对手——零售业——那样受到关注，后者处理人类行为和进入商业的资金。但是，事实证明，对于一个人来说，如何把钱花出去(给员工)和如何保管这些钱(毕竟，你确实需要齿轮来弥补车轮对你有利的转动)也同样重要。

令人遗憾的是，人们很难通过事实和数字找到理解前者的方法。People Analytics 除了晦涩难懂之外，还非常羞于提供任何数据样本，即使是那些可以公开获得的样本。考虑到隐私的氛围，其中一些是正确的。毕竟，我也讨厌营销人员拿到我的号码，试图进行徒劳的销售，尽管我从来没有在陌生电话的基础上进行过购买(唯一的例外是信用卡，他们给我打了很多次电话，并宣布我已经对他们的提议说了一次“是”。非常肯定我没有。

> 数据！数据！数据！没有粘土我做不出砖——夏洛克·福尔摩斯

尽管如此，引用著名侦探的话。没有数据，你不可能成为数据科学家。我说的不是你在公司内部得到的那些。但是，那些可以公开获得的。鉴于即使是最少的一个数据集也包含个人信息，而人们不希望暴露这些信息，因此 HR 数据集永远不会可用。最重要的是，分析行为模式。

在 kaggle 中很少有与 HR【https://www.kaggle.com/datasets?search=HR】相关的好的公共 HR 数据集。多亏了一小群人的出色工作。大部分都和流失预测有关。

从总体上看人力资源数据集，我意识到我们缺乏一个全面的数据集，可以探索超出一些人口统计或绩效相关的因素。因此，我整理了一个模拟的人力资源数据集，它将涵盖数据科学家希望获得的一些非常可爱的功能。同样，为什么这很重要。默认情况下，语法数据生成是创建真实世界复杂性的非常紧密的表示所必需的

*   缺少正确且没有缺失值的干净数据
*   需要担心所有共享内容的合法性
*   通过感知的复杂性，可以在不止一种情况下创建模型，特别是针对任何想要鼓励的行为模式进行调整

最后一点是模拟数据集能够真正带来增值的真正原因。你可以探索那些原本无法计划的情况。例如，自动化导致工作岗位流失——自工业革命以来一直如此，但无法跟踪，因此，我们无法计划将员工转换到新的工作岗位。就此而言，这与就业机会的创造或损失有点关系，挑战了专精于某个角色的传统。因为角色可能变得多余，但是人们可以适应。(显然，这也是一个新事物——结账 [AQ](https://www.fastcompany.com/40522394/screw-emotional-intelligence-heres-the-real-key-to-the-future-of-work) 。).这个想法是为了保护人们而不是工作。

# 模拟数据集

事不宜迟，这里是一个假想组织的样本数据。公司的要旨。这是一个相对年轻的公司，年轻员工充满活力。它在第三产业的核心业务——可能是一家科技分析初创公司。该公司引以为豪的是，它采用了一种开放的思维模式，既重视员工的技术能力(即智商)，也重视他们的精力、同情心、努力程度(即情商)。他们展示了作为模范雇主的一些理想行为，在这种情况下，任何行业的人都可以创造价值(阅读:扁平的层级结构)，渴望了解他们如何能够更好地帮助他们的员工拥有出色的职业生涯(阅读:创新驱动，强调保留)，同时坦率地承认有时需要走强硬路线(阅读:位于自动化和市场中断预期的行业)

以下是在上述公司简介的基础上增加的功能:

列-ID、姓名、性别、婚姻状况、出生日期、电子邮件、Linkedin、博客、文章、任期、Cos_Worked、学术 _Level、体育 _Level、Domains_Worked、Manager_OrNot、Span、Hierachy _ Level、Prvs_WorkExp、Rated、Soft_skills、Pay_Level、Pay_Level_Peers、Avg_Promotion_Rate、Avg_Hike_Rate、Bonus Level、冗余 _Possible、add_random_features

离职者—要求离开、自愿离开、原因
入职者—角色类型、招聘方式
调动—部门内调动、跨专业调动

[](https://www.kaggle.com/stitch/hr-data-for-mldloptimization-practice) [## 人力资源数据-用于 ML/DL/优化实践

### 下载数千个项目的开放数据集+在一个平台上共享项目。探索热门话题，如政府…

www.kaggle.com](https://www.kaggle.com/stitch/hr-data-for-mldloptimization-practice) 

以下是关于如何创建数据集的内核链接:

[](https://www.kaggle.com/stitch/hr-dataset-creation) [## 人力资源数据集创建

### 下载数千个项目的开放数据集+在一个平台上共享项目。探索热门话题，如政府…

www.kaggle.com](https://www.kaggle.com/stitch/hr-dataset-creation) 

内核也是如何为建模实践创建语法数据集的一个很好的例子。关于数据集创建的独家文章，您可以点击此[链接](https://dziganto.github.io/data%20science/eda/machine%20learning/python/simulated%20data/Simulated-Datasets-for-Faster-ML-Understanding/)。

这是一项正在进行的工作。我计划增加多年劳动力流动。除此之外，在接下来的几周内，在流失预测、劳动力优化、薪酬预测、雇佣质量等方面创建专门的内核。也许，它甚至会成为启动人员分析的一个很好的参考系列。再见。