# 需求预测和 ML

> 原文：<https://medium.com/analytics-vidhya/demand-forecasting-and-ml-8ee9484f6fd2?source=collection_archive---------12----------------------->

对任何供应链主管来说，需求预测接近正确都是梦想成真。

很明显，它驱动着你所有的挑战，从供应短缺、库存过剩到客户满意度。

在本文中，我们讨论如何以及在哪里可以在需求预测中利用人工智能。

![](img/d04b48de4895f89515c387af6374db07.png)

埃文·丹尼斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

# 从前…

根据 1971 年的《哈佛商业评论》,需求预测基本上有三种基本类型。

> 定性技术(专家小组、市场研究等)
> 
> 时间序列分析和预测(移动平均、指数平滑等)
> 
> 因果模型。(回归、计量经济学模型等)

[](https://hbr.org/1971/07/how-to-choose-the-right-forecasting-technique) [## 如何选择正确的预测技术

### 实际上，今天的高管们在做每一个决定时，都会考虑某种预测。对需求的合理预测…

hbr.org](https://hbr.org/1971/07/how-to-choose-the-right-forecasting-technique) 

# 在最近的时间里…

快进到更近的时间，

久负盛名的 2018 年 M4 预测竞赛就 ML 对需求预测的影响给出了一些令人惊讶的见解。

> 在 17 种最精确的方法中，有 12 种是大多数统计方法的“组合”。
> 
> 最令人惊讶的是一种“混合”方法，它利用了统计学和 ML 特性。
> 
> 第二种最准确的方法是七种统计方法和一种最大似然法的组合，平均的权重由最大似然算法计算，该算法被训练为通过维持测试来最小化预测误差
> 
> 六种纯 ML 方法表现不佳，没有一种比组合基准更准确，只有一种比 nave 2 更准确。

*来源:*[*https//www . science direct . com/science/article/ABS/pii/s 016920701830078*](https://www.sciencedirect.com/science/article/abs/pii/S0169207018300785)

结果清楚地表明

> 统计模型的组合比单一模型效果更好
> 
> 统计学和机器学习的混合体有更好的机会。

# **我如何决定:**

1.  *在因果关系(变量对产出的影响)、计量经济(经济因素对产出的影响)、定性分析(专家意见)方面，您已经尽了最大努力使用统计建模方法，但仍有多个模型给出了可比较的结果。*

机器学习层可以给你额外的优势来决定每个模型的权重

![](img/866184543661226529f7ed2a1e7577e9.png)

照片由[凯罗尔·斯特法斯基](https://unsplash.com/@karolhere?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

![](img/2f62310d926e6d50bd617df45a636592.png)

[沃洛德梅尔·赫里先科](https://unsplash.com/@lunarts?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

2.*你的数据变量或特征相当不直观，而且你有大量的数据。ML 技术可以帮助你降维。*

3.*如果您的预测项目是没有或只有少量历史数据的新产品，ML 可能不是您的首选技术。*

![](img/4efc25e44954957a0fc4fe56af32b294.png)

[产品学校](https://unsplash.com/@productschool?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

4.*注意统计模型和机器学习模型之间的范式差异。统计学是这两种技术的基础。
在统计建模中，你用尽所有的数据，然后想出一个“适合”这些数据的模型。*

![](img/dfa3d8e07d574c468885530a33f13b89.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上 [Enayet Raheem](https://unsplash.com/@raheemsphoto?utm_source=medium&utm_medium=referral) 拍摄的照片

在 ML 中，你把数据分成训练、验证和测试数据集。

![](img/338eb342b0977b049bdc67ab878746ee.png)

照片由 [Clarisse Croset](https://unsplash.com/@herfrenchness?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

5.统计模型给你的不仅仅是预测。它能给你信心区间。示例:今年的需求预测有 95%的可能性介于 2 亿美元至 2.5 亿美元之间。

标准形式的 ML 技术只关注预测。例如:今年的预测是 2.12 亿美元，MAPE 为 8%(平均绝对百分比误差)，没有置信区间信息。

> **因此，当谈到需求预测时，统计和机器学习方法不应被视为竞争技术，而是协作和补充技术。**

希望这篇文章能让你对 ML 对需求预测的影响有一个直观的认识。

请关注这篇关于如何让人工智能为供应链服务的文章。在那之前，

预测愉快。

再见了。

![](img/b379ce9b9e99658f44e3bd3bb16fcd5c.png)

照片由 [Rico Zamudio](https://unsplash.com/@iamricozamudio?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄