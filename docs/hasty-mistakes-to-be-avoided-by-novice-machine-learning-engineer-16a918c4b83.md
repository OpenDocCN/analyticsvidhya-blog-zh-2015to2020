# 新手机器学习工程师应避免的草率错误

> 原文：<https://medium.com/analytics-vidhya/hasty-mistakes-to-be-avoided-by-novice-machine-learning-engineer-16a918c4b83?source=collection_archive---------13----------------------->

在任何 ML 项目中，在进入算法之前执行一系列活动是非常重要的。理解商业目的；特征分析、选择、缩放和工程；异常值处理；探索性数据分析仅举几例。

![](img/6e53801a60f908c5b9f86853402704c3.png)

泰勒·尼克斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

在这篇文章中，我将详细阐述*商业目的*和*特性选择*的重要性。我们可以在不同的文章中讨论其他话题。

# 理解业务目的

如果不了解为什么需要 ML 模型，那么构建任何 ML 模型都是没有意义的。数据科学家必须非常清楚地了解他们正在解决的业务问题。这将有助于设计更好的模型。

假设我们正在处理一个不平衡的数据集，例如信用卡欺诈、检测癌症等，其中阳性案例非常少，业务目的是检测这些阳性案例。在这样的要求下，我们正在寻找一种[敏感型](https://en.wikipedia.org/wiki/Precision_and_recall)。但是如果一个工程师建立了一个精确的模型，那将是一场灾难。例如，让我们取一个信用卡数据集，其中 99%的样本是负面的(真实的)，1%是正面的(欺诈的)。建立一个简单的模型，把所有事情都归类为负面的，会给出 99%的准确率，这对于不理解商业目的的人来说是难以置信的。然而，这根本没有解决业务目的。在这种情况下，具有较高召回值的模型服务最好。

在另一个例子中，我和我的同事在一个拥有超过 500 万条飞行记录的数据集上工作，其商业目的是预测起飞延误(回归问题)。经过很多努力，我可以得到 0.26 的 R2 分数( [Ref: Kaggle Kernel](https://www.kaggle.com/karthikcs1/analysis-of-flight-delay) )，但是我的同事骄傲地告诉我，他的 R2 分数是 0.92。我大吃一惊，开始回顾我错过了什么。我发现我的同事使用的一个特征是*到达延迟*，它与我们的目标变量*出发延迟有大约 92%的相关性。*太好了。但是等等，商业目的是预测出发延迟。在预测出发延误的同时用到达延误作为特征有意义吗！！绝对不行。到达延误意味着航班已经起飞并到达目的地。预测过去有什么意义。在从事一个 ML 项目时，理解目的和拥有一些商业头脑是最重要的。

# 特征选择

实时数据集通常具有大量的特征。我见过大多数 ML 工程师新手直接跳入考虑‘所有’特征的算法，并检查准确性。我甚至看到 ID 号(没有任何相关性)被用作特征，从商业角度来看，这真的很傻！我们知道不是所有的特征对目标变量都同等重要。仔细选择构建更好模型所需的特征非常重要。

为了使这项工作更容易，在库中封装了许多统计技术，如[*【sk learn】*](https://scikit-learn.org/stable/modules/feature_selection.html)*。*在高层次上有 2 组技术— [单变量](https://en.wikipedia.org/wiki/Univariate_analysis)和[多变量](https://en.wikipedia.org/wiki/Multivariate_statistics)。让我们来讨论其中的一些:

[**低方差特征**](https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance) **:** 该方法寻找样本空间中方差为零或很小的特征，并将其从模型中移除。这种分析可以发生在一个变量上，而不依赖于其他变量。

[**选择 k 个最佳**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) **:** 该方法通过用 Y(目标变量)拟合 X(自变量)并基于提供的度量计算得分来选择前“k”个最佳特征。评分可以基于 chi2、ANOVA F 值、百分位数等。

[**Select Percentile**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile)**:**这与 *Select k best* 非常相似，只是不提供特征数量，我们可以提供 percentile。高于百分位数的最高得分特征将被选择用于模型构建

[**递归特征消除**](https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination) **:** 在该方法中，使用外部估计器通过逐个消除特征来递归拟合模型。完成后，它将提供与模型相关的功能集。

[**简单相关分析**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html) **:** 在这个简单的方法中，我们可以检查每个特征与目标变量的相关程度。基于此，我们可以选择高度相关的。我们还应该看看 X 变量之间是否有很高的相关性。如果是，需要选择其中的一个，而不是全部。

我们可以使用一种以上的方法来筛选特征，然后在它们之间执行一个集合交集来最终确定特征。请参考[这个内核](https://www.kaggle.com/karthikcs1/online-news-popularity-lightgbm-gridsearch)，其中遵循了类似的方法

# 如此匆忙的原因

以下可能是一个 ML 工程师新手在重要和先决条件活动之前如此匆忙地投入算法的原因。

1.  缺乏数据科学方面的适当培训/教育
2.  大多数 ML 工程师都是传统的程序员。他们习惯于在没有适当设计的情况下编码和检查结果。
3.  ML 算法让超级变得非常容易，工程师们无法抗拒地投入其中:-)。
4.  没有统计学背景，也不热爱数据分析
5.  没有商业头脑，没有商业分析师背景。

# 摘要

我们已经讨论了一个业余 ML 工程师的行为方面，他们匆忙地进入算法并执行试错调整来改善结果。相反，他们应该关注许多重要的先决条件步骤。本文讨论了理解业务目的和**特性选择**的重要性**。**