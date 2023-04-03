# 电子商务客户推荐系统第一部分(基于内容的推荐)

> 原文：<https://medium.com/analytics-vidhya/content-based-collaborative-and-their-hybrid-approaches-for-recommendation-making-to-e-commence-e2015830a04f?source=collection_archive---------1----------------------->

本博客涵盖了基于内容的推荐方法，并附有代码说明。

第一部分:[基于内容的](/@rmwkwok/content-based-collaborative-and-their-hybrid-approaches-for-recommendation-making-to-e-commence-e2015830a04f)

第二部分:[协同](/@rmwkwok/content-based-collaborative-and-hybrid-recommender-for-e-commence-customers-part-ii-e0b9e5a8b843)

第三部分:[混合动力](/@rmwkwok/content-based-collaborative-and-hybrid-recommender-for-e-commence-customers-part-iii-2e225d2e6d21)

![](img/08d86fe7111f70777676cb94ae9c1cb1.png)

# 介绍

推荐系统在行业中被广泛用于了解客户行为和推荐产品。在本系列的这篇文章和接下来的两篇文章中，我们将探索不同类型的推荐系统——基于内容的、协作过滤的和混合的。让我们首先了解这三者之间的基本区别。

## 基于内容、协作和混合

假设你有一些商品要推荐，如果你对两者都非常了解，或者换句话说，如果你有足够的顾客和商品的数据，那么**基于内容的方法**会给你非常安全的推荐。然而，他们可能会绑定到你已经知道你的客户喜欢什么。

另一方面，如果你不太了解双方，那么**协同过滤方法**会通过训练一个客户嵌入和一个物品嵌入给你一些运气。在这种情况下，它们的特征是由算法决定的，而不是由你的知识决定的。

与基于内容的方法不同，基于内容的方法推荐与客户过去喜欢的项目相似的项目，协作过滤方法推荐与您选择相似的客户喜欢的项目。这种方法非常有效，但是它需要客户和产品之间的交互历史，如果没有交互历史，对于特定的客户或特定的产品，就会出现所谓的“冷启动”问题，也就是说，在不了解某个人的情况下，你无法有针对性地预测这个人。在这种情况下，最后，一个基于知识的方法将做工作，找出一些新来的人，你发出问卷，收集他们的个人偏好，并征得他们的同意。没有魔法，而是你知道多少。

因此，根据对客户的了解程度，您可以设计一个策略来充分利用您的数据。当然，这些方法在实现上有所不同，人们甚至可以结合这些方法的发现来构建一个**混合推荐器**。虽然人们可以想出尽可能多的实现方法，但我将展示我的一种方法，并简要讨论每种方法(基于内容的、协作的和混合的)。在阐述我的代码之前，我将给出一段我要用来训练的数据。

# 数据

本博客使用的数据集可从 [Kaggle](https://www.kaggle.com/retailrocket/ecommerce-dataset) 下载。它包含客户查看/添加到购物车/购买商品的事件，以及每个商品的类别层次结构。大约 235000 个项目被分成大约 1300 个类别，这些类别又被分成大约 400 个父类别。在这项工作中，我将向客户提供类别建议，而不是项目建议，因为类别的大小可以在有限的资源和数据中处理，这足以展示技术。

# 基于内容的方法

这需要深入了解推荐的商品，以及顾客对这些商品的偏好或评分。需要预先定义和测量对象的特征。在一篇文章的例子中，它们可以是作者的名字、主题、语言等等。这些可以组合起来，并给出每个对象的良好的、可区分的描述。如果一个客户一直给某一类型的文章很高的评价，那么基于内容的将能够提供非常相似的建议。

我们将深入研究客户类别数据，你准备好了吗？

![](img/a3bd0420e3bc2b3413933e769e785439.png)

(上)顾客对 4 个类别的评价——鞋子/蔬菜/巧克力/桌子。类别与其所属的父类别之间的(下级)关系。

首先，我们可以将客户的评分和对象的描述列表成两个矩阵(表格)。上面的部分表格说明了这一点。

机器学习处理数字，所以上表会转化成下面这个矩阵。

![](img/3af3bd44dbf09677da0159c97bfa39dc.png)

将星星转换成数字。客户在不同的行，类别在不同的列。

虽然下面的表已经是数字的，但原则上，它已经可以用于计算。然而，因为在我的例子中，一个类别只对应于一个或几个父类别，导致了一个稀疏矩阵，父类别不能很好地概括类别的描述。因此，我将把下面的表格转换成下面的密集嵌入矩阵。

![](img/93b6c921d265fa70a4c4e082a6e9a1f7.png)

将对象矩阵转换成嵌入。不同行中的类别和不同列中的父类别。

原始的人类可读对象矩阵和这个嵌入的矩阵之间的区别在于，(1)列的数量大大减少，这意味着您需要更少的内存来使用它，以及(2)嵌入的列的含义变得抽象，因此更难解释。差数 2 当然不是故意的，但却是差数 1 的直接后果。为了理解这一点，我们应该快速地看一下嵌入是如何从原始矩阵产生的。

*注意:如果您想直接使用原始对象矩阵而不做任何嵌入，您可以跳过下一节*

## 嵌入的数学

![](img/cf194f63fc7c63927c1b6ac7fb49e65d.png)

将一个巨大的对象矩阵分割成两个更小的嵌入体的过程。

嵌入的产生是一个数学过程，将原始的巨大对象矩阵(具有 C×P 参数)分裂成两个具有偏差项(具有 C×E+E×P+C×1+1×E 参数)的更细的嵌入。右手侧的参数是可变的，其服从于最小化左右之间的差异。换句话说，我们正在寻找右手边的最佳参数集，以便它能最好地类似于原始矩阵。由于它们被如此安排以使彼此相似，因此*类别嵌入被认为是在降低维度中捕捉类别的本质，或者我们正在将类别的描述从其与数百个父类别的关系压缩到几个嵌入特征*。这解释了最后一段中的差异数字 1，以及嵌入的特征是人类可理解的特征的压缩的事实，如果没有非常专门的分析，它们不能用文字直接解释，因此差异数字 2 随之而来。我将在[关于协同过滤的文章](/@rmwkwok/content-based-collaborative-and-hybrid-recommender-for-e-commence-customers-part-ii-e0b9e5a8b843)中讨论实现这个优化过程的代码。

## 回到基于内容

有了下面两个矩阵或数据，我们就可以开始提出建议了。我会一步一步地解释这个算法。

![](img/e13e691a85348d7cf62c3e1b881c88c2.png)

用于推荐的数据。这只是相应矩阵的局部视图，预计会非常大。

第一步:对于每个客户，将他/她对一个类别的评分乘以该类别的嵌入功能，这样对一个类别的高评分将被放大更多。

![](img/7ed50e76e26fc1fff523235d5efbae0d.png)

通过评级放大类别的特征。每位顾客都有一张较低的桌子。

```
*# Code for step 1*
import numpy as np*# vis_cat_mat: matrix for visitors' rating to categories, 2d array
# cat_par_mat: normalized embedding from category-parent relations, 2d array*step1 = np.stack([ vis[None,:].transpose() * cat_par_mat\
                   for vis in vis_cat_mat])*# result in a 3D array. 0th axis: customer, 1st: category, 2nd: embedded features*
```

步骤 2:对于每个客户，对不同类别的嵌入特征值求和(或求平均值)。这将为您提供客户总结的首选嵌入式功能。

![](img/24223ff231dfae53a8be984d02b8f022.png)

对每个客户的类别求和。每位顾客都有一张较低的桌子。

```
step2 = step1.sum(axis=1)*# result in a 2D array. 0th axis: customer, 1st: embedded features*
```

步骤 3:对于每个客户，通过将每个值除以平方值的和来归一化汇总的特征值。这是下一步的准备工作。

![](img/d6a69724320dd4750e3c8bb0a9435e07.png)

将每个值除以所有值的平方和。每位顾客都会有一张这样的桌子

```
step3 = step2/np.square(step2).sum(axis=1, keepdims=True)*# result in a 2D array. 0th axis: customer, 1st: embedded features*
```

第四步:对于每个客户，*将概括的嵌入特征点*到每个类别的嵌入特征上。因为两者都已经预先标准化，所以这个点操作为我们提供了客户偏好和类别之间的*余弦相似性的度量。余弦相似度的范围从-1 到 1，1 表示最相似，而-1 表示最不相似。因此，如果一个类别的值为 1，意味着它与客户的偏好非常相似，那么推荐它很可能会受到客户的青睐。*

![](img/c5066ef1f252616811368196206a5af0.png)

将偏好和类别点在一起。如果它们相似，它将产生一个接近 1 的值。每位顾客都会有一张这样的桌子。

```
step4 = np.stack([(v[None,:]*cat_par_mat).sum(axis=1) for v in step3])*# result in a 2D array. 0th axis: customer, 1st: similarity*
```

第五步:对产生的相似性进行排序，找到最佳的 *k* 个类别，以产生 *k* 个推荐。

# 摘要

本文讨论了基于内容的推荐算法。它需要一个对象矩阵，对所有对象进行良好的描述，一旦客户对某些对象的偏好或评级可用，则该算法可以“总结”客户的偏好，以找出其他类似的对象进行推荐。

[协作](/@rmwkwok/content-based-collaborative-and-hybrid-recommender-for-e-commence-customers-part-ii-e0b9e5a8b843)和[混合](/@rmwkwok/content-based-collaborative-and-hybrid-recommender-for-e-commence-customers-part-ii-e0b9e5a8b843)方法将在单独的文章中讨论。为了显示这些方法在我的数据(非常有限)上的差异，我绘制了访问了推荐类别*而没有真正通知他们推荐的客户的百分比*。x 轴表示推荐的数量。y 轴是客户的百分比。虽然推荐的数量越少，百分比越低，这听起来很糟糕，但是，根据商业案例，推荐者不仅要做出精确的预测，还要通过提出合理的建议来拓宽客户的视野，这些建议可能是他们听不到的。这一情节将在[混合篇](/@rmwkwok/content-based-collaborative-and-hybrid-recommender-for-e-commence-customers-part-iii-2e225d2e6d21)中讨论。

![](img/52668974540cfe71f9f0032ae4a3ddd8.png)