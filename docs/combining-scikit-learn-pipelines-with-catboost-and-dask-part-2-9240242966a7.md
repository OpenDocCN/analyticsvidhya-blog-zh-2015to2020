# 将 Scikit-Learn 管道与 CatBoost 和 Dask 结合起来(第 2 部分)

> 原文：<https://medium.com/analytics-vidhya/combining-scikit-learn-pipelines-with-catboost-and-dask-part-2-9240242966a7?source=collection_archive---------4----------------------->

![](img/2d63c13d7b63daeb8dc608c3e36ba137.png)

约翰·贝克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

当我们在将 Sklearn 管道与 Catboost 和 Dask 相结合的迷雾中继续我们孤独的旅程时，当我们将 Catboost 模型集成到我上一篇文章的结果项目中时，我们开始看到管道尽头的亮光(绝对是双关语)。

在我之前的[帖子](/@kigelman.nir/combining-scikit-learn-pipelines-with-catboost-and-dask-part-1-c08d67b44815)中，我谈到了如何使用 sklearn-pandas 包和 sklearn 管道来维护整个管道中的数据帧结构。

[](/@kigelman.nir/combining-scikit-learn-pipelines-with-catboost-and-dask-part-1-c08d67b44815) [## 将 Scikit-Learn 管道与 CatBoost 和 Dask 结合起来(第 1 部分)

### 我写这篇文章是为了帮助那些像我一样迷失在管道、CatBoost 模型和 Dask 领域的人…

medium.com](/@kigelman.nir/combining-scikit-learn-pipelines-with-catboost-and-dask-part-1-c08d67b44815) 

在这篇文章中，我将谈论如何集成 CatBoost 模型，以及为什么这是一个问题。

# 问题是

让我们从为什么我要写一篇关于使用 Catboost 模型的帖子开始…这肯定是一个简单的任务，就像我们从 Sklearn 了解并喜欢的其他模型一样。的确如此。

![](img/16f56de638d7e7136518704b61072128.png)

当我们想要将 Catboost 模型合并到一个可以在运行时改变数据结构的动态管道中时，问题就出现了。
为什么这是一个问题？因为我们使用 Catboost 来支持分类特征，并且如果我们希望 Catboost 发挥其魔力，我们需要让他知道我们的数据中有哪些分类特征，但是我们开始使用的特征不是 Catboost 模型将满足的特征。
Catboost 模型将满足我们在管道中的后续步骤将确定的一些随机特性集。

为了克服这个问题，我们需要以某种方式跟踪我们的分类特征，因此 Catboost 模型将知道剩余的分类特征是什么。

# 解决方案

我发现最有效和最简单的解决方案是在每个分类特征后添加一个后缀，用一个常量字符串表示这是一个分类特征。
然后，我们可以编写自己的简单定制 Catboost，在运行时检索包含分类后缀的列名。

# 代码示例

继续我之前的[帖子](/@kigelman.nir/combining-scikit-learn-pipelines-with-catboost-and-dask-part-1-c08d67b44815)中的例子。
首先，我们需要定义分类后缀字符串。

然后，我们需要将这个后缀添加到我们的分类特征中。

## 自定义 Catboost 分类器

现在，我们创建自己的自定义 Catboost 分类器，并在运行时使用参数“cat_features”调用 fit 方法。

## 特征选择

为了在我们的管道中加入一个特性选择步骤，我们需要确保特性选择转换方法将返回一个数据帧，这样带有分类后缀的列名将保持不变。

## 把所有的放在一起

最终，我们的管道应该是这样的:

**我的** [**GitHub**](https://github.com/kinir/catboost-with-pipelines/blob/master/sklearn-pandas-catboost.ipynb) **上有更丰富的例子包括网格搜索过程的笔记本。**

[](https://github.com/kinir/catboost-with-pipelines) [## kinir/catboost-带管道

### 此时您不能执行该操作。您已使用另一个标签页或窗口登录。您已在另一个选项卡中注销，或者…

github.com](https://github.com/kinir/catboost-with-pipelines) 

# 结论

在本文的第 2 部分中，我解释了如何将 CatBoost 模型与 Sklearn 管道集成，并且仍然能够使用其 fit 方法的“cat_features”参数。在本文的第 3 部分，我将解释如何将 Dask 集成到我们现有的代码中。

未完待续…