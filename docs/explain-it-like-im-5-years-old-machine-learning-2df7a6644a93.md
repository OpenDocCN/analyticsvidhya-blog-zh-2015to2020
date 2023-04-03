# 就像我 5 岁的孩子一样解释——机器学习

> 原文：<https://medium.com/analytics-vidhya/explain-it-like-im-5-years-old-machine-learning-2df7a6644a93?source=collection_archive---------16----------------------->

![](img/674bf4076909c710f4ce6bce10e911ef.png)

[https://www . disruptive static . com/WP-content/uploads/2018/05/machine-learning-ecommerce-blog-1 . jpg](https://www.disruptivestatic.com/wp-content/uploads/2018/05/machine-learning-ecommerce-blog-1.jpg)

机器学习。这是一个大多数人都听说过的时髦词，但有充分的理由。你可能在生活中的某个时候遇到过。你喜欢网飞给你的账户提供的推荐吗？对着亚马逊的 Alexa“说话”怎么样？或者你可能已经注意到，当你上传一张照片到网站时，脸书的照片标签建议奇怪地[准确](https://www.forbes.com/sites/amitchowdhry/2014/03/18/facebooks-deepface-software-can-match-faces-with-97-25-accuracy/#610046ff54fc)。这当然不是魔术，但由于机器学习，它可能会出现这种情况。

## **它到底是什么？**

简单地说，机器学习是允许计算机学习的技术。这个想法是，通过给计算机足够的数据，并指示它以特定的方式处理数据，它可以被“训练”来对你试图预测的任何事情做出越来越准确的预测。这种类型的“训练”的理想之一是，不需要为计算机编写一个庞大的明确的指令列表来进行预测。有了足够的数据，随着我们给它的数据量和质量的提高，它有望做出更好的猜测。

举个简单的例子，假设你有 1000 张不同汽车的图片。如果你只看一张照片，很可能你会很快认出照片中的汽车图案。这是为什么呢？你如何将它与其他物体区分开来，比如橘子或火车？你真的觉得有必要查看其他 999 张图片才能知道你看到的图片是一辆汽车吗？很可能不是，很可能是因为你在生活中已经看到了足够多的汽车，对它们的样子有了一个大致的概念:4 个轮子和某种类型的框架。但是电脑呢？就此而言，它怎么能做出类似或任何其他类型的预测呢？计算机需要某种类型的指导来处理这些数据，以提取答案。它需要人类给它提供足够的数据，这样它也能像我们一样有经验地识别模式。

## 监督学习

![](img/cce5e31fc2e031938cbcdfa9df2f4a75.png)

[https://www.cc.gatech.edu/social-machines/projects.html](https://www.cc.gatech.edu/social-machines/projects.html)

在电脑上储存大量数据是一回事，但能够从中获得某种洞察力或预测则完全是另一回事。原始数据和生成的预测之间看似很大的差距是什么？算法。让我们来看看一个非常简单但仍然有用的算法，您可能在某个时候见过:

![](img/a08392041a69c8a28691011a383c6f40.png)

简单线性回归方程

让我们假设上述等式的 **m** *(俗称*为*【斜率】)* 和 **b** *(y 轴截距)*的值不会改变。对于每一对 **x** 和 **y** 值， **x** 是输入到我们的等式中的值， **y** 是相应的输出值，当绘制出来时，将创建一条简单的线。

这可能很容易概念化，但尽管如此，它可能会为我们所拥有的数据提供一些见解。也许我们拥有的数据包括每个学生为一组先前的考试学习的小时数、 **x** 值、**和他们各自在该考试中的分数、 **y** 值**。**然后[我们可以计算出 **m** 和 **b**](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/regression-analysis/find-a-linear-regression-equation/) 来给出我们最终的方程，这个方程可以根据学生学习的小时数来预测他们在考试中的表现。**

向上倾斜的线可能表明学生学习的时间越多，学生可能得到的分数越高。但是一个变量能说明学生成绩背后的全部故事吗？还有其他因素可能在起作用吗？

学生每周可能工作的小时数或学生每晚的平均睡眠时间是多少？这如何改变我们对学生成绩的预测？如果是，改变多少？

在我们开始用计算机可以用来给出某种类型的预测输出的数据实现任何算法之前，我们应该非常仔细地考虑我们对数据和算法的选择。我们希望通过选择正确的数据类型并利用有意义的算法，确保计算机基于对我们想要预测的内容有意义的数据进行学习。

在这些情况下，我们*监督*计算机的学习。如果没有适当的指导，计算机可以很容易地为 m 和 T42 选择数字，这些数字可能完全偏离我们简单模型的基础，从而最终做出非常不准确的预测。

## 算法

虽然我们的上述模型非常简单，但它可以扩展，因此如果我们认为有必要，它也可以考虑其他变量。也就是说，我们可以轻松地添加另一组 **x** 及其对应的斜率值，并将其相乘，如下所示:

![](img/4f625dba2a66546d9329453f583056ea.png)

[https://i0 . WP . com/broker stir . com/WP-content/uploads/2018/04/multiple _ linear 2 . png](https://i0.wp.com/brokerstir.com/wp-content/uploads/2018/04/multiple_linear2.png)

但是如果我们试图建模的东西不完全遵循线性模式呢？我们的预测还有更好的选择吗？绝对的。例如，我们可以使用下图所示的[逻辑回归模型](https://en.wikipedia.org/wiki/Logistic_regression),来帮助我们确定一个事件是否会发生的概率，例如失败或通过、活着或死了、赢了或输了。

![](img/34e5b76dbf270e3aec149a29cab72f48.png)

[https://UC-r . github . io/public/images/analytics/logistic _ regression/plot 2-1 . png](https://uc-r.github.io/public/images/analytics/logistic_regression/plot2-1.png)

也许现存的最有趣的算法之一是*决策树*。决策树是由一系列的类和节点组成的，你猜对了，它们形成了一个树状结构。每个节点都由一种分类规则组成，这种分类规则将根据这种“测试”的结果确定数据下一步的去向。最终，数据将沿着树向下结束它的旅程，并到达某种类型的分类标签。《纽约时报》的下图是其预测能力的一个很好的例子:

![](img/d9fda6db348670240776338d7bf6e547.png)

[https://archive . nytimes . com/www . nytimes . com/image pages/2008/04/16/us/2008 04 16 _ OBAMA _ graphic . html？scp = 5&sq = Decision % 2520 奥巴马% 2520 克林顿& st=cse](https://archive.nytimes.com/www.nytimes.com/imagepages/2008/04/16/us/20080416_OBAMA_GRAPHIC.html?scp=5&sq=Decision%2520Obama%2520clinton&st=cse)

在这里，我们可以看到 2008 年总统选举中巴拉克·奥巴马和希拉里·克林顿之间选票分配的更详细的分析。该树能够提供更深入的统计洞察，准确了解人口中的哪些部分投票给了哪个候选人；否则可能不会立即显现的洞察力。

## 无监督学习

之前我提到过这样一个想法，即通过确保对输入数据进行深思熟虑的选择，来指导计算机的学习。同样，这可以帮助确保计算机为我们给它的算法选择最佳值，并有希望输出相当准确的预测。毕竟，如果它需要运行成千上万次迭代，以便为我们选择的算法获得那些最优值，我们希望确保它对数据的“训练”是有意义的，对吗？

但是，假设我们没有一个很好的线索来告诉计算机使用什么样的数据作为一个合理的预测器来训练，或者知道一个“好”的预测可能是什么样的。在我们之前的学生考试成绩的例子中，我们可能想要在模型中使用的数据类型似乎相当简单。然而，根据我们想要预测的情况，这种情况可能不会发生。幸运的是，有一些技术可以帮助我们找到一些隐藏的模式或结构，这些模式或结构可能不会立即显现出来，更好的说法是*无监督学习。*

在无监督学习中，没有很好的分类数据，我们可以输入到计算机可以训练的特定算法中，以便它可以输出似乎合理的预测。在这种情况下，计算机基本上被留下来“弄清楚”在数据中可能存在什么样的模式或者它拥有什么类型的结构。克服这一点最常见的方法之一叫做*集群*。

![](img/24a9259f078d96b3efe6bc0389719e81.png)

[https://www . imper va . com/blog/WP-content/uploads/sites/9/2017/07/k-means-clustering-on-spherical-data-1 v2 . png](https://www.imperva.com/blog/wp-content/uploads/sites/9/2017/07/k-means-clustering-on-spherical-data-1v2.png)

聚类是一种为正在分析的数据创建某种类型的分类的方法。我们的目标是将可能使用的全部数据点分成不同的组。每个组中的数据将彼此共享某些相似性，然后可以给每个组分配一个标签。由此，人们可以开始从新划分的数据中得出某些推论。

聚类的一些常见应用包括能够出于营销目的表征和发现不同的客户群，甚至检测客户数据中可能指示欺诈活动的模式。

## 机器学习及其他

自 1959 年计算机科学家亚瑟·塞缪尔在 IBM 工作时开始使用这个术语以来，机器学习无疑已经走过了漫长的道路。该领域在 20 世纪 80 年代和 90 年代开始发展，当时计算能力总体上得到提高，并且在过去 10 年中确实蓬勃发展，因为现在存在大量数据，这些数据可以补充用于已经创建的各种算法。

机器学习的应用远不止网飞向你推荐的电影。2019 年 7 月在网上发表的一篇为[国立卫生研究院](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6642356/)撰写的文章指出，机器学习“正在对癌症诊断产生巨大影响”。 [EliteDataScience](https://elitedatascience.com/machine-learning-impact) 的另一项研究报告称，一项涉及谷歌 55 辆自动驾驶汽车的研究已经累计行驶了 130 多万英里，“在安全性方面超过了人类驾驶的汽车”。learn.g2.com[引用了](https://learn.g2.com/future-of-machine-learning)[大云](http://www.bigcloud.io/)创始人&首席执行官 Matt Reaney 的话说，将量子计算融入机器学习可能会影响数百万人的生活，特别是在医疗保健领域，因为我们当前面临的复杂问题可能会在很短的时间内得到解决。

谁也不知道机器学习下一步会把我们带到哪里，但有一点是肯定的，当我们到达那里时，我会很兴奋。

来源:

[](https://www.geeksforgeeks.org/clustering-in-machine-learning/) [## 机器学习中的聚类

### 它基本上是一种无监督学习方法。无监督学习方法是一种方法，其中我们画…

www.geeksforgeeks.org](https://www.geeksforgeeks.org/clustering-in-machine-learning/) [](https://www.tutorialspoint.com/data_mining/dm_cluster_analysis.htm) [## 数据挖掘-聚类分析

### 集群是属于同一类的一组对象。换句话说，相似的物体被组合在一个…

www.tutorialspoint.com](https://www.tutorialspoint.com/data_mining/dm_cluster_analysis.htm) [](https://sonix.ai/articles/difference-between-artificial-intelligence-machine-learning-and-natural-language-processing) [## 人工智能，机器学习，自然语言有什么区别…

### 理解所有围绕人工智能(AI)的首字母缩略词几乎比理解潜在的…

sonix.ai](https://sonix.ai/articles/difference-between-artificial-intelligence-machine-learning-and-natural-language-processing) [](https://www.technologyreview.com/artificial-intelligence/machine-learning/) [## 机器学习

### 关于机器学习算法及其应用的新闻和报道，来自麻省理工技术评论。

www.technologyreview.com](https://www.technologyreview.com/artificial-intelligence/machine-learning/) [](/free-code-camp/the-hitchhikers-guide-to-machine-learning-algorithms-in-python-bfad66adb378) [## Python 中机器学习的搭便车指南

### 包含实现代码、教学视频等

medium.com](/free-code-camp/the-hitchhikers-guide-to-machine-learning-algorithms-in-python-bfad66adb378)