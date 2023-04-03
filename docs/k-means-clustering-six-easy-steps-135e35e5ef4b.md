# 赶时间！k 均值聚类:六个简单的步骤

> 原文：<https://medium.com/analytics-vidhya/k-means-clustering-six-easy-steps-135e35e5ef4b?source=collection_archive---------16----------------------->

## 幕后发生了什么——k-均值聚类——以及如何应用该算法。

![](img/2b08dfbaf5e2bfd6dc9a59a37bf5b639.png)

布鲁克·拉克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

这个博客是我对无监督学习理解的一部分。

我将简单介绍一下什么是无监督学习，然后给你六个简单的步骤来理解 k-means 聚类。

在无监督学习中，人类的参与最少(如果我们将人类视为机器)，数据是无标签的。

无监督学习主要用于电子商务中的产品交叉销售。

因此，如果你有一个数据集，其结果显示了一群不同的人(多样性定义为年龄、种族、教育、收入、地理区域等。)你知道他们每个月在产品上花了多少钱，你会**根据他们每个月花了多少钱来分组，而不考虑其他因素。**

接下来，你可以向他们出售更多的产品，这些产品是同一消费群体中的其他人已经购买过的。

在聚类分析中，数据根据相似程度而不是根据类别或标记进行分组。

例如，在一家公司，你可以根据员工的表现(高、中、低)对他们进行分组，而不需要根据他们的部门、性别或年龄对他们进行标记(独立)。

![](img/7cb51b9eb4a24af358871333e9b70d47.png)

照片由 [Mpho Mojapelo](https://unsplash.com/@mpho_mojapelo?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

**步骤 1** :在对任何数据应用 k-means 算法之前，检查数据集中是否实际存在聚类。应用霍普金斯统计学。r 库- > factorextra 和函数->get _ clut _ trend()。如果值接近 0，则数据中存在聚类。如果结果接近 1，则意味着没有聚类。

**第二步:**将数据分成簇。借助 nbclust()选择 k

**第三步:**选择 k 簇中间的 k 点。

**步骤 4:** 计算欧几里德距离，并保持将 k 簇中的新点向 k 点的中心移动。

**步骤 5:** k 点继续向该聚类中的点的基于聚类的新点均值的中心移动(我们不会改变该点，除非有一个聚类具有比先前计算的聚类更好的距离/均值)。

**步骤 6:** 进行评估—执行轮廓指数评估聚类—轮廓值在-1 到 1 之间变化(接近 1 是好的。)- >来自 factorextra 库的函数剪影()。

我参考了 Connor Brereton 的《从第一原理学习机器》。

关于实际应用以及如何执行客户参与案例研究的 k-means 聚类— [点击此处](https://github.com/RutvijBhutaiya/Thailand-Customer-Engagement-Facebook)

[](https://github.com/RutvijBhutaiya/Thailand-Customer-Engagement-Facebook) [## rutvijbhutaya/泰国-客户参与-脸书

### 10 个泰国时尚和化妆品零售商的脸书页面。不同性质的帖子(视频、照片、状态…

github.com](https://github.com/RutvijBhutaiya/Thailand-Customer-Engagement-Facebook)