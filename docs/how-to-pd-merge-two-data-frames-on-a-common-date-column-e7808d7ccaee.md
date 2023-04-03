# 如何 pd.merge()一个公共日期列上的两个数据帧。

> 原文：<https://medium.com/analytics-vidhya/how-to-pd-merge-two-data-frames-on-a-common-date-column-e7808d7ccaee?source=collection_archive---------0----------------------->

![](img/cf76cde16bb53c230a5e62636e3c7f0d.png)

马库斯·斯皮斯克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

使用新冠肺炎病例和死亡数据进行数据分析的主要挑战之一是将这两个数据框架在日期上合并在一起。由于每个 CSV 数据文件中有重复的日期条目，合并两个数据帧并不简单。以下是我如何应对这一挑战的解释。

# **1。数据下载- Cases.csv**

我已经从 coronovirus.data.gov.uk 下载并保存了 cases.csv 文件到我的 Kaggle 网页中。这是对这些数据的一个快速浏览。

![](img/8341dff5fd5b46ccf0e443285c00bfd0.png)

**清理和数据准备—** 转换日期数据类型，检查数据帧长度，检查空值。

![](img/4f9bb954f471a1153aa74bc8f8db9446.png)

**检查任何重复-** 269 个唯一的日期表明这些日期是重复的，即重复几次，因此我们得到长度 975。

![](img/36bce68a611781b7ddeda10285fdf6cd.png)

**groupby()、sum()** —使用 dates 列上的 groupby()和 newCases 列上的 sum()返回长度为 269 的 series 对象。这将把所有重复的日期组合成一组，并将它们各自的案例相加。

![](img/f0bed9697f4718b15735b318ac826e6c.png)

**系列到数据框** — Groupby()函数返回一个系列对象。所以下一步是将这个 series 对象转换成一个 269 行的新数据帧。

![](img/7c0cce732be13f4028cfa326b57a024a.png)![](img/d6fbd2d0618e2b74bcb238d51139dc84.png)![](img/2e321571ed05cbe259702d34ea1ff2df.png)

**Cases_df 数据帧现已准备就绪，可与另一个数据帧合并。**

# 2 **。数据下载— Deaths.csv**

我已经从 coronovirus.data.gov.uk 下载并保存了 deaths.csv 文件到我的 Kaggle 网页中。这是对这些数据的一个快速浏览。

![](img/871c197a552d7436dac84f7e6b71410f.png)

**清理和数据准备—** 转换日期数据类型，检查数据帧长度，检查空值。

![](img/46975e6b5061dfc8cbb74fe689e711bd.png)

**检查任何重复-** 226 个唯一的日期表明这些日期是重复的，即重复了几次，因此我们得到长度 862。

![](img/889eefa9b1c9ca5d20ecb8fb251c53d5.png)

**groupby()，sum()** —使用 dates 列上的 groupby()和 newDeaths 列上的 sum()返回长度为 226 的 series 对象。这将把所有重复的日期归为一组，并把他们各自的死亡时间加起来。

![](img/460299099ee27298adbedc6134a78a0c.png)![](img/938afd395f8109802260984ea3414735.png)

**系列到数据帧** — Groupby()函数返回一个系列对象。所以下一步是将这个 series 对象转换成一个 226 行的新数据帧。

![](img/2c9ddb100635c91d94e015eabb95b1f1.png)![](img/e84fc86b023ed8cb4ba3f64c0f048eb7.png)![](img/2b3957b1759e4238e0d313dae9fa7ff6.png)

**死亡数据框现已准备好与病例数据框合并。**

# **合并病例和死亡数据框**

![](img/87a192e6436f9b28e72325c7c8a7225e.png)![](img/3599ef292f0962dee7b65702e0be45ec.png)

# **下面是我们使用 pd.merge()时实际发生的情况**

为了解释这个过程，我创建了两个微型数据框架，分别命名为病例和死亡。

![](img/cb912fdebd4432487fe6421c25b6b1d8.png)

病例数据框中有 3 个日期条目(十月、九月、六月)。

![](img/141b30b6c361e695734c6ea826ef018e.png)

上面的死亡数据框中有 3 个日期条目。(10 月、9 月、8 月)

# 内部联接时合并

![](img/b73596de8a3f8438bb555ad4431da88e.png)

观察:内联接输出有十月，九月。这两个日期都出现在病例和死亡数据框中。

# 外部连接时合并

![](img/87d1c96afed808ad272f45a5d6380019.png)

观察:外连接输出有十月、九月、六月和八月。所有数据均来自病例和死亡数据框。

# 在左连接时合并

![](img/a2b3352ce696ba2e0c5decf269c4b24e.png)

观察:左连接输出有十月，九月，六月。所有病例的日期都存在，并且只取死亡病例的匹配日期。

# 右连接时合并

![](img/9669747cf30c7ee22932f9e6dd3a69e7.png)

观察:右连接输出有十月、九月、八月。所有死亡日期都存在，并且只取病例中匹配的日期。

总结本文，关键要点是，如果您必须在一个公共列上合并两个数据框，并且公共列有重复项，一种方法是将一个数据框的公共列中的所有值组合在一起。这将删除所有的重复。新的分组对象可以转换回新的数据框。这将使合并更容易，并且在执行合并时具有准确的值。希望这篇文章是有帮助的。谢谢你。