# 详细的 z 分数

> 原文：<https://medium.com/analytics-vidhya/z-score-in-detail-9dd0f0afa142?source=collection_archive---------5----------------------->

让我们了解什么是 Z 分数，它的应用，以及如何使用它来比较不同尺度的多个观察值(尺度几乎是指一系列数据，但并不完全相同。如果不同尺度下的观测值相同，那么当我们把它们放到同一尺度下时，它们的值肯定会不同。)

![](img/b1b3ae42da5cc51ff46103d1236a7bc6.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Carlos Muza](https://unsplash.com/@kmuza?utm_source=medium&utm_medium=referral) 拍摄的照片

我们将对 Z 分数进行深入讨论。但在此之前，我们需要了解什么是正态分布和标准正态分布。

# 什么是发行版？

好吧，对于刚接触统计学的人，我想让你们知道什么是分布。用简单的方式想一想。

今天是你的生日！你在班上**分发糖果。你会怎么做？让我告诉你我会如何分配它们。我会给我的两个最好的朋友三颗糖，给我的两到十个朋友，剩下的人只能得到一颗糖。**

**嗯，这就是分配！**

# **统计中的分布是显示变量的可能值及其出现频率的函数。**

**所有这些分布都有“概率分布函数”或“概率质量函数”，这取决于分布，但现在，这不是我们的茶。我们在这里学习分布的基础知识。**

**太好了！你已经学习了什么是分布。所以现在让我们试着去理解正态分布。**

# **什么是正态分布？**

**正态分布是关于平均值对称的分布(嗯，平均值就是所有观测值的平均值)。正态分布中的大多数观察值都集中在平均值附近。**

**我希望你对分布和正态分布有一个简单的概念。让我们看看标准的正态分布，这是非常简单的。它只是正态分布的一个特例。**

# **什么是标准正态分布？**

**标准正态分布是均值和标准差分别为 0 和 1 的正态分布。**

> **z 得分只能对服从正态分布的观察值进行计算。**

# **什么是 Z 分数？**

**Z 得分是一种数值度量，它描述了一个值与一组值的平均值之间的关系。**

> **z 得分以平均值的标准偏差来衡量。**

****例子****

**假设有三个学生，他们的英语考试成绩分别是 12，16 和 23。平均值是 17。**

**除了平均值，我在 Z 得分的定义中使用了另一个术语，即标准差。**

# **什么是标准差？**

**标准差是一个量，表示一个组的成员与该组的平均值相差多少。在上面的例子中，平均值是 17，观察值是 12，16 和 23。标准差怎么算？我会为此编写一个简单的 python 代码！**

```
import math
marks = [12,16,23]
mean = sum(marks)/len(marks)
print('Mean:',mean)# Here comes the standard deviation
# First let us calculate the individual deviations of observations from their meanmarks_dev = [abs((x-mean)**2) for x in marks]
st_dev = math.sqrt(sum(marks_dev)/len(marks_dev))
print('Standard deviation:',round(st_dev,2))
```

# **Z 分怎么算？**

> **z =(数据点—平均值)/标准偏差**

## **我们在这里做什么？**

**我们只是把平均值缩小到零。因此，让我们计算上面示例中标记的 Z 分数，这样我就可以非常详细地解释什么是 Z 分数。**

```
# Calculating Z scores
Z_scores = [round((x-mean)/st_dev,2) for x in marks]
Z_scores
```

**我们得到了 12，16 和 23 的 Z 值分别为-1.10，-0.22 和 1.32。**

# **这个 Z 分数到底意味着什么？**

**让我们考虑 Z 值为 23。它是 1.32，这意味着 23 是它的平均值的标准差的 1.32 倍！也就是说，由于平均值是 17，标准差是 4.55，23–17 是 6，等于 1.32 *标准差。这才是重点！**

# **我为什么要计算 Z 分数？我是说，我们可以直接比较分数，对吧？**

**不要！你不能这样。让我给你举一个例子，两个学生试图进入一所 M.tech 大学，但通过不同的考试。**

**Suresh 出现在 GATE 考试中，并希望使用他的分数进行录取。然而，Archana 没有参加 GATE，但她在 PGECET 考试中表现出色(注意，GATE 和 PGECET 是两个不同的研究生入学考试)。**

**Suresh 的分数是 73，而当年的平均门分数是 87，标准差是 23。**

**Archana 的分数是 345，其中平均 PGECET 分数是 374，标准偏差是 115。**

**你能从分数上告诉我谁做得比较好吗？我没有那种超能力。那么我们的下一个问题来了。**

# **Z score 是如何用来比较不同尺度上的多个分数的？**

**我很想知道谁能上大学。你也是吗？到时候就知道了！**

```
Suresh_score = 73
Archana_score = 345
avg_gate_score = 87
avg_pgecet_score = 374
gate_stdev = 23
pgecet_stdev = 115Suresh_z_score = (Suresh_score-avg_gate_score)/gate_stdev
print("Suresh's Z score is:",Suresh_z_score)Archana_z_score = (Archana_score-avg_pgecet_score)/pgecet_stdev
print("Archana's Z score is:",Archana_z_score)if Suresh_z_score > Archana_z_score:
    print("Suresh made it to the university!")
elif Suresh_z_score == Archana_z_score:
    print("That's some great news! Both of them made it!")
else:
    print("Archana made it to the university!")Suresh's Z score is: -0.6086956521739131
Archana's Z score is: -0.25217391304347825
Archana made it to the university!
```

**那么，为什么阿卡纳能去大学？能不能比较一下阿卡纳和苏雷什的 Z 分？哪个更高？**

**Archana 的分数约为-0.2522，高于 Suresh 的分数，约为-0.6087。**

# **更高的 Z 分是好还是坏？**

**我会说这完全取决于问题陈述和情况。在上面的例子中，我们在寻找一个比其他人做得更好的候选人。所以，我们选择了 Z 值较高的那个。**

**如果我们以沿海城市遭受海啸袭击的次数为例，并与其他沿海城市进行比较，Z 值最高的城市将是受灾最严重的城市。**

**好吧！我们已经知道，Z 分数有助于比较不在同一尺度上的数据。这个的其他用途是什么？**

# **离群点检测**

**是的，Z 分数也可以用于异常值检测。如果我忘记了上面提到的，如果 Z 值小于-3 或大于 3，那么这个观察值可能被认为是异常值。**

> **什么是离群值？**
> 
> **离群值是与数据中的其他值有显著差异的值。**

**好吧！那么我们来看一个问题。**

**以下是对大海得拉巴地区房屋的 15 个观察样本。**

## **房屋面积以平方码为单位。**

```
Observations = [200,234,523,1255,623,324,65,123,192,4332,433,235,543,720,239]
```

**现在让我们使用 Z 分数来检测数据中的异常值。**

```
from statistics import mean
from statistics import stdev

Observations = [200,234,523,1255,623,324,65,123,192,4332,433,235,543,720,239]avg = mean(Observations)
st_dev = stdev(Observations)# Now that we've calculated mean and standard deviation, it's time for outlier detection.outliers = list()
for i in Observations:
    z = (i-avg)/st_dev
    if z <= -3 or z>=3:
        outliers.append(i)
outliers
```

**所以我们可以看到，4332 是数据中唯一的异常值，这是相当明显的，一个人的房子建在这么大的面积上是非常罕见的。**

# **参考**

**[1]:索尔·麦克劳德:Z-Score:定义、计算与解释:[https://www . simply psychology . org/Z-Score . html #:~:text = The % 20 value % 20 of % 20 The % 20z，standard % 20 deviation % 20 above % 20 The % 20 mean。](https://www.simplypsychology.org/z-score.html#:~:text=The%20value%20of%20the%20z,standard%20deviation%20above%20the%20mean.)**

**【2】:[**斯蒂芬妮格伦**](https://www.statisticshowto.com/contact/) 。来自[**StatisticsHowTo.com**](https://www.statisticshowto.com/)的“Z 分数:定义、公式和计算”:对我们其余人的基本统计！[https://www . statistics show to . com/probability-and-statistics/z-score/](https://www.statisticshowto.com/probability-and-statistics/z-score/)**

**[3]:可汗学院:[https://www . khanacademy . org/math/AP-statistics/density-curves-normal-distribution-AP/measuring-position/v/z-score-introduction](https://www.khanacademy.org/math/ap-statistics/density-curves-normal-distribution-ap/measuring-position/v/z-score-introduction)**