# 使用数据 101:数据清理

> 原文：<https://medium.com/analytics-vidhya/playing-with-data-101-data-cleaning-6ee079fd16e0?source=collection_archive---------45----------------------->

在 Jupyter 笔记本的 Python3 中使用 Pandas 进行基础数据清理。

![](img/5685101b05d2f22cf1b56fafda4263a1.png)

从互联网上

*熊猫*来源于词语“**潘** el **da** ta”。Pandas 软件包为我们提供了数据帧操作。通常，Pandas 将是我们创建和操作表格数据的首选包，这些数据表示为 DataFrames，每一列存储为一个 Series 对象。

在这篇博客中，我们将使用 2017 年和 2018 年的 SAT 和 ACT 分数来演示一些基本的数据清洗以及我们可以从我们的数据中发现什么。

***一些背景:*** *SAT 和 ACT 成绩被广泛用于教育机构的录取决定和颁发择优奖学金。尽管大多数大学并不偏爱其中一种考试，但 SAT 和 ACT 的参与率在美国各州都有很大差异。*

## 数据读取

我们得到了四个数据集:2017 年 SAT 成绩，2017 年 ACT 成绩，2018 年 SAT 成绩和 2018 年 ACT 成绩，分别来自 SAT 数据来源[这里](https://blog.collegevine.com/here-are-the-average-sat-scores-by-state/)，ACT 数据来源[这里](https://blog.prepscholar.com/act-scores-by-state-averages-highs-and-lows)(网站已经更新到 2019 年)。它们不是大数据集，所以我们可以手动读取它们并检查错误。在这 4 个数据集中，我们总共有大约 1000 个项目。除了“州”，其他的看起来都像数字。`States`列包含美国的 50 个州和首都 DC。然后，我们对数据帧进行代码检查，以查看行数和列数，以及是否有丢失的值。

## 数据清理

即使数据框架不包含缺失值，`sat_2017`也没有国家参与率或分数行。所以我们删除这一行。此外，我们发现有 3 个错误:

1.  在`sat_2017`，马里兰州`Math`的平均分数是 52，这是不可能的低，因为最低的数学分数是 200。
2.  在`act_2017`中，马里兰州的`Science`分数为 2.3，这也是不合理的，因为`Composite`分数为 23.6。
3.  `act_2017`中`Composite`的类型是 object，因为在第 51 行中值是一个字符串`20.2x`。我们修复了错误，然后将所有数字列更改为`int`或`float`。

2017 年 SAT 成绩

ACT 分数 2017

## 合并数据帧

在合并数据框之前，我们需要用全部小写字母重命名列，并且不使用空格。此外，列名应该是唯一的和信息丰富的，因为我们在 4 个数据帧中有几个相同的列名。
当我们合并数据帧时，我们发现`act_2018`中有`District of columbia`而不是`District of Columbia`中有很多`NaN`。我们也修好了。
我们分两步完成:

1.  将 2017 年的数据帧合并为 2017 年的综合 SAT 和 ACT 成绩，将 2018 年的数据帧合并为 2018 年的综合 SAT 和 ACT 成绩
2.  合并 2017 年和 2018 年的数据帧，作为 2017 年和 2018 年的最终得分

然后，我们导出如下所示的数据文件:

## 外部研究

艾奥瓦州、堪萨斯州和新墨西哥州是 2017 年考试参与率最低的三个州。艾奥瓦州、堪萨斯州和阿拉斯加州是 2018 年考试参与率最低的三个州。有趣的是，ACT 的总部设在爱荷华州的爱荷华市。另一方面，一些州的 SAT 或 ACT 参与率高达 100%。这是否意味着一些州出于多种原因将 SAT 或 ACT 作为所有合格学生的要求，但对于许多学生和家长来说，SAT 或 ACT 分数对于申请奖学金至关重要。对于申请奖学金的学生来说，很多奖学金在“更聪明”的州更有竞争力。对于那些想把自己与州内同龄人进行比较的学生来说，州平均成绩也非常有用。[来源](https://blog.prepscholar.com/average-sat-and-act-scores-by-stated-adjusted-for-participation-rate)

为什么那些州的 SAT 和 ACT 参与率低？这是否意味着他们的教育系统有问题？
研究显示，越来越多的大学不把 SAT/ACT 作为录取的强制因素。两所常青藤联盟学校已经决定，他们的许多研究生项目不需要考试分数，这是教育机构越来越不希望将高风险考试作为录取和拒绝学生的一个因素的新证据。
原因是许多研究表明，SAT 和 ACT 成绩与家庭收入、母亲的教育水平和种族密切相关。从 2018 年 9 月到 2019 年 9 月，近 50 所授予学士学位的认证学院和大学宣布，他们将放弃 SAT 或 ACT 分数的入学要求。这使得获得认证的学校数量达到 1050 所，约占总数的 40%。[来源](https://www.washingtonpost.com/education/2019/10/18/record-number-colleges-drop-satact-admissions-requirement-amid-growing-disenchantment-with-standardized-tests/)

虽然教育机构正在采用更多的方法来吸引更广泛的申请人，但 CollegeBoard 和 ACT Inc .正在相互竞争，以提高参与率。