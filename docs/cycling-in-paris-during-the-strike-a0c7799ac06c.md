# 罢工期间在巴黎骑自行车🚲

> 原文：<https://medium.com/analytics-vidhya/cycling-in-paris-during-the-strike-a0c7799ac06c?source=collection_archive---------19----------------------->

自 2019 年 12 月 5 日以来，巴黎一直在举行罢工。截至 1 月 12 日，罢工给巴黎人的日常生活带来重大改变已经超过一个月了。虽然法国有 30 多个工会参与反对养老金制度改革，但在这篇文章中，我探讨了罢工在巴黎人日常交通习惯中的作用。更具体地说，我专注于巴黎罢工期间的骑行趋势。

# 数据收集

当我有写这篇文章的想法时，我在谷歌上搜索了巴黎的自行车数据。然后我很高兴地发现，自 2018 年以来，关于骑行的数据集在[巴黎数据平台](https://parisdata.opendatasoft.com/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id&disjunctive.name&sort=date&dataChart=eyJxdWVyaWVzIjpbeyJjaGFydHMiOlt7InR5cGUiOiJhcmVhc3BsaW5lIiwiZnVuYyI6IkFWRyIsInlBeGlzIjoiY291bnRzIiwic2NpZW50aWZpY0Rpc3BsYXkiOnRydWUsImNvbG9yIjoiI0ZGQ0QwMCJ9XSwieEF4aXMiOiJkYXRlIiwibWF4cG9pbnRzIjoiIiwidGltZXNjYWxlIjoiaG91ciIsInNvcnQiOiIiLCJjb25maWciOnsiZGF0YXNldCI6ImNvbXB0YWdlLXZlbG8tZG9ubmVlcy1jb21wdGV1cnMiLCJvcHRpb25zIjp7ImRpc2p1bmN0aXZlLmlkIjp0cnVlLCJkaXNqdW5jdGl2ZS5uYW1lIjp0cnVlLCJzb3J0IjoiZGF0ZSIsInEudGltZXJhbmdlLmRhdGUiOiJkYXRlOlsyMDE5LTEyLTExVDIzOjAwOjAwWiBUTyAyMDE5LTEyLTE5VDIyOjU5OjU5Wl0ifX19XSwiZGlzcGxheUxlZ2VuZCI6dHJ1ZSwiYWxpZ25Nb250aCI6dHJ1ZSwidGltZXNjYWxlIjoiIn0%3D)公开可用。数据仅包含 60 个不同地点每小时骑自行车的人数。不幸的是，不可能从数据中获得骑自行车者的唯一数量，因为在一条骑自行车者路线上的多个计数器会导致统计中的重复。

尽管骑自行车者的数量不是唯一的，但与之前间隔的数据比较将有助于发现变化。在比较数据时，我关注两个维度:

*   月环比(妈妈):4 周前的同一天，骑自行车的人数是多少？(如**2019–12–05**vs**2019–11–07**)，
*   同比:一年前的同一天，骑自行车的人数是多少。(例如**2019–12–05**vs**2018–12–05**)。

数据集在开放数据库许可证( **ODbL** )下获得许可。在我的 [GitHub](https://github.com/bhakyuz/experimentation/tree/master/paris-strike) 资源库中可以公开获得用于准备本文的数据处理和可视化的 r 代码。

# 罢工期间的日常骑行趋势

罢工开始的第一天，骑自行车的人数比一年前增加了两倍，比上个月增加了 105%。

总体而言，截至 **2020 年 1 月 12 日，**骑自行车的人数同比增长 333%,这取决于一天中的星期:

*   工作日: **+362%** 同比
*   周末: **+248%** 同比增长

我想巴黎人现在更喜欢骑自行车上班。他们周末宁愿呆在家里，如果他们无论如何都需要出去，他们更喜欢骑自行车或 T7 Velib T8 T9。

在下面的图表中，我们看到了每天骑自行车的总人数，以及前一年和前一个月的数字。正如所强调的，在周末，自行车的使用相对少于平日。

![](img/0b31019157877dddfad84b7df04dbdcc.png)

2019/20 巴黎养老金改革罢工期间的日常骑行趋势

# 每小时循环趋势

我想看的第二件事是罢工期间骑自行车的小时分布。回想我之前的假设，我期望在工作日的高峰时段看到自行车使用的显著增加。

如下图所示，工作日上午 8 点到 10 点和下午 5 点到 7 点骑自行车的人要多得多。相反，在周末，我们看到更多的自行车使用在早上开始缓慢平衡，逐渐增加，直到晚上，然后恢复正常。

![](img/3b824959980ad68f5bcbe60a0c2df952.png)

每小时骑行趋势:工作日与周末

当我深入研究细节时，我发现与去年相比，在工作日高峰时段，自行车的使用率增加了 310%到 340%。令人惊讶的是，在工作日期间，骑自行车的最大跳跃是在早上 6 点到 7 点之间，几乎是 430%。在周末期间，最大的增加发生在下午 4 点到 8 点，大约为 425%。

当我们查看每天的数据时，我们看到在工作日期间骑行分布相当稳定，例如，周一至周五没有太大差异。我也注意到周六和周日有类似的趋势。然而，只有星期五和星期六深夜例外。正如我们在每日图表中所看到的，在周五和周六的深夜，骑自行车的人数仍然较高，与去年相比增加了 400%,而在其他时间，骑自行车的人数增加了近 200%。

![](img/a47a66169c7605b234664fddefa8416c.png)

每个工作日的每小时骑行趋势

# 自行车交通的关键点

最终，我渴望在巴黎地图上强调自行车交通，以检测骑自行车者的密度。考虑到骑自行车的人在巴黎并不是均匀分布的，这是相当棘手的。这就是为什么如果附近有两个或更多的柜台，下面的可视化是基于骑车人的最大数量。我们在地图上容易注意到的另一个问题是，巴黎还没有到处安装计数器，这意味着我们不知道自行车的使用情况，尤其是 17、18 和 20 区。

![](img/aa6e74010d8cd2137e0475c571c84fa2.png)

巴黎的自行车密度和自行车计数器

正如地图上用红色突出显示的，我们很容易注意到 ***夏特莱–莱斯哈勒*** 区域是骑自行车者最繁忙的地方。这个地区也在我的日常自行车路线上，我可以说，看到它是最活跃的自行车中心并不奇怪。

骑自行车人数最多的柜台是第 4 区的 ***Rue Rivoli*** 和第 19 区的***48 quai de la marne***。在罢工期间，他们平均每天都在超过 6000 名骑自行车者的路线上。另一方面，第 14 区的***Porte D ' ORLéans***是最空闲的点，每天只有大约 230 名骑自行车的人。

# 结论

随着巴黎养老金改革罢工的进行，公共交通正一天天恢复正常。我仔细观察了罢工对巴黎自行车运动的影响。我们看到在工作日的高峰时间和周末晚上的深夜骑自行车的人数明显增加。骑自行车的关键点是 ***沙泰勒–Les Halles***，因为它位于市中心。