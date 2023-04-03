# 用代码进行数据分析

> 原文：<https://medium.com/analytics-vidhya/thanos-or-spectre-who-can-beat-captain-marvel-data-analysis-b9bfff5bc4b2?source=collection_archive---------17----------------------->

## 漫威和特区漫画人物的数据分析

![](img/2642ae8080b7ebb881738712e2d8fbb0.png)

萨诺斯

几个月过去了，灭霸已经死了。我们知道即使没有无限之石他也有多强大。但让我们假设他还活着，他正在寻找无限宝石，现在我们有了一个来自 DC 的超级恶汉，也是“幽灵”。

![](img/1ed6f13eac2e8e0d40c8ae3efd52e863.png)

幽灵

**是的，所以漫威 VS DC。**

让我再加一个转折，一个漫威英雄也是一个叛徒，那就是漫威队长，她现在也在 infinity stones 后面。我们将看到谁有更多的权力，谁是无限宝石的合适人选。

![](img/278b99f9fa30bc744d61ae89ede28c32.png)

这叫做阴谋。

数据是从 kaggle 导入的，我会把链接放在最后，为了数据可视化我用过 matplotlib 库，sns 库和 plotly 库。

加载所有必需的包，然后加载所有数据。

![](img/ccb709e949e6028a742a5632f6dcb292.png)![](img/a5a323eb585a26dcf3f927991834c04a.png)

让我们找出哪些行的权重为空，然后找出所有要素中缺失值的总和。

![](img/5ebc6fa7e841c1912fb027f9c27570b7.png)

让我们定义一个函数，它将为任何数据帧的任何特征给出所有的空值。

![](img/69ada7b49270c411af920adad3584a2b.png)

删除不必要的列

![](img/18cae7c56c2453ed1a4401edb06672a2.png)

用 NAN 替换所有负值，并找出身高和体重中 NaN 值的数量。

![](img/f717fafc85b1318b046596df13bb15d1.png)

使用中位数输入值，并创建一个仅包含身高和体重的新数据框。

![](img/1f2478f467d3a0274f364b05fd8f8de3.png)

每个出版物的超级英雄分布

![](img/31eaede8d18809f9a8b06f1bdb3578b7.png)![](img/3a90f3aac858b61268eb5c842a5ec7dc.png)

当英雄的数量多于恶棍的数量时就没什么意思了，所以让我们来看看

英雄数量对恶棍数量

![](img/8ea622819ffc90340f297d5ca5a9425a.png)![](img/1246b78016e0a779b38a481267a77eec.png)

让我们使用 plotly 库来绘制它

![](img/c5959fbb31085c5691ec623be94e37e2.png)

数据可视化

![](img/3149f7d8a718f39ca7bca785c2388486.png)

条形图

让我们找出性别分布并画出来。

![](img/9358c0ba59b03adcf6798c54ffa8a4b8.png)![](img/192aab189656207ebb6daaeee504ee51.png)

性别分布按好的排列，坏的排列，中性的排列，未知的排列并画出来。

![](img/528e41af34c593a97371abc5ffcc8e87.png)![](img/f3f234c4c9e66f09b29e8a1a6d1b7c13.png)![](img/7f1287c82ce4945df5ff6e3d0540ea55.png)![](img/15c9cdd4576115d81cada703642ae323.png)

按性别排列的分布

![](img/063ebc99346a37b0959d735177082672.png)![](img/a744f661d7502f0af71a52b33285b0c6.png)

让我们导入另一个数据集。

![](img/77addc4b42654869b653c1fd10dea4ac.png)

让我们看看有多少超级英雄能像金刚狼一样自愈

让我们计算每个英雄的身体质量指数，看看谁是最健康的，谁是最懒惰的。

![](img/1eaceffb94b7b987ccda4753e741f58b.png)

五大肥胖英雄

![](img/7de84bff397c912f16753bb675908ae1.png)

现在让我们看看灭霸和幽灵的总能量

![](img/6e14ed615b044772a5ab1839a40f7a44.png)

这里很清楚，DC·维利安将击败漫威·维利安，并将成为无限宝石的最佳候选人。

但是惊奇队长会让这种事情发生吗？

让我们看看惊奇队长的全部力量

![](img/f060432c8901dd35152233fd77915907.png)

可悲的是，如果我们只是假设总力量是获得无限宝石的标准，那么 ***SPECTRE 将击败漫威队长*** 。

根据数据集，DC 有这样令人兴奋的别墅，为什么我们没有在电影中看到？

可能以后吧。

也许复仇者联盟可以限制 SPECTRE 在未来获得无限宝石。