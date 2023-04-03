# 被数字愚弄了。辛普森悖论。

> 原文：<https://medium.com/analytics-vidhya/fooled-by-the-numbers-the-simpsons-paradox-e703c93d29c4?source=collection_archive---------20----------------------->

![](img/b7d9326218c6b46bdf31f24ce8d998b6.png)

Amber Lamoreaux 拍摄的图片

人们喜欢绝对。一个事物不是黑就是白。不是两者都有。绝对不是彩虹。这很难正确衡量，因此也不可能管理。

作为一名数据分析师，我经常被要求提供一个简单的总数。一个集合体。一个单一的标准来统治他们。在网络分析领域，这个数字通常是转化率。有多少访问过我的网站的人按照我的要求做了？

在我们的假网站上，我们推出了两个主题，一个是哈利波特的魔法，另一个是星球大战的力量。我们问我们的访问者他们更喜欢哪个版本。因为我们的网站有一个非常初级的开发者，所以我们不能运行 A/B 测试。结果我们刚推出《哈利波特》一周，下一周又推出《星球大战》。

> ***提示:*** *不要在你的网页上以这种方式测试。*

![](img/75ce015ec83f2f961b74d800a12414e9.png)

结果具有统计学意义。

哈利波特的魔法打败了原力使用者。就算是黑暗面的力量也救不了星球大战主题的设计者。我们就实现哈利波特主题吧。

不会这么快。

在我们测试《星球大战》主题的时候，有一些公共假日，更多的人通过移动设备而不是个人电脑访问网络。当我们只查看移动结果时，我们观察到以下情况。

![](img/2f6b3512f83bd58e01982758a4d789da.png)

结果在统计学上是显著的。

看起来星球大战在手机上赢了。

所以，如果总体结果是哈利波特在手机上受欢迎，而《星球大战》受欢迎，那就意味着哈利波特在电脑上一定大获全胜，对吗？

![](img/ac0132759cc9fbbb47677f7ffe624cd5.png)

结果在统计学上是显著的。

星球大战赢了。“这是什么黑魔法？”哈利问。

《星球大战》怎么可能在这两种情况下都更好，但总体上却更差？

这被称为**辛普森悖论。**

# 定义

**辛普森悖论**是概率统计中的一种现象，一个趋势出现在几组不同的数据中，但当这些组组合在一起时就消失或反转。

**潜伏变量**是既影响因变量又影响自变量的变量，造成虚假关联。混杂是一个因果概念，因此不能用相关性或关联性来描述。

```
*We are evaluating the performance of our web based on the conversion rate. However, the conversion rate is impacted by the type of device the user has. The data show that in week when Star Wars was tested more users chose mobile to visit our web which influenced both the visits and the conversion rate. In this example the device is the lurking variable.*
```

# 如何处理辛普森悖论？

因为有很多方法可以确保你不会成为辛普森悖论陷阱的受害者，所以我决定将其简化为三个直接行动。

1.  质疑汇总数据。

试着想一想，是否存在多个群体，其中被测试的“事物”的效果各不相同。

```
*The conversion rate varies greatly between PC and mobile. The device of the user is in causal relation with the conversion rate. It is lower on mobile than on the PC.*
```

> ***动作:*** *将总数据分解成更小的组。验证是否在小组中也观察到了结果。*

2.获取尽可能多的数据背景。没有上下文的数据应该被理解为试图操纵。

```
*You marketed the design of the new theme for your web as a competition. The designer whose proposal is better gets paid. The other doesn't. The Harry Potter one knows that on public holidays more people visit on the PC where the conversion rate is higher. That is why he rushed his design to be completed and tested before the holidays start.*
```

> ***动作:*** *问题数据生成。我们是否在相似的条件下收集了数据？我们在征求反馈时使用了同样的方法吗？*

3.试着为你的实验找出任何潜在的变量。哪些因素影响了数据没有显示的结果？

```
*We should have mapped all potential causes influencing the conversion rate. Device is one of them, time, demography of the visitors might play a role as well. Either test only on specific audience or evaluate the results for all sub-groups.*
```

> ***动作:*** *拼凑出详细的因果图。思考数据中的潜在群体及其对结果的潜在影响。画一张什么影响什么的地图。*

# 将数据分成更小的组似乎是所有问题的解决方案，那么可能会有什么问题呢？

在我们的小例子中，我们只处理了两个主题和两个设备。想象有 5 个主题，4 个设备，10 个操作系统，20 个浏览器和 30 种语言等等。您可以将数据分成的类别数量没有限制。你分割的越多，每组得到的样本量就越小。由于样本量较小，您的结果可能不具有统计学意义，因此您根本无法确定主题之间是否存在差异。

另一个问题存在于决策阶段。如果你发现每个网页浏览器都有不同的主题表现最佳，你会投资个性化开发你的网页吗？你会在不同的主题中设计和开发每个新功能吗？如果你负担不起个性化的方法，那么你可能会瞄准最适合大多数用户的主题。

这让我们进入了最重要的部分——决策。

# 如果总体趋势与子组中的趋势不匹配，我应该如何判断？

你需要意识到数据中的因果关系。你需要知道你想要什么。没有适用于所有情况和所有目的的正确决策。这要看情况。

```
*If the goal is to apply the theme which will bring you more conversions, decide for the Star Wars theme. It performs better on both devices.**If you took a bet with a friend which design performed better and he will ask 20 random people who have seen the Star Wars theme and 20 random people who have seen the Harry Potter theme whether they liked it, choose the Harry Potter theme.*
```

为了更好地阐明我所说的知道你想要什么，这里还有一个例子。

```
*If a political party tells you how they lowered all taxes compared to the previous government, beware. Maybe they are just misusing the Simpson's paradox for their own gains.* *They lowered the tax for people with income below 1k EUR monthly to 5% to help the lowest income families. At the same time they lowered the tax for all other incomes to 20%.*
```

你在比较两个不同的时间段，为了验证政府是否操纵，你需要分解数据。

```
*Thanks to the overall economic growth some families could have moved from a lower income group to a higher income group. This means previously they have been paying 10% lowered to 5% and now they are in a group which used to pay 21% lowered to 20%.
The income group is the lurking variable in this case.*
```

这种组间转换极大地影响了每组的样本，从而影响了效果。

```
*With the economic growth in place almost no people are below 1k EUR. This means that for many people the tax changed from previous 10% to 20%. In absolute values the government collects more in taxes than the previous government even though they lowered all the taxes.*
```

如果你的问题是，政府是否会降低所有的税收，那么答案是不会。他们降低了税收，这是事实，但人们支付的总税收增加了，他们没有说。

低税率对你更有利吗？如果你在同一个组，那么是的。如果你进入了高收入群体，那么对你来说就不是了。你比以前支付更多。

> ***外卖:*** *用数据阐明你要回答的问题。确定可能影响结果的因素以及需要考虑的因素(潜在的变量)。将数据分成相关的组。确保你了解数据收集的方法，以及它是否与解释一致。*

# 辛普森悖论的常见潜伏变量

性别、工作日、年份、设备类型、操作系统、价格类别、客户的收入水平、购买的产品类型、案例的严重性

# 参考

[维基百科:辛普森悖论](https://en.wikipedia.org/wiki/Simpson%27s_paradox)

[维基百科:混淆](https://en.wikipedia.org/wiki/Confounding)

[简介辛普森悖论— jupyter 笔记本示例](https://github.com/WillKoehrsen/Data-Analysis/blob/master/statistics/Simpson's%20Paradox.ipynb)

[辛普森悖论:如何用一个数据集证明两个相反的论点](https://towardsdatascience.com/simpsons-paradox-how-to-prove-two-opposite-arguments-using-one-dataset-1c9c917f5ff9)

[加德纳·马丁:关于归纳逻辑的结构和一些概率悖论](https://flowcytometry.sysbio.med.harvard.edu/files/flowcytometryhms/files/herzenbergfacshistory.pdf)

[维基百科:因果图](https://en.wikipedia.org/wiki/Causal_model#Causal_diagram)

[论辛普森悖论和万无一失原则](https://www.jstor.org/stable/2284382?seq=1)

[微观物理学:辛普生悖论](https://www.youtube.com/watch?v=ebEkn-BiW5k)