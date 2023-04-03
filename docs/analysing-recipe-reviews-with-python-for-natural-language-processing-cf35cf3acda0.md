# 用 Python 自然语言处理分析菜谱评论

> 原文：<https://medium.com/analytics-vidhya/analysing-recipe-reviews-with-python-for-natural-language-processing-cf35cf3acda0?source=collection_archive---------10----------------------->

![](img/2bd3cc7635aac6ae7846a2c1d6161d03.png)

布鲁克·拉克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

# 情感分析

在这篇博客中，我将重点关注情感分析，这是自然语言处理的一个方面。我将演示我们如何将一系列综述作为单独的字符串读取，处理数据以便进行分析，以及我们如何使用 Sci-Kit Learn (SKLearn)、自然语言工具包(NLTK)和术语频率-逆文档频率(TF-IDF)来完成这些工作。

## 为什么有用？

情感分析[SA]的工作原理是试图理解单词或短语背后的情感或语气，它有着广泛的应用。它主要应用于社交媒体网络，以了解公众对世界新闻、商业产品、政治和各种其他领域的看法。公司可以利用这一点进行有价值的反馈和市场研究，交易员可以根据对政府政策变化的反应来预测股票市场的波动，在某些情况下(如剑桥分析公司在脸书的帮助下)可以用来操纵公众舆论。

然而，NLP 的应用并不局限于此，当然也不仅仅是经济上的动机。医疗保健也有应用。有人分析社交媒体上的帖子来预测抑郁症[1]，分析电子健康论坛上的阅读用户帖子，以帮助对医疗状况进行分类[2]，并改善医疗保健和患者体验[3]。

在这篇博客中，我将使用来自 Food.com(之前的 GeniusKitchen)的菜谱评论对来自 Kaggle 的[数据集进行 SA。幸运的是，这些评论带有相关的评级，这大大简化了这个过程，因为我们不必评估它是积极的还是消极的陈述。](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions#RAW_interactions.csv)

# 处理数据

我使用 pandas library 读取数据，并根据与每个评论相关的评级创建单独的数据框架——如果评级为 4 或 5，则为正面；如果评级为 1 或 2，则为负面；如果评级为 3，则为中性。总的来说，数据集包含超过 100 万条评论，但是为了公平地训练模型，我们需要偶数条正面和负面评论。

![](img/d516aebc045992441b1e8b5ddc3ed67d.png)

从这里，我从每个组中随机抽取 26，000 条评论，并将每个组转换为一个列表，然后将它们分成两半，这样就有了一个训练和测试集。现在我们需要清理每一个评论进行分析！

## 正则表达式

正则表达式[regex]是 python 的一个库，它允许你为 NLP 处理某些单词、符号和短语。如果你想了解这是如何做到的，他们有非常好的文档，可以在这里找到。
下面我创建了两个函数，一个用来删除标点符号，另一个用来删除换行符之类的符号——这是收集数据的结果。

![](img/cb4b8a7442d55fd7c041361f2c59f7fc.png)

这是清洁前后的样子:

```
Before cleaning:
DELICIOUS!!!!!  I made 1/2 of the recipe for a late breakfast/brunch today and just loved these treats.  I used a frozen pie shell that I thawed out, but think next time I would use the refrigerated kind or even try the phyllo or puff pastry, as my dough was rather dry after being frozen.   Thanks for sharing the recipe Scoutie!!  Made for Newest Tag.After cleaning:
delicious  i made 1 2 of the recipe for a late breakfast brunch today and just loved these treats  i used a frozen pie shell that i thawed out but think next time i would use the refrigerated kind or even try the phyllo or puff pastry as my dough was rather dry after being frozen   thanks for sharing the recipe scoutie  made for newest tag
```

从这里你可以看到，在处理这些数据(filo 或 phyllo，1/2)时仍然会有很多问题，但对于我们的初步模型来说已经足够好了。

现在，我们需要建立我们的语料库(本质上是我们训练集中所有单词的列表)，为此，我将所有评论从正面到负面放在一个列表中。然后，我们可以使用 TF-IDF 或 CountVectorizer 来创建一个矩阵，其中包含我们所有的评论和单词！

# 结果

下面，我首先使用了 TF-IDF，将一个应用于训练模型，一个应用于测试模型。这会将每个评论拆分成单独的单词，并为每个单词分配一个“令牌”。然后，它在每个评论的 vector 上创建一个向量矩阵，其中包含我们语料库中的所有唯一单词，用 1 或 0 表示该单词在该评论中是否存在(这就是 binary=True 所做的)。我们的对象的 n_grams 部分告诉我们要查找的单词数——这里我们将范围设为 1-2，因此我们将获得单个单词以及两个单词的短语。例如，一个人说“爱它”，另一个人可能说“不爱”，这是 n 元语法的完美用例。
然后，我对训练数据进行了训练测试分割，并分别为正面、中性和负面评价分配了 0、1 和 2 个目标变量。这样，我们就可以在应用逻辑回归使我们的模型符合最终测试数据之前，在验证集上测试我们的模型。这不是一个必要的步骤，但是当我们选择使用哪个模型并希望执行我们的最终评估时，可以保持我们的测试数据不变。

![](img/8c73b1be50bfa516ac4314624ea0f048.png)

在这里，我们将目标变量 2、1 和 0 分别分配给正面、中性和负面评论。

由于我们可以预测正面、中立和负面的评论，这在技术上是一个多类问题。然而，这个问题的细节已经超出了这篇文章的范围，所以从这里开始，我将把它作为一个二元问题来处理——消极的和积极的。

现在将它应用于测试数据:

![](img/f12fd9653237a9f8da2b2ad875eb53f0.png)

77% —还不错！X 指的是训练数据回顾，X_final_test 是我们未接触的测试数据。

为了向你们展示这到底做了什么，我将向你们展示我们语料库的词汇，以及我得到的积极和消极词汇的最准确的值。以下是语料库(' word': token):

![](img/d273b02413de19400ee0626dc76af5ce.png)

最准确单词的结果是:

![](img/e0b2d3676c148814ad1e37307b16daf4.png)

这似乎很符合人们的预期。

# 深加工

从上面可以看出，我们的模型精度还不错！尽管如此，我们仍然有办法改进这一点。一种常见的技术是删除停用词(如 it/he/she/the/as ),它扭曲了语料库中所有词的频率，尽管我发现删除停用词实际上会降低我的模型准确性，所以我选择保留它们。然而，我们也可以看到，最好的话有重复(不是，不是)。这就是词干分析和词条解释的用武之地，它们将这些“重复”归结为一个词——失望，失望都变成了失望。这为我们提供了文档中这些单词的实际计数的更好的准确性。

# 观点

这里还有很多工作可以做，以改进我们的模型，我希望在未来的博客帖子中做这些工作。我还将尝试使用除逻辑回归之外的其他模型将此作为多类分类模型，因为由此得出的结果非常有趣！

另外，如果你觉得这很有用或者喜欢这篇博文，请不要忘记！任何问题和意见也非常欢迎。感谢阅读！

## 引用和资源

感谢 Shuyang-Li 提供此数据集:[https://www . ka ggle . com/Shuyang l94/food-com-recipes-and-user-interactions # RAW _ interactions . CSV](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions#RAW_interactions.csv)

[1]王 x，张 c，季 y，孙 l，吴 l，鲍 z .【2013】【微博社交网络中基于情感分析的抑郁检测模型】。载于:李 j 等(编)*知识发现和数据挖掘的趋势与应用。PAKDD 2013。计算机科学讲义，第 7867 卷。施普林格、柏林、海德堡*

[2] Shweta Yadav，Asif Ekbal，Sriparna Saha，Pushpak Bhattacharyya (2018)使用社交媒体的医疗情感分析:建立患者辅助系统*第十一届语言资源与评估国际会议论文集(LREC 2018)*

[3]Greaves F，Ramirez-Cano D，Millett C，Darzi A，Donaldson L
使用情感分析从网上发布的自由文本评论中获取患者体验*J Med Internet Res 2013；15(11):e239*