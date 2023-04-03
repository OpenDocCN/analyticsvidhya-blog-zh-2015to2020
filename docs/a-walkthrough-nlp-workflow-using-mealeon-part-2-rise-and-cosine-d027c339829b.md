# 使用 MeaLeon 的 NLP 工作流演练:第 2 部分，升余弦

> 原文：<https://medium.com/analytics-vidhya/a-walkthrough-nlp-workflow-using-mealeon-part-2-rise-and-cosine-d027c339829b?source=collection_archive---------17----------------------->

嘿伙计们！

![](img/f2685f5c30d100aa35bc47098155ca87.png)

布拉德·巴莫尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

上次[时间](/analytics-vidhya/a-walkthrough-nlp-workflow-using-mealeon-part-1-prepping-the-data-8ffe46820f56)，我通过谈论我的项目开始了自然语言处理(NLP)工作流程的演练:一个名为 [MeaLeon](http://mealeon.herokuapp.com/) 的全栈机器学习食物推荐器。

在上一篇文章中，我介绍了如何通过结合自然语言工具包(NLTK)、WordNet 词汇化和“专家”知识(我对烹饪的熟悉程度)，将一组文档(食谱)和原始文本成分转换成词根的标记。在本文中，我将介绍矢量化和相似性分析。

**载体去战利品**

为什么我们要对单词进行矢量化？记住，计算机仍然不能像人类一样处理语言。我们必须将真实的文本转换成对数字处理有意义的东西。我们通过创建标记化的词根词部分做到了这一点，下一步我们将在这里进行一些数学计算。

我们要做的是在一个大维度空间中为每个成分列表创建一个向量。

![](img/2ac90fb1ea220da9e39f3b2918a6551e.png)

如果你不熟悉数学，请放松。可以这样想:当你问谷歌地图或苹果地图如何到达某个地方时，你会得到“2D”空间的指示。

题外话:如果有海拔变化，技术上来说会是 3D 的，但我有点认为你总是在地面上，当你在路上时，没有人会告诉别人去海拔高或低的地方。

当你取这些方向时，你可以用数学方法把它们加起来，得到一个方向向量，它会告诉你在北/南和东/西轴上需要走多远。我们将为我们的食谱扩展这一点，将每种配料变成一个轴，这样每个配料列表都可以描述为更大空间中的一个向量。我想在这里画出一个食谱样本，但这是不可能的:MeaLeon 实际上有点小，只有 2000 多种配料，但这意味着有 2000 多个轴，这无法在 2D 空间中显示。

![](img/3c5651ca8814f1cd31b6edfaa414a29a.png)

现在，具体的实现是通过 scikit-learn 的 CountVectorizer 完成的。通过调用单独的函数和库来完成所有这些步骤是可能的，这是我最初做的，但是您将意识到您可能必须使您的结果适应 scikit-learn 更喜欢使用的东西……所以您还不如将所有东西都放在一个管道中。

**度量衡:有人喜欢(一个)热**

有几种方法可以创建单词向量。最不复杂的可能是使用一个 Hot 编码器，该编码器简单地将单词的存在表示为 1 或 0。在这里，我实际上将使用 CountVectorizer 来执行每个成分列表的一个热编码。在 MeaLeon 的工作流程中，实际的 OneHotEncoder 出现了不寻常的问题，并且与可能更好的术语频率-逆文档频率(TF-IDF)矢量化方法不兼容。

**TF-IDF TL；博士**

什么是 TF-IDF？就此引用[维基百科文章](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)的第一行:“在[信息检索](https://en.wikipedia.org/wiki/Information_retrieval)，**TF–IDF**或 **TFIDF** ，简称**词频–逆文档频率**，是一个数字统计量，旨在反映一个词对于集合中的[文档](https://en.wikipedia.org/wiki/Document)或[语料库](https://en.wikipedia.org/wiki/Text_corpus)的重要性。”

我会给出一个食品/饮料的类比:当麦当劳和星巴克第一次出现时，检查各个位置是很酷和有趣的。现在，由于它们无处不在，它们实际上并没有那么特别。如果你喜欢这两个或其中一个地方，我无意冒犯你，但让我们现实一点:当你走出一家星巴克，看到另一家星巴克，你点的星冰乐会不会觉得不可替代？

接下来，让我们回到米利安！

在完成 EDA 和[阅读文档](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)之后，我意识到 CountVectorizer 可以被配置为只返回唯一的令牌。这意味着我可以很容易地重构我的逻辑，同时保留数据管道，以备后用 [TF-IDF 矢量器](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)。当您查看文档时，您可以看到它与 CountVectorizer 的相似之处。原因何在？嗯，在 TF-IDF 矢量器页面中提到过，但是 TF-IDF 矢量器实际上无论如何都是使用 CountVectorizer 的。

现在，TF-IDF 矢量器是一种更好的生成矢量的方法，因为它考虑了所有成分列表中每种成分的出现，并降低了频繁出现的单词的重要性。对于 MeaLeon，配料“盐”和“胡椒”在逻辑上应该被认为不太重要:这两种配料应该出现在几乎每个食谱中，不应该被认为像“柠檬草”或“鸭肉”一样重要。顺便说一句，如果你没有在甜点中放一点盐(比如巧克力饼干)，我认为你应该考虑一下=)

好了，现在向量空间和它们的 transformer 对象已经创建好了，我们可以使用这个 transformer 把新的菜谱转换成搜索向量。更重要的是，这些搜索向量可以与一些度量标准一起使用，以计算与数据库的相似性。对于 MeaLeon，我已经通过 CountVectorizer 或 TFIDF 使用了 OneHot，现在必须决定相似性度量。一般来说，这些可以分为两类:距离或余弦相似性。

**盼儿和远:距离还是余弦相似？**

距离通常是[欧几里得距离](https://en.wikipedia.org/wiki/Euclidean_distance)，与我们在二维空间中使用的距离非常相似。这基本上是“这是我们正在看的东西，而我们正在看的另一个东西离我们有多远”。对于 MeaLeon 来说，欧几里得距离不是计算距离的好度量。为什么？好吧，一个单独的食谱可以有许多不同的成分，为了简单起见，让我们假设一个食谱有 10 种成分，相应的食谱成分向量在 10 个维度中有 1，在所有其他维度中有 0。如果我们使用一个有 10 种完全不同成分的新食谱，并找出它们之间的欧几里德距离，我们将得到 Sqrt(20)。但是如果我们使用一个有 20 种成分的食谱，其中 10 种是相同的，10 种是完全不同的，欧几里得距离仍然是 Sqrt(20)。您可能会问“这是不是太具体了？这种事什么时候会发生？”

我立刻想到了巧克力饼干和辛辛那提辣椒意大利面的噩梦。以下是两者的简化版本:

[巧克力饼干](https://www.allrecipes.com/recipe/10813/best-chocolate-chip-cookies/):

面粉

碳酸氢钠

盐

黄油

糖

蛋

香草精

绍科拉特

辛辛那提辣椒配意大利面:

牛肉

奥尼恩斯（姓氏）

番茄

醋

伍斯特郡酱

大蒜

辣椒粉

莳萝

肉桂色

红辣椒粉

丁香

所有香料

月桂叶

盐

绍科拉特

是的，辛辛那提辣椒里有巧克力。除此之外，所有的成分都不一样！哦，等等，我们还没有添加来自[意大利面条/意大利面](https://www.allrecipes.com/recipe/11991/egg-noodles/?internalSource=rotd&referringId=494&referringContentType=Recipe%20Hub)的独特配料:

面粉

蛋

黄油

欧几里得距离在这里会变小，因为 5 种成分的重量会变为零，但是没有人会建议辛辛那提辣椒配意大利面来代替巧克力饼干。或者根本没有。这份食谱忽略了提及辣椒的水状/稀薄稠度或通常加在上面的堆积如山的奶酪。

无论如何，欧几里得距离在这里不是一个好的选择，因为它倾向于在距离计算中强调矢量的大小。

相反，MeaLeon 使用[余弦相似度](https://en.wikipedia.org/wiki/Cosine_similarity)。这降低了量值的重要性，反而更关心分量向量的方向。在这里，这应该允许我们在一个[的多维空间](https://www.machinelearningplus.com/nlp/cosine-similarity/)中更好地比较不同的食谱。

好了，我们已经有了配方向量和在向量之间进行比较的方法…是时候看看我们得到了什么了？让我们把这个留到下次吧，因为我们已经超过 1000 个单词了！

与此同时，如果你有问题或意见，请留下，我会尽我所能解决它们！希望能很快见到你们！

来源:

[](/analytics-vidhya/an-overview-of-word2vec-where-it-shines-and-where-it-didnt-cb671b68a614) [## Word2Vec 的概述，它在哪里发光，在哪里不发光

### 嘿伙计们！我带着一个我在 MeaLeon 工作时非常熟悉的话题来到这里。自然语言…

medium.com](/analytics-vidhya/an-overview-of-word2vec-where-it-shines-and-where-it-didnt-cb671b68a614) [](http://mealeon.herokuapp.com/) [## 米利安

### 你喜欢吃什么？在文本框中输入，从下拉菜单中选择菜肴，然后返回…

mealeon.herokuapp.com](http://mealeon.herokuapp.com/)  [## tf-idf

### 在信息检索中，tf-idf 或 TFIDF 是词频-逆文档频率的缩写，是一种数值型词频。

en.wikipedia.org](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) [](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) [## sk learn . feature _ extraction . text . count vectorizer-sci kit-learn 0 . 22 . 1 文档

### class sk learn . feature _ extraction . text . count vectorizer(input = ' content '，encoding='utf-8 '，decode_error='strict'…

scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) [](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) [## sk learn . feature _ extraction . text . tfidf vectorizer-sci kit-learn 0 . 22 . 1 文档

### class sk learn . feature _ extraction . text . tfidf vectorizer(input = ' content '，encoding='utf-8 '，decode _ error = ' strict '……

scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) [](https://en.wikipedia.org/wiki/Euclidean_distance) [## 欧几里得距离

### 在数学中，欧几里得距离或欧几里得度量是两点之间的“普通”直线距离…

en.wikipedia.org](https://en.wikipedia.org/wiki/Euclidean_distance) [](https://www.allrecipes.com/recipe/10813/best-chocolate-chip-cookies/) [## 最佳巧克力饼干

### 酥脆的边缘，耐嚼的中间，所以，很容易制作。试试这个广受欢迎的巧克力曲奇食谱吧。

www.allrecipes.com](https://www.allrecipes.com/recipe/10813/best-chocolate-chip-cookies/) [](https://www.allrecipes.com/recipe/206953/authentic-cincinnati-chili/) [## 正宗辛辛那提辣椒食谱

### 耗时 2 天的正宗食谱。你的是双份的(装在一大盘意大利面条里)；3 路(添加切碎的…

www.allrecipes.com](https://www.allrecipes.com/recipe/206953/authentic-cincinnati-chili/) [](https://www.allrecipes.com/recipe/11991/egg-noodles/?internalSource=rotd&referringId=494&referringContentType=Recipe%20Hub) [## 鸡蛋面食谱

### 美味的家庭自制鸡蛋面条是一种简单的面粉，盐，鸡蛋，牛奶和黄油揉成一个面团，滚…

www.allrecipes.com](https://www.allrecipes.com/recipe/11991/egg-noodles/?internalSource=rotd&referringId=494&referringContentType=Recipe%20Hub)  [## 余弦相似性

### 余弦相似性是内积空间中两个非零向量之间相似性的一种度量

en.wikipedia.org](https://en.wikipedia.org/wiki/Cosine_similarity) [](https://www.machinelearningplus.com/nlp/cosine-similarity/) [## 余弦相似性-理解数学及其工作原理？(使用 python)

### 余弦相似性是一种度量标准，用于衡量文档的相似程度，而不考虑文档的大小。数学上…

www.machinelearningplus.com](https://www.machinelearningplus.com/nlp/cosine-similarity/)