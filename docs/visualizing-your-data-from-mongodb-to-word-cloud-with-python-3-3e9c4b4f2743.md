# 使用 Python 3 将数据从 MongoDB 可视化到 Word Cloud

> 原文：<https://medium.com/analytics-vidhya/visualizing-your-data-from-mongodb-to-word-cloud-with-python-3-3e9c4b4f2743?source=collection_archive---------10----------------------->

*   [技术堆栈](#0d55)
*   [连接到 MongoDB](#e758)
*   [分词](#177b)
*   [生成文字云图片](#80da)
*   [翻译成另一种语言](#f290)

词云现在已经变得越来越流行，因为它是一种直接而有效的数据交流方式。主要优点是它有助于简化理解复杂图表的过程，这些复杂图表通常需要一些先决知识来帮助消化。

一个规则代表了单词云的简单性——文本大小，即文本越大越重要。这个显而易见的特性保证了来自各种背景的人能够容易地理解内容。

在本文中，我们介绍了一个将从幼儿日常活动中收集的数据转换为 word cloud 的示例。word cloud 的设计有助于将幼儿的活动可视化，使家长和看护者可以轻松看到，然后利用这种方式来培养他们教育孩子的方式。

在深入研究技术细节之前，让我们先看看结果。

![](img/de376269f1ae1e08cea820b86ce352a2.png)![](img/0027823f705fc3d3a0cfe24bc94ce38b.png)

上面的单词云是基于老师在一段固定时间内对一个孩子的评论。左边的云是评论中使用的原始语言，右边的词云是其中文翻译。比其他字体大的单词突出了孩子的注意力和与他人交流的情感。

# 技术栈

**MongoDB** :存储采集的数据
**Python 3** :处理数据并生成文字云图片
**库** : [spaCy](https://spacy.io/) ， [Numpy](https://numpy.org/) ， [wordcloud](https://amueller.github.io/word_cloud/) ， [googletrans](https://pypi.org/project/googletrans/)

# 正在连接到 MongoDB

选择托管的 MongoDB 云服务 MongoDB Atlas ，是因为 M0 沙盒是免费的，它节省了部署和维护本地数据库的时间。

连接细节可以放入环境变量中，并从 Python 中读取，如下所示。如果需要使用 Docker 和 Kubernetes 将应用程序作为服务部署到云中，那么使用环境变量是一种很好的做法。

```
MONGO_USERNAME = os.getenv('MONGO_USERNAME')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
MONGO_URL = os.getenv('MONGO_URL')
```

然后需要初始化 MongoDB 客户端。所有与 MongoDB 相关的代码都可以放在一个单独的 Python 文件中，以便组织和将来重用。

在本例中,`db`将从命令行初始化，因此这里为空。如下添加了一个用于读取集合的简单函数。

从主脚本中检索数据。

`collection`是 MongoDB 集合名，`data_entry`是值的键名。这里有一个样本数据。这里用的`data_entry`就是下面的`detail`。

# 标记单词

在上一节中，段落被加载到一个数组中。如果数据量很大，可以考虑流式传输。

这些段落需要分解成单词。这听起来像是一个简单的工作，但是需要大量的努力来确保最终结果的准确性。处理自然语言对计算机来说是一个挑战，因为它们永远无法直接理解单词。

自然语言处理和机器学习是大课题。由于本文关注的是单词 cloud，所以只考虑几个步骤来处理这些文本。

1.  丢弃不需要的符号
2.  删除组织和人员姓名
3.  规范化单词(词干)
4.  过滤停用词

spaCy 是自然语言处理的强大工具。我们用它来标记名字和词干，这只是冰山一角。

第一步是初始化空间并加载没有不需要的符号的数据。数据被加载到名为`content`的数组中，空间由`en_core_web_sm.load()`初始化

其次，我们使用 spaCy 来过滤组织和人名。它们被分类为`label_`等于`PERSON`和`ORG`。命名实体的完整列表可在[这里](https://spacy.io/api/annotation#named-entities)找到。

然后，我们在一个循环中进行词干提取、标点符号删除和停用词删除。可以通过获取`lemma_`属性来提取令牌的基本形式。

基于对数据的研究，我们决定也有一个定制的停用词列表。引入了一个名为`stop-words.txt`的文件供脚本动态加载。

这是一个标记化单词的例子。

```
['play', 'dough', 'disco', 'time', 'construction', 'activity', 'afternoon', 'engaged', 'stamp', 'pat', 'play', 'piano', 'play', 'dough', 'piece', 'spend', 'lot', 'morning', 'explore', 'painting', 'equipment', 'stamp', 'roll', 'create', 'different', 'print', 'friend', 'time', 'decorate', 'carefully', 'gingerbread', 'biscuit', 'colourful', 'flake', 'explore', 'conker', 'paint', 'roll', 'shake', 'box', 'show', 'able', 'independently', 'find', 'way', 'suit', 'conker', 'create', 'picture', 'watch', 'closely', 'see', 'different', 'mark', 'conker', 'make', 'move', 'keen', 'help', 'teacher', 'create', 'gloop', 'morning', 'spoon', 'mix', 'water', 'corn', 'flour']
```

# 生成 Wordcloud 图像

随着数据的收集和处理，下一步是将其可视化为单词云图像。 [Numpy](https://numpy.org/) 和 [wordcloud](https://amueller.github.io/word_cloud/) 让这一步变得简单明了。

与其他在线单词云网站类似，可以选择一个遮罩图像作为基础图像。然后，可以调整相当多的参数，如`font_path`和`background_color`，以生成符合您特定需求的图像。例子可以在官方的 wordcloud 库网站[这里](https://amueller.github.io/word_cloud/auto_examples/index.html)找到。

# 翻译成另一种语言

由于数据可能来自不同的来源，而单词 cloud 受众可能来自不同的国家，因此可能有必要将单词翻译成另一种语言以便更好地理解。

[googletrans](https://pypi.org/project/googletrans/) 是一个非官方的 Python 库，实现了 Google Translate。我们在代码中初始化这个库，并且一次发送一组单词给 Google Translate。

做翻译似乎很简单，但是需要注意一些问题。

1.  该库不是来自谷歌，所以稳定性没有保证。
2.  单次提交的最大文本大小为 15k
3.  单词在不同的上下文中可能有不同的翻译
4.  有些翻译需要更正

如果稳定性是一个问题，建议使用官方的谷歌翻译 API。

根据我们的令牌化数据，可以通过对多次提交的数组进行分段来简单地限制大小。

最困难的问题是上面的第三点。如果单词的翻译需要基于上下文，其他复杂的处理也需要到位。

在本文中，我们采用一种简单的方法来翻译没有上下文的单词。由于远程 Google Translate API 调用的时间开销很大，我们无法一次提交一个单词。最简单和最直接的解决方案是找到一个好的分隔符，并将其添加到单词列表中，使翻译与上下文无关。在下面的代码片段中，令牌以 500 个为一组提交，以`>`作为分隔符。

最后一步是检查你的翻译，并纠正那些与整体上下文无关的意思，否则翻译就是不够好。新的翻译映射文件`translation-map-zh-cn.csv`用于如下校正。左侧的文字将由右侧的文字替换。

```
股票,分享
铅,带头
匙,勺子
```

读取文件并进行更正的代码如下。

总之，本文涵盖了 4 个主题。连接到远程 MongoDB，标记单词，生成单词云图像，最后将单词翻译成其他语言。目标是用最有意义的单词生成单词云。根据收集的数据和单词云的目的，对原始数据的深入研究对于制定自己的单词标记化计划至关重要。

这里是这个展示项目的 GitLab 链接。如何为自己的数据生成文字云，请看[自述](https://gitlab.com/conanzxn/mongodb-wordcloud/-/blob/master/README.md)。欢迎公关！