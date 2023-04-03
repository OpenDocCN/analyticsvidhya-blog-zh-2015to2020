# clump——用于轻松查找相关故事和文档的 Python 包

> 原文：<https://medium.com/analytics-vidhya/clump-a-python-package-for-easily-finding-related-stories-and-documents-3ed992a35f8c?source=collection_archive---------20----------------------->

![](img/9f34693724e2ccf1cff1a7129a8ce134.png)

来源 [MichaelGaida](https://pixabay.com/users/MichaelGaida-652234/) 通过 [pixabay](https://pixabay.com/photos/news-daily-newspaper-press-1172463/)

易于使用的库，用于分组、标记和查找相似的文档，使用先进的自然语言处理技术。

这个包的最初目的是寻找相似或相关的文档。这方面的典型用例是希望显示特定新闻故事相关内容的新闻应用程序。

许多开发人员没有时间投资学习这方面的最佳实践，因此该模块提供了一个简单的包，用于加载所有要考虑的内容，然后一个函数，给出一段文本，将找到所有相关的故事。

在这篇文章中，我们将看看如何从多个网站加载新闻，并为某个特定的提要找到相似的故事。

## 入门指南

第一步是在你的终端上安装`pip`包。在这个例子中，我们还将使用`feedparser` 包从新闻网站获取提要。使用以下命令从终端安装这些组件:

```
$ pip3 install clump
$ pip3 install feedparser
```

## 正在加载新闻源

为了获取一些内容，我们将从一些新闻网站加载 feeds 来查看一些最近的内容，并找出其中的相似之处。

这可以通过以下方式实现:

```
import feedparserfeeds = ["https://www.theguardian.com/uk/rss",
    "https://www.telegraph.co.uk/news/rss.xml",
    "http://rss.cnn.com/rss/cnn_topstories.rss"]# load all articles from the feeds
training_articles = [entry.title for url in feeds for entry in feedparser.parse(url).entries]print(training_articles)
```

运行这段代码应该会显示一组新闻标题。

## 构建模型

扩展上面的代码，我们可以将这些文章从`clump`传递到 VectorSpaceModel，以便训练算法:

```
from clump.model import VectorSpaceModelmodel = VectorSpaceModel(training_articles)
```

## **寻找相似的故事**

现在，我们可以查询我们的模型来查找与新故事相似的训练过的故事。写这篇文章时恰逢国际妇女节，这是一个非常受欢迎的主题。以下是一个查询示例:

```
related = model.find_similar('International Women's Day: Duchess of Sussex surprises schoolchildren')
```

这返回了以下故事:

```
5 facts about International Women’s DaySurprise! Duchess Meghan goes to church with the queen, marks International Women’s Day at a school assembly#EachforEqual is the theme for International Women’s Day 2020\. Here’s what you should know.
```

## 高级选项—最大匹配数

默认情况下，`find_similar`最多返回 5 条记录。这可以通过为`n`输入一个特定值来改变，如下所示:

```
related = model.find_similar('Coronavirus: How to wash your hands - in 20 seconds', n=3)
```

**高级选项—相似性阈值**

当您涉及一系列主题时，可能很难找到密切相关的内容。这种方法中的基本算法考虑文本片段之间的相似性百分比，介于 0 和 1 之间(0 =无相似性，1 =相同)。默认情况下，这是 0.4，因此如果我们不太担心相似性，我们可以将其减少到 0.3:

```
similar = model.find_similar('Coronavirus: How to wash your hands - in 20 seconds', distance_threshold=0.3)
```

结果按相关性排序，因此最相关的故事出现在结果的最前面。

## 将所有这些放在一起:完整的示例

对于这个例子，我们将根据来自 10 个不同新闻源的故事来训练我们的模型，然后看看 BBC 的最新故事是如何关联的。

```
 import feedparser
from clump.model import VectorSpaceModelfeeds = [“https://www.theguardian.com/uk/rss",
   “https://www.telegraph.co.uk/news/rss.xml",
   “http://rss.cnn.com/rss/cnn_topstories.rss",
   “http://rssfeeds.usatoday.com/usatoday-NewsTopStories",
   “https://abcnews.go.com/abcnews/usheadlines",
   “http://rss.nytimes.com/services/xml/rss/nyt/World.xml",
   “http://feeds.feedburner.com/time/newsfeed",
   “https://feeds.a.dj.com/rss/RSSWorldNews.xml",
   “http://feeds.foxnews.com/foxnews/latest",
   “https://www.huffpost.com/section/front-page/feed"]test_feed = “http://feeds.bbci.co.uk/news/rss.xml"# load all articles from the feeds
training_articles = [entry.title for url in feeds for entry in feedparser.parse(url).entries]
test_articles = [entry.title for entry in feedparser.parse(test_feed).entries]# create the model
model = VectorSpaceModel(training_articles)# find all the similar items for the test feed
for test_article in test_articles:
    similar = model.find_similar(test_article, distance_threshold=0.3)
    if len(similar) > 0:
        print(test_article + “\n\t” + “\n\t”.join(similar) + “\n”)
```

非常感谢对功能的任何意见或要求。