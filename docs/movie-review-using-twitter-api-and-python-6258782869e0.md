# 使用 Twitter API 和 Python 进行电影评论分析

> 原文：<https://medium.com/analytics-vidhya/movie-review-using-twitter-api-and-python-6258782869e0?source=collection_archive---------10----------------------->

Twitter API 和几个 Python 库可以用来检索和分析关于特定电影的公共 tweets，以确定观众对它的感受。这是一篇关于我如何做到的短文。

![](img/01a76226c4d5e0f16b2c009db8492ae3.png)

这篇文章是为那些想使用 Twitter API 并做一些简单但对人们有用的事情的初学者写的。如果你对 API 非常陌生，你必须首先阅读更多关于 API 的一般知识([点击这里](https://www.upwork.com/hiring/development/intro-to-apis-what-is-an-api/))或者如果你知道什么是 API，你可以阅读关于 [Twitter API](https://developer.twitter.com/en/docs/basics/getting-started) 的特性。简而言之，Twitter APIs 允许所有 Twitter 用户访问公共推文、转发、时间表、列表、收藏等，还允许发布、更新或删除推文。我在这篇文章中介绍了使用 Python 连接和访问 Twitter API 的先决条件和基础知识。

Twitter APIs 可以使用 ruby、php、.net，perl，asp，java，python 等。我使用 Python 来访问和分析使用 API 的推文，因为我发现 Python 更容易编码，它为我们提供了大量的库和包可供使用，还因为 Python 的庞大而活跃的用户社区随时准备在你需要时提供帮助。

# 入门指南

首先，这些是先决条件:

## 创建一个 Twitter 应用

Twitter API 可以通过 Twitter 应用程序访问。如果你有一个 Twitter 帐户，你可以很容易地创建自己的应用程序。首先，使用你的 twitter 证书在 [Twitter 开发者门户](https://developer.twitter.com/en/apply-for-access)上申请开发者权限。其次，在他们的[创建应用程序页面](https://developer.twitter.com/en/apps)上创建一个应用程序，在应用程序提交页面填写您的详细信息。如果你没有自己的网站，你可以使用 https://www.twitter.com。同样，在注册应用程序时，您也可以使用 Twitter 条款页面和政策页面。应用程序激活几乎是在你给出所有细节后立即发生的，包括应用程序的用途、示例用例、商业用途等。创建应用程序后，您将在密钥和令牌部分找到 API 密钥和其他令牌。这些是使用您编写的程序访问 Twitter API 所需要的，所以请将它们放在手边。

## 安装 Python 3.7 或任何最新版本

安装 Python 相当简单。你可以简单的按照他们[官方页面](https://www.python.org/downloads/)上的安装步骤。

# 主程序

你可以在这里找到我的 Python 代码。该计划细分如下:

## 图书馆

我使用了两个 Python 库:tweepy 和 textblob。Tweepy 是 Python 的 Twitter 库，专门用于使用 Python 访问 Twitter API。Textblob 是一个用于处理文本数据的库，它提供了执行 NLP(自然语言处理)任务的方法，如词性标注、情感分析、翻译等。

因为这两个不是 Python 中的默认库，所以您必须安装它们。您只需在 cmd 中输入下面几行就可以简单地安装这两个库

```
pip install tweepy
```

cmd 将自动在您的系统上安装 tweepy。同样的，

```
pip install textblob
```

将在您的系统上安装 textblob 库。

## 证明

如前所述，您将需要您的消费者密钥、消费者机密、访问密钥和访问机密，以便为您的代码提供与 Twitter API 进行交互、整理数据和运行分析所需的权限。如果您遵循第 6 行到第 13 行的代码，您会注意到

```
api=tweepy.API(auth)
```

将成为你所有 Twitter 呼叫的处理者。这个“api”拥有授权和认证，可以像您一样访问 Twitter API。

## 搜索推文

我使用 GET 方法调用 Twitter 搜索 API。这在 Python 代码中没有显式显示，因为我使用 tweepy 库来访问 Twitter API。Tweepy 图书馆给你。search('#searchterm ')方法，它负责使用 GET 方法，您只需要担心如何给它提供正确的参数。

Twitter 标准搜索 API 在一个请求中最多只返回 100 条推文。因此，如果您想要分析 100 多条推文(大多数情况下都会这样)，您必须多次调用搜索 API，收集推文，将它们存储在本地，然后进行分析。为此，我编写了一个小循环，运行多次来收集所需数量的 tweets，并将它们存储在本地的 Python 字典中。在这个程序中，我要求用户输入一个搜索词，通常是电影标签(例如#saahoreview)，使用这个标签搜索推文，并存储它们以供进一步分析。

## 情感分析

TextBlob 库是一个非常强大的库，有许多工具和方法来对“文本”进行大量分析。我在这个库中使用了一个小方法来运行我的分析，这相当容易。有了本地存储的推文列表，你现在可以分别对每条推文进行情感分析。下面是如何进行情感分析的总结:

```
analysis=TextBlob(tweet)
tweet_sentiment = analysis.sentiment.polarity
```

如果 Tweet 情绪不好，analysis . perspective . polarity 返回负值，如果 Tweet 情绪好，返回正值，如果情绪中性，返回 0。

我们可以用这个来获得每条推文的情绪，如果情绪好，正面情绪加 1，如果情绪不好，负面情绪加 1，否则，如果情绪是中性的，中性情绪加 1。在对所有的推文运行这个之后，你将得到带有积极、消极和中性情绪的推文的总数。

你现在可以计算正面、负面和中性推文的百分比——这给出了认为这部电影好、坏和一般的观众的百分比。因此，你知道观众的感受。

我使用的这个方法是进行情感分析的一个非常基本的方法，还有很多其他的方法，包括训练序列法——我还在学习。虽然这种方法不是理解在线评论的最佳方式，但如果你分析更多的推文，比如说 10000 条推文，这就给出了他们感觉的近乎正确的图片。

感谢您的阅读！👍