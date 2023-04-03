# 使用 twitterscraper 库抓取推文

> 原文：<https://medium.com/analytics-vidhya/twitter-scraper-using-twitterscraper-library-dd76802a8ddf?source=collection_archive---------2----------------------->

在这里，我们将学习如何使用 twitter-scrapper 库来抓取特定主题的推文

首先，我们需要使用下面提到的命令安装 twitterscraper 库。

```
!pip install twitterscraper
```

现在安装后，检查它是否是当前版本。当我写这篇博客的时候，我用的是最新的 1.4.0 版本。

现在我们需要导入三样东西，它们是 query_tweets、datetime 和 pandas。

```
fromtwitterscraperimport query_tweets
import datetime as dt
import pandas as pd
```

现在，我们将提到想要抓取推文的时间跨度。然后，我们将提到我们想要多少条推文(这里我们没有得到完全相同的推文数量，但大致如此)。之后，我们将提到我们需要哪种语言的推文。Twitter 支持 34 种不同类型的语言。

```
begin_date=dt.date(2020,4,14)
end_date=dt.date(2020,5,14)
limit=1000
lang='english'
```

如果我们不提及 begin_date 和 end_date，那么它将抓取所有的推文。如果我们没有提到任何限制，那么它将会抓取所有的推文。如果我们不提及 lang，它将收集所有语言的推文。

现在我们将通过 **query_tweets( )** 函数收集推文。在这里，我们将第一个参数作为我们想要收集 tweets 的标签或主题发送，之后我们将所有上述参数作为参数。

```
tweets=query_tweets("#lockdown AND #India ",begindate=begin_date,enddate=end_date,limit=limit,lang=lang)
```

现在，从 2020 年 3 月 14 日到 2020 年 4 月 14 日，推文将包含大约 1000 条关于一级防范禁闭和印度的英文推文。

但是这里 tweets 是一个对象，所以我们必须使用下面提到的代码将它转换成熊猫数据帧。

```
df=pd.DataFrame(t.__dict__ **for** t **in** tweets)
```

现在，我们可以轻松地可视化我们收集的所有推文。这个 **df** 数据帧将有 22 列或特征。那些特征分别是**未命名:0** 、**有 _ 媒体**、**标签**、 **img_urls** 、**被 _ 回复**、**被 _ 回复**、**点赞**、**链接**、 **parent_tweet_id** 、**回复**、**回复 _ 回复**

这个 twitterscraper 的优点是我们可以抓取任何时间段的推文，而如果我们使用 twitter API，那么我们最多只能抓取过去 15 天的数据，不能超过 15 天。