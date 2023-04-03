# 使用 Python 提取 Twitter 数据

> 原文：<https://medium.com/analytics-vidhya/extracting-twitter-data-using-python-9627f1d3b77a?source=collection_archive---------15----------------------->

![](img/7f3b7de7cca9452da1d9616b98d3d97b.png)

文本分析，也称为文本挖掘，是一种从文本数据中提取价值、可操作信息和见解的途径和方法。它使用 NLP 和信息检索技术将非结构化数据转换为结构化形式，以便进一步用于分析，从这些数据中得出对最终用户有帮助的模式和见解。

它的一些应用有:
社交媒体分析
欺诈检测
垃圾邮件过滤

Twitter 是一个我们可以找到不同主题的大量数据的地方。这些数据可用于发现与特定关键词/标签相关的趋势，衡量品牌情感或收集关于新产品和服务的反馈。

在本帖中，我们将借助 Python 从 twitter 中提取数据

![](img/6b8bdf572530b2549ac6a6ae37cb897c.png)

**入门**

要开始，您需要做以下事情:

1.  使用您的 Twitter 帐户，您将需要申请开发人员访问权限，然后创建一个应用程序，该应用程序将生成用于从 Python 访问 Twitter 的 API 凭证。
2.  如果您还没有 Twitter 帐户，请创建一个。
3.  在[https://apps.twitter.com/](https://apps.twitter.com/)登录或创建一个 Twitter 账户。
4.  创建新应用程序(右上角的按钮)
5.  用唯一的**名称、网站名称(如果没有，请使用占位符网站)和项目描述填写应用程序创建页面。接受条款和条件并进入下一页。**
6.  项目创建完成后，单击“密钥和访问令牌”选项卡。您现在应该能够看到您的**消费者密钥**、**消费者密钥**、**访问令牌密钥**和**访问令牌密钥**。
7.  我们需要所有这些凭证，所以请确保打开此选项卡。

## **导入库**

```
from twython import Twythonfrom textblob import TextBlobimport pandas as pd
```

复制粘贴应用程序提供的凭证，并将其保存在以下变量中

```
consumer_key = “wXXXXXXXXXXXXXXXXXXXXXXX1”consumer_secret = “qXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXh”access_token = “9XXXXXXXX-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXi”access_token_secret = “kXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXT”
```

现在是时候使用我们上面定义的凭证来创建我们的 Twython 实例了

```
twitter = Twython(ConsumerKey, ConsumerSecret, AccessToken, AccessSecret)
```

## 主页时间线

这将返回认证用户和您关注的用户发布的最新 Tweets 的集合

```
home_timeline = twitter.get_home_timeline()
print(home_timeline)
```

## 用户时间线

这将返回你最近发布的推文的集合

```
my_timeline = twitter.get_user_timeline()
```

下面的代码将只打印你的推文文本

```
for tweet in my_timeline: print(tweet['text'])
```

## 将推文转换成数据帧

这将把我们的推文文本转换成数据帧，然后我们可以使用它进行分析

```
home_df = pd.DataFrame(home_timeline)
home_df.to_excel('Home_timeline.xlsx', index=False)print(home_df[['id','text']])
```

## 搜索推文

这将显示所有基于标签的推文，并返回所有使用这些关键词的推文

```
search = twitter.search(q="#covid #pakistan", count=10, tweet_mode="extended") 
#supply whatever keyword data you want in variable q
#count represent number of tweets we want
```

## 按用户名发布推文

这将显示您在 screen_name 参数中输入的用户的所有 tweets

```
twitter.show_user(screen_name="ImranKhanPTI")
```

现在，我们将转换指定用户名发布的所有推文，并将其保存在数据框中

```
user_timeline = twitter.get_user_timeline(screen_name="ImranKhanPTI",count=10, tweet_mode=’extended’)usr_timeline = pd.DataFrame(user_timeline)
usr_timeline.to_excel('MyTimelineData.xlsx', index=False)print(usr_timeline['full_text'])
```

> *确保使用“全文”,因为这个键包含整个 tweet 文本。*

将 tweets 数据及其在 twitter 上发布的日期保存到一个列表中

```
alltweets = []for status in user_timeline: newtweet = (status['full_text'], status['created_at']) alltweets.append(newtweet)print(alltweets)
```

**使用 TextBlob 库将推文转换成另一种语言**

```
for i, tweet in enumerate(alltweets): (txt, time_date) = tweet twt_blob = TextBlob(txt) lang = twt_blob.detect_language() print("\n — — ", str(i), " ", lang, " ", time_date, " —-") if (lang == ‘ur’): print(txt) print(twt_blob.translate(to=’en’))
```

# 结论

Twitter 的 API 在数据挖掘应用中非常有用，可以提供对公众意见的大量洞察。我鼓励你阅读更多关于 [Twython](https://twython.readthedocs.io/en/latest/) 和其他相关的图书馆，例如 [Tweepy](http://docs.tweepy.org/en/v3.5.0/api.html)

在这篇文章中，我们介绍了访问 twitter 数据并将其保存在电子表格中的基础知识，以及如何使用其他有用的功能从 Twitter 中获得更多见解。

> *在我的博客和 YouTube 频道查看更多文章* ***http://uzairadamjee.com/blog*** [***https://www.youtube.com/channel/UCCxSpt0KMn17sMn8bQxWZXA***](https://www.youtube.com/channel/UCCxSpt0KMn17sMn8bQxWZXA)

我们的教程到此结束。感谢您的阅读:)