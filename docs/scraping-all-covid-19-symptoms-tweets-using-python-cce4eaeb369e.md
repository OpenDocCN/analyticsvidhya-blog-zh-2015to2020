# 使用 Python 抓取所有新冠肺炎症状推文。

> 原文：<https://medium.com/analytics-vidhya/scraping-all-covid-19-symptoms-tweets-using-python-cce4eaeb369e?source=collection_archive---------13----------------------->

## [深入分析](https://towardsdatascience.com/in-depth-analysis/home)

## **大多数新冠肺炎患者一直在 twitter 上写他们的康复之路，因此我们在这里详细介绍如何收集所有信息并将其存储为 csv 文件，以供决策和进一步分析**

Twitter 是一座关于人们情绪的数据金矿。与其他社交媒体数据点和发布方式相比，在 twitter 上获取的信息更加结构化。Twitter 也提供了一种检索这些信息的简单方法。

本文主要解释了如何使用 Dmitry Mottl 的 GetOldTweets3 在 Python 中快速简单地从 Twitter 中抓取所有与 COVID 症状相关的推文。

![](img/6774d6c94f82ca78a65392dfd09ac56b.png)

来源:推特

如果你想直接跳到代码，你可以在我的 GitHub [这里](https://github.com/cheruiyot/Python_scrapetweets/blob/master/Get_Old_tweets_final.ipynb)访问这个教程的 Jupyter 笔记本。该代码是项目描述的最终支出

GetOldTweets3 由 Dmitry Mottl 创建，是 Jefferson Henrqiue 的 GetOldTweets-python 的改进分支。这个包允许你检索大量的推文和一周前的推文。
下图显示了可以检索到的与 tweet 相关的信息。

# 先决条件

使用 GetOldTweets3 不需要 twitter 的授权，但是你只需要 pip 安装这个库，然后你就可以马上开始了。您还需要 pandas 或 pyspark 来操作数据，这将在后面讨论。

为此，导入以下代码行:

```
# Pip install GetOldTweets3 if you don’t already have the package
# !pip install GetOldTweets3# Imports
import GetOldTweets3 as got
import pandas as pdimport findspark
findspark.init()
findspark.find()
import pyspark
findspark.find()#alwaysimport this for every pyspark analyticsfrom pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSessionconf = pyspark.SparkConf().setAppName(‘appName’).setMaster(‘local’)
#sc = pyspark.SparkContext(conf=conf)
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession(sc)
```

# 从文本搜索查询中抓取推文

为了查询所有的 COVID 症状，我主要关注三个项目:查询词、推文数量和位置

有了以上三个变量，我能够确保检索到主要热门城市(伦敦、纽约和巴黎)的新冠肺炎症状。

以下查询创建了一个包含巴黎附近所有 Tweets 的文本文件，并包含单词 COVID 症状:

```
text_query = ‘COVID symptoms’
count = 7000
geocode=”Paris”# Creation of query object
tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query).setMaxTweets(count).setNear(geocode)# Creation of list that contains all tweets
tweets = got.manager.TweetManager.getTweets(tweetCriteria)# Creating list of chosen tweet data
 text_tweets = [[tweet.date, tweet.text,tweet.id,tweet.username,tweet.geo] for tweet in tweets]
```

# 进一步操作 tweets 文件

为了在 python dataframe 中存储我们的数据，我们使用下面的代码片段:

```
tweets_df = pd.DataFrame(text_tweets, columns = ['Datetime', 'Text','TweetID','username','geo'])
```

如果你是熊猫专家，你可以直接将文件写入 csv。
对我来说，我喜欢使用 pyspark，以便利用 pyspark 的处理功能。
我将上面的熊猫数据帧转换成 pyspark，然后写入 csv 文件，如下所示:

```
tweets_df_spark.coalesce(1).write.save(path='C:\\Users\\brono\\First_batch\\Finalextract2.csv', format='csv', mode='append', inferSchema = True)
```

请注意，我将我的推文附加到现有的，以便运行我的推文，并附加到一个已经存在的文件。

# 删除重复的已清理推文文件

为了把这些放在一起，我添加了下面几行代码来删除重复抓取了两次的 tweets。
我还添加了最后两行代码来读取追加文件，并在写入新文件夹之前删除重复的文件。
请注意，每次运行代码时都会覆盖该文件中的数据

```
Finaldf = spark.read.csv("C:\\Users\\brono\\First_batch\\Finalextract2.csv", inferSchema = True, header = True)
Finaldf = Finaldf.dropDuplicates(subset=['TweetID'])

Finaldf.sort("TweetID").coalesce(1).write.mode("overwrite").option("header", "true").csv("C:\\Users\\brono\\First_batch\\Cleaned_data.csv")
```

# **将代码放在一起**

我将上述内容编译成一个函数，如下所示:

```
def text_query_to_csv(text_query, count):
    # Creation of query object
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query).setMaxTweets(count).setNear(geocode)
    #.setSince(newest_date1).setUntil(newest_date1)
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)# Creating list of chosen tweet data
    text_tweets = [[tweet.date, tweet.text,tweet.id,tweet.username,tweet.geo] for tweet in tweets]# Creation of dataframe from tweets
    tweets_df = pd.DataFrame(text_tweets, columns = ['Datetime', 'Text','TweetID','username','geo'])

    # Createspark spark dataframe
    tweets_df_spark = spark.createDataFrame(tweets_df)# Converting tweets dataframe to csv file

    tweets_df_spark.coalesce(1).write.save(path='C:\\Users\\brono\\First_batch\\Finalextract2.csv', format='csv', mode='append', inferSchema = True)
    # Read the appended file and remove duplicates and write the clean as one file
    Finaldf = spark.read.csv("C:\\Users\\brono\\First_batch\\Finalextract2.csv", inferSchema = True, header = True)
    Finaldf = Finaldf.dropDuplicates(subset=['TweetID'])

    Finaldf.sort("TweetID").coalesce(1).write.mode("overwrite").option("header", "true").csv("C:\\Users\\brono\\First_batch\\Cleaned_data.csv")
```

最终代码发布在 Github[https://gist . Github . com/cheruiyot/369 e5d 99489 ef 55558 ce 1 D5 df 2087 c 64](https://gist.github.com/cheruiyot/369e5d99489ef55558ce1d5df2087c64)

要访问已经收集的 9000 多条推文，你可以联系