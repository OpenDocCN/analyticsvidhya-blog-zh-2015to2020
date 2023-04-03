# 借助 Apache Spark 激发您的分析能力🌟

> 原文：<https://medium.com/analytics-vidhya/spark-your-analytics-with-apache-spark-f4a47936b2f0?source=collection_archive---------8----------------------->

在深入到一个简单的用例之前，让我们先了解一下 Apache Spark 的定义。
**Apache Spark:** Spark 是一个内存优化的数据处理引擎，可以对 Hadoop 数据(在 HDFS)执行操作，性能优于 MapReduce。

这意味着，Apache Spark 提供了以分布式方式运行单个作业的能力，其中，范围从两个节点到 1000 个节点的节点群集(为简单起见，是一组计算机)被组装起来在内存中执行一项作业，并且具有足够的容错能力来跟踪任何丢失的节点，并将其任务(职责)分配给群集中任何其他可用的节点，最终所有节点都向驱动程序节点(领导节点)报告作业的结果。

![](img/c37dbd8685e99527ff8eb3817a1b2885.png)

图片来源:[https://unsplash.com](https://unsplash.com/)

一个简单的用例是假设用户想要找到他的商店中销量最大的杂货项目，该商店在 2010 年至 2011 年期间拥有大约 10，000 名客户，用户只需运行一个提供该项目的详细信息的 SQL 查询。
现在，同一用户希望确定 2000 年至 2019 年期间每位顾客购买最多的杂货，假设销售额很高，并且在某一天至少有 3000 笔销售额。这为我们提供了每个客户的销售记录计数(2000–2019)总计 _ 天数*总计 _ 销售数量= 9、***、***、***..或者可能更多。在这个用例上运行传统的 SQL 查询不是一个好主意。
Apache Spark 开始行动，拯救我们的创造力。

**示例用例:**
让我们考虑一个简单的 CSV(逗号分隔值)文件，它包含 4 列，大约有 9997851 条记录。
userId，movie，rating，timestamp
18295，909090，2.5，1515203646000
在上述数据的帮助下，可以导出 n 个列，这有助于开发者和/或数据分析师基于用户行为做出建设性的决策。
对于这个场景，我在装有 **IntelliJ IDE** 并安装了**Spark 的本地机器(笔记本电脑)上运行 Spark(我将在下一篇文章中讨论如何设置 IntelliJ 和 Spark)**。如果您愿意的话，也可以在一个 100 倍节点的集群上使用 spark session-cluster_level 设置运行。

**计划对数据执行任何类型的分析时要问的问题？
-数据质量如何？
-数据中有多少空值或空字符串？
-这些数据是否包含足够的用户标识或产品标识，以便做出决策？
-是否需要格式标准化？比如日期格式、User_id 格式或替换空值等……
-我可以基于任何列值(比如日期****-**-01、日期****-**-02)对数据进行分区以提高性能吗？**

在这个故事中，让我们基于单个 User_Id 导出下面的列。

```
user_id — The User ID.
mov_watch_count — No. of Movies watched by that User. 
latest_movie_rated — The Movie_id which User recently rated.
first_movie_rated — The Movie_id which User First rated.
more_than_six_rtngs — Does the user rated more than 6 movies? Y/N
less_than_two_rtngs — Does the user rated less than 2 movies? Y/N
highest_rating — The Highest rating given by the User ever.
lowest_rating — The Lowest rating given by the User ever.
```

让我们快速浏览一下如何实现这一点的代码片段，下面将包含重要的。

创建一个 spark 会话，它是 Spark 库世界的入口。

```
**//Create your sparkSession to Light it up.**
val spark = SparkSession
  .*builder*()
  .appName("userIdAndRatings")
  .master("local[*]")
  .config("spark.sql.warehouse.dir", "file:///C:/temp")
  .getOrCreate()
**//Read the CSV File(There are multiple ways to do this.)**
val rawRecords = spark.read.format("csv")
  .option("header", "true")
  .option("delimiter", ",")
  .option("inferSchema", "true")
  .load("C:/sparktempfiles/training.csv")
rawRecords.printSchema()
rawRecords.createOrReplaceTempView("raw_records_view")
**//Fix the Time Column from Epoch time to Human readable Date**
val rawRecordsTimeFix =
  """
    |select userId, movieId, rating, date_format(from_unixtime(timestamp/1000), 'yyyy-MM-dd') as dateOfRating
    |from raw_records_view
  """.stripMargin
```

想出最终表格或数据帧或 CSV 头文件的布局。

```
**//Create LayOut for your Output Table** val *userRatingsLayout* = *StructType*(
  StructField("user_id", IntegerType, true) ::
    StructField("mov_watch_count", IntegerType, true) ::
    StructField("latest_movie_rated", IntegerType, true) ::
    StructField("first_movie_rated", IntegerType, true) ::
    StructField("more_than_six_rtngs", StringType, true) ::
    StructField("less_than_two_rtngs", StringType, true) ::
    StructField("highest_rating", DoubleType, true) ::
    StructField("lowest_rating", DoubleType, true) ::
    *Nil*)
```

创建一个原始记录对象，然后对 user_id 进行分组，并将其转换为一个列表，为每个用户派生新列。

```
**//Construct DataFrame which can shown on Console, can be saved as CSV file or as a Table.**
val rawRecordsConstruct = spark.sql(rawRecordsTimeFix).map(RawRatings.*parse*(_)).groupByKey(_.user_id).mapGroups {
  case (user_id, ratings_iter) => {
    val ratings_List = ratings_iter.toList
val ratings_attr_vals = *getRatingsAttributes*(ratings_List) *Row*(user_id, ratings_attr_vals.mov_watch_count, ratings_attr_vals.latest_movie_rated, ratings_attr_vals.first_movie_rated, ratings_attr_vals.more_than_six_rtngs, ratings_attr_vals.less_than_two_rtngs, ratings_attr_vals.highest_rating, ratings_attr_vals.lowest_rating)
  }
}(*userRatingsColumnsEncoder*)
```

创建一个为我们的表布局计算列值的方法。

```
def getRatingsAttributes(rawList: List[RawRecords]): ratingsRec = {
 **//Create variables which has to be returned.** var user_id = 999999
  var mov_watch_count = 999999
  var latest_movie_rated = 999999
  var first_movie_rated = 999999
  var more_than_six_rtngs = "NA"
  var less_than_two_rtngs = "NA"
  var highest_rating: Double = 999999.0
  var lowest_rating: Double = 999999.0 user_id = rawList.map(x => x.user_id).head.toInt
  mov_watch_count = rawList.map(x => x.movie_id).size
  latest_movie_rated = rawList.sortWith(_.date_of_rating > _.date_of_rating).map(x => x.movie_id).head
  first_movie_rated = rawList.sortWith(_.date_of_rating < _.date_of_rating).map(x => x.movie_id).head
  more_than_six_rtngs = if (rawList.map(x => x.rating).size > 6) "Y" else "NA"
  less_than_two_rtngs = if (rawList.map(x => x.rating).size < 2) "Y" else "NA"
  highest_rating = rawList.map(x => x.rating).sortWith(_ > _).head
  lowest_rating = rawList.map(x => x.rating).sortWith(_ < _).head *ratingsRec*(user_id, mov_watch_count, latest_movie_rated, first_movie_rated, more_than_six_rtngs, less_than_two_rtngs, highest_rating, lowest_rating)
}***//Show the Results on IntelliJ Run Window.*** *println*("Showing the Results now")
rawRecordsConstruct.show(false)
```

是时候点击 IntelliJ 上绿色按钮(或)运行程序来查看输出结果了。Spark 开始工作，不到一分钟就返回结果。

**注意**:上面的 User_id 和 movie_id 都是重构的。

好了，你认为我们还可以在这里添加哪些列，比如给定用户的**一年内没有电影被观看，一个月内没有电影被观看，**等等..
如果我们可以用**‘movie_id’s and Movie _ Name’**加入 Movie _ Id 的 back，并根据用户评级找到最受欢迎的电影或向用户推荐电影，我们将很快看到这些结果。

欢迎任何建议。
找到我的 GitHub 存储库页面，其中包含了本文使用的全部代码:[https://github.com/pavanobj/spark_series](https://github.com/pavanobj/spark_series)
网上有很多免费的数据集，可以用来进行各种分析。

感谢您花时间阅读本文，下次再见😃