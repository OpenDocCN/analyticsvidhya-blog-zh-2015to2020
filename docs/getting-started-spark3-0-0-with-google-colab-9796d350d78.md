# 在 Google Colab 中开始使用 Spark 3.0.0

> 原文：<https://medium.com/analytics-vidhya/getting-started-spark3-0-0-with-google-colab-9796d350d78?source=collection_archive---------3----------------------->

![](img/5a0f8a89558855a9b5297cf75cb02721.png)

由马拉扎设计，使用了 www.canva.com 的[和谷歌图片中的图片](http://www.canva.com)

A pache Spark 是一个闪电般快速的集群计算系统，解决了以前最受欢迎的 Map Reduce 系统对大型数据集的限制。它是数据科学家和机器学习工程师处理大数据问题的首选框架。Spark engine 是用 Scala 编写的，Scala 被认为是可伸缩计算的首选语言。然而，Apache Spark 提供了 Java、Scala、Python 和 r 的高级 API。作为一名数据科学家`pyspark`是我利用 Spark 并行和分布式处理的首选 API。这可能是一种偏见，但是你可以选择适合你的应用程序的 API。

根据 [Apache Spark 新手入门](https://www.kdnuggets.com/2018/10/apache-spark-introduction-beginners.html)

> Spark 是 Hadoop 的子项目之一，由 Matei Zaharia 于 2009 年在加州大学伯克利分校的 AMPLab 创建。它于 2010 年在 BSD 许可下开源。它在 2013 年被交给了 Apache programming establishment，现在 Apache Spark 已经从 2014 年 2 月变成了最好的 Apache venture。现在结果非常好。

今年，spark 将庆祝作为一个开源项目的 10 周年纪念。在过去 10 年中，spark 成为利用并行和分布式计算框架进行大数据处理的事实上的选择。

在 2020 年 6 月 10 日通过投票后，Spark 3.0.0 于 2020 年 6 月 18 日发布。不过，Spark 3.0.0 的预览版是在 2019 年末发布的。

***Spark 3.0 大概比 Spark 2.4 快两倍。***

对于数据科学家和机器学习工程师来说，`pyspark and MLlib`是 Apache Spark 附带的两个最重要的模块。spark 3.0.0 中有大量与上述两个模块相关的新特性。阅读下面发布的说明，了解更多信息。

[基于发布的注释](https://spark.apache.org/releases/spark-release-3-0-0.html)

> Python 现在是 Spark 上使用最广泛的语言。PySpark 在 Python 包索引 PyPI 上的月下载量超过 500 万次。这个版本改进了它的功能和可用性，包括熊猫 UDF API 重新设计的 Python 类型提示，新的熊猫 UDF 类型，以及更多的 Python 错误处理。

对于对 JVM 操作系统了解有限的研究人员来说，设置 spark 通常被认为是一个复杂而耗时的步骤。在本文中，我将带您快速了解如何在 google colab 上安装 Apache Spark 3.0.0。

## 安装 Apache Spark 3.0.0

打开 google colab 笔记本，使用以下命令安装 Java 8，下载并解压 [Apache Spark 3.0.0](http://apache.osuosl.org/spark/spark-3.0.0/) 并安装`findpyspark`。根据您的连接速度，不会超过几分钟。

```
# Run below commands in google colab# install Java8
!apt-get install openjdk-8-jdk-headless -qq > /dev/null# download spark3.0.0
!wget -q http://apache.osuosl.org/spark/spark-3.0.0/spark-3.0.0-bin-hadoop3.2.tgz# unzip it
!tar xf spark-3.0.0-bin-hadoop3.2.tgz# install findspark 
!pip install -q findspark
```

## 设置环境变量

一旦执行了上面的命令，就该向环境添加相关的路径了。通过环境变量指向正确的版本，可以管理 spark 的多个版本。运行下面的命令集，指向之前下载的 Apache Spark 3.0.0 版本。

```
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.0-bin-hadoop3.2"
```

## 快速安装测试

现在是时候测试我们的 spark 安装及其版本了。我们应该可以使用 Spark 3.0.0 版本和带有 3.0.0 版本的`pyspark` 。

```
import findspark
findspark.init()from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
# Test the spark
df = spark.createDataFrame([{"hello": "world"} for x in range(1000)])
df.show(3, False)
```

要检查`pyspark`版本，使用下面的一组命令。强烈建议在运行应用程序时始终记录版本。

```
# Check the pyspark version
import pyspark
print(pyspark.__version__)
```

## 工作 Google Colab

我还创建了一个工作谷歌 colab，可以在下面找到。

作者:M.A .拉扎

## 结论

在这篇简短的文章中，我们学习了如何在不到两分钟的时间内设置 Spark 3.0.0。

## 参考资料/阅读/链接

1.  [http://apache.osuosl.org/spark/spark-3.0.0-preview2/](http://apache.osuosl.org/spark/spark-3.0.0-preview2/)
2.  [https://medium . com/@ sushantgautam _ 930/Apache-spark-in-Google-collaboratory-in-3-steps-E0 acbba 654 e 6](/@sushantgautam_930/apache-spark-in-google-collaboratory-in-3-steps-e0acbba654e6)
3.  [https://notebooks . gesis . org/binder/jupyter/user/data bricks-koalas-kuv5 qckt/notebooks/docs/source/getting _ started/10min . ipynb](https://notebooks.gesis.org/binder/jupyter/user/databricks-koalas-kuv5qckt/notebooks/docs/source/getting_started/10min.ipynb)
4.  [https://medium . com/@ sushantgautam _ 930/Apache-spark-in-Google-collaboratory-in-3-steps-E0 acbba 654 e 6](/@sushantgautam_930/apache-spark-in-google-collaboratory-in-3-steps-e0acbba654e6)
5.  1[https://towards data science . com/introduction-to-Apache-spark-207 a 479 c 3001](https://towardsdatascience.com/introduction-to-apache-spark-207a479c3001)
6.  [https://spark.apache.org/](https://spark.apache.org/)
7.  [https://medium . com/@ amjadraza 24/spark-ifying-pandas-data bricks-koala-with-Google-colab-93028890 db5](/@amjadraza24/spark-ifying-pandas-databricks-koalas-with-google-colab-93028890db5)