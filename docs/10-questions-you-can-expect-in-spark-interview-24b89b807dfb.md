# 在 Spark 面试中你会遇到的 10 个问题

> 原文：<https://medium.com/analytics-vidhya/10-questions-you-can-expect-in-spark-interview-24b89b807dfb?source=collection_archive---------1----------------------->

## Apache Spark 访谈中常见的问题

![](img/ada7e8653423b9f2a675128583e305e4.png)

嘿伙计们，

随着 Apache Spark 成为批处理和 ETL 的尖端技术，数据工程师的职位在最近非常吃香，了解它可以很容易地让你找到一份数据工程师的工作。因此，在本文中，我将展示 Apache Spark 访谈中的 10 个问题，请注意，我不会包括诸如“什么是 Dataframe？”，“什么是火花 RDD？”或者“如何读/写 orc 文件？”正如我所料，一个去 Apache Spark 面试的同事应该已经知道这些事情，再重复这些事情是没有意义的。

所以这些都说了，还是跳到 Q/A 吧。

## Spark 比 Hadoop 好吗？为什么？

是的，Spark 明显优于 Hadoop，主要原因之一是它比 Hadoop 更快，因为内存处理有助于减少读/写操作的延迟。基本上，当我们使用 map reduce 范式时，在完成每个任务时，都会在磁盘上写入数据，当需要再次使用数据时，会再次执行读取操作。但是，在 Spark 中，处理将在内存中完成，数据帧将被缓存以备将来使用，从而提高性能。此外，Spark 附带了 Spark ML、Spark SQL、Spark Streaming 等库，这使它更加丰富。

## 联合和再分配的区别是什么？

当谈到优化你的 spark 工作时，这是讨论的热门话题。这两个函数基本上都允许我们操纵数据帧的分区数量，但是它们的用途是不同的。重新分区将对数据进行完全洗牌，因此我们可以增加或减少分区的数量，但联合只会将数据从一个分区转移到另一个分区，从而只会减少使用它的分区的数量。合并会更快，因为混洗会更少，但是如果分区的数量必须增加或者数据是倾斜的，并且我们希望通过重新混洗来减少分区的数量，那么我们应该使用重新分区方法。

## 什么是广播加入？

广播连接也用于优化 Spark 作业(尤其是连接)。当小数据帧与相对较大的数据帧连接时，我们可以广播小数据帧，这将向每个节点发送小数据帧的副本，这将导致更快的连接执行和更少的洗牌。下面给出了语法。

```
import pyspark.sql.functions as fn
final = big.join(fn.broadcast(small),["common_id"])
```

当广播较小的数据帧时，我们可以将其分区减少到 1 以获得更好的性能(取决于您的使用情况)。

## 什么是懒评？

Apache Spark 有两个重要的方面，一是行动，二是转化。Transformation 包括 filter、where、when 等函数，调用这些函数时，Spark 并不实际执行这些转换，而是堆叠起来，直到调用一个动作。当一个动作被调用时，所有的转换都在此时执行，这有助于 Apache Spark 优化作业的性能。操作示例有 show()、count()、collect()。

## cache()和 persist()有什么区别？

这两个 Api 都用于在不同级别的内存中持久存储数据帧，但在持久存储中，我们可以将存储级别指定为 MEMORY_ONLY、MEMORY_AND_DISK、DISK_ONLY 等，而在 cache()中，我们不能指定存储级别，默认情况下被视为 MEMORY_ONLY。

## 秩和密秩的区别？

这是一个 sql 问题，但我把它包括在内，因为如果我们进入窗口分区部分，我们会遇到这个问题。假设我们有一个如下所示的数据集:

```
Name  Salary  Rank  Dense_rank
Abid  1000     1      1
Ron   1500     2      2
Joy   1500     2      2
Aly   2000     4      3
Raj   3000     5      4
```

这里的薪水是递增的，我们得到的是数据集的 rank()和 dense_rank()。因为 Ron 和 Joy 有相同的薪水，所以他们得到相同的等级，但是 rank()会留下一个洞并保持“3”为空，而 dense_rank()会填充所有的空隙，即使遇到相同的值。

## 如何通过 Spark SQL 连接 Hive？

![](img/7a94bee7ad4068d4bf6337c32e02f64f.png)

对此的解决方案是将 hive-site.xml 和 core-site.xml 复制到 spark conf 文件夹中，这将为 spark job 提供有关 hive metastore 的所有必需元数据，您必须启用 Hive 支持，并在配置中指定 Hive 的仓库目录位置，同时启动 Spark 会话，如下所示:

```
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL Hive integration example") \
    .config("spark.sql.warehouse.dir", warehouse_location) \
    .enableHiveSupport() \
    .getOrCreate()
```

点击阅读关于此次访问的详细信息

## RDD 比 Dataframes 好吗？

不，数据帧在执行上比 rdd 更快，在语法上也更容易。你可能会怀疑数据帧在后端被转换成 RDD，那么 RDD 有多慢？答案是 Dataframe 使用 Catalyst Optimizer，这使它比 rdd 运行得更快，另一方面，rdd 在执行期间不使用任何优化器。这些 Api 之间的比较已经在 2017 Spark Summit 的[这个视频](https://www.youtube.com/watch?v=Ofk7G3GD9jk)中详细解释过了。

## 如何在 Spark 中读取一个 xml 文件？

这很简单，spark 有 [spark-xml](https://mvnrepository.com/artifact/com.databricks/spark-xml_2.10/0.2.0) 包，它允许我们将 xml 文件解析为数据帧，考虑下面给出的 xml 文件:

```
<person>
    <name>John</name>
    <address>Some more Data</address>
</person>
```

然后，为了读取，我们可以在读取期间指定模式和根标签，如下所示:

```
xmlDF = spark.read
      .format("com.databricks.spark.xml")
      .option("rootTag", "person")
      .option("rowTag", "name")
      .option("rowTag", "address")      
      .xml("yourFile.xml")
```

这将为您提供以“姓名”和“地址”为列的数据框架。

## 在纱线模式下运行 Spark 时，需要在纱线集群的所有节点上安装 Spark 吗？

不，当通过 YARN 模式提交作业时，没有必要在所有节点上安装 Spark，因为 Spark 运行在 YARN 之上，并使用 YARN 引擎来获得所有需要的资源，我们只需在一个节点上安装 Spark。点击阅读更多关于纱线部署模式[的信息。](https://spark.apache.org/docs/latest/running-on-yarn.html)

所以，这就是所有人希望你发现我的文章有帮助。一定要看看我以前关于 Spark Delta 的文章，其中我解释了 Spark 上的酸性物质。直到那时再见！

[](/analytics-vidhya/spark-delta-lake-d05dd480287a) [## 火花三角洲湖

### 嘿伙计们，

medium.com](/analytics-vidhya/spark-delta-lake-d05dd480287a)