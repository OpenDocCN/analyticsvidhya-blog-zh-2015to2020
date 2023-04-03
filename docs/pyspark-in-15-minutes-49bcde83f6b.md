# 程序员对 PySpark 过于简单的介绍

> 原文：<https://medium.com/analytics-vidhya/pyspark-in-15-minutes-49bcde83f6b?source=collection_archive---------12----------------------->

![](img/b0756d6e4c9ad1631f7aa303d886de92.png)

# 什么是阿帕奇火花？

Apache Spark 是一个快速的通用集群计算系统。它提供了 Java、Scala、Python 和 R 的高级 API，以及支持通用执行图的优化引擎。它还支持一套丰富的高级工具，包括用于 SQL 和结构化数据处理的 [Spark SQL](https://spark.apache.org/docs/latest/sql-programming-guide.html) ，用于机器学习的 [MLlib](https://spark.apache.org/docs/latest/ml-guide.html) ，用于图形处理的 [GraphX](https://spark.apache.org/docs/latest/graphx-programming-guide.html) ，以及 [Spark Streaming](https://spark.apache.org/docs/latest/streaming-programming-guide.html) 。

这是技术领域最热门的新趋势之一。它运行速度快(由于内存操作，比传统的 [Hadoop MapReduce](https://www.tutorialspoint.com/hadoop/hadoop_mapreduce.htm) 快 100 倍)，提供健壮的分布式容错数据对象(称为 [RDD](https://www.tutorialspoint.com/apache_spark/apache_spark_rdd.htm) )，并与机器学习和图形分析的世界完美集成。

Spark 在 [Hadoop/HDFS](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html) 上实现，大部分用 [Scala](https://www.scala-lang.org/) 编写，一种类似 Java 的函数式编程语言。幸运的是，Spark 提供了一个奇妙的 Python 集成，称为 **PySpark** ，它允许 Python 程序员与 Spark 框架进行交互，并学习如何大规模操作数据，以及如何在分布式文件系统上处理对象和算法。Spark 以简单接口库的形式支持多种编程语言:Java、Python、Scala 和 r。

需要记住的一点是，Spark 不是像 Python 或 Java 那样的编程语言。它是一个通用的分布式数据处理引擎，适用于各种环境。它对于大规模和高速的大数据处理特别有用。

PySpark 的下载和安装请参考[官方文档](https://spark.apache.org/docs/latest/)。

在本文中，我们将学习 PySpark 的基础知识。有很多概念(不断发展和引入)，因此，我们只关注简单例子的基本原理。

**本文必备:**

1.  基本 Python 语法
2.  熟悉[λ函数](https://python-reference.readthedocs.io/en/latest/docs/operators/lambda.html)

# 弹性分布式数据集(RDD)

许多 Spark 程序都围绕着弹性分布式数据集(RDD)的概念，这是一个可以并行操作的容错元素集合。SparkContext 驻留在驱动程序中，通过集群管理器管理工作节点上的分布式数据。使用 PySpark 的好处是，所有这些复杂的数据分区和任务管理都在后台自动处理，程序员可以专注于特定的分析或机器学习工作本身。

创建 rdd 有两种方法——在驱动程序中并行化现有集合，或者引用外部存储系统中的数据集，例如共享文件系统、HDFS、HBase 或任何提供 Hadoop InputFormat 的数据源。

为了说明基于 Python 的方法，我们将在这里给出第一种类型的例子。我们可以使用 Numpy random.randint()创建一个包含 20 个随机整数(0 到 10 之间)的简单 Python 数组，然后创建一个 RDD 对象，如下所示:

```
from pyspark import SparkContext
import numpy as np
sc=SparkContext(master="local[4]")
lst=np.random.randint(0,10,20)
A=sc.parallelize(lst)
```

***注意论点*中的‘4’。它表示 4 个计算核心(在您的本地机器中)将用于这个 SparkContext 对象**。如果我们检查 RDD 对象的类型，我们得到如下结果，

```
type(A)
>> pyspark.rdd.RDD
```

与并行化相对的是集合(使用 collect())，它将所有分布的元素集合起来并返回给头节点。

```
A.collect()
>> [5, 1, 1, 6, 9, 4, 2, 2, 3, 2, 2, 7, 7, 7, 8, 6, 9, 9, 6, 1]
```

但是 A 不再是简单的 Numpy 数组。我们可以使用 glom()方法来检查分区是如何创建的。

```
A.glom().collect()
>> [[5, 1, 1, 6, 9], [4, 2, 2, 3, 2], [2, 7, 7, 7, 8], [6, 9, 9, 6, 1]]
```

现在停止 SC，用 2 个内核重新初始化它，看看重复这个过程会发生什么。

```
sc.stop()
sc=SparkContext(master=”local[2]”)
A = sc.parallelize(lst)
A.glom().collect()
>> [[5, 1, 1, 6, 9, 4, 2, 2, 3, 2], [2, 7, 7, 7, 8, 6, 9, 9, 6, 1]]
```

RDD 现在分布在两个块上，而不是四个！

您已经了解了分布式数据分析的第一步，即控制如何将数据划分为更小的数据块以供进一步处理。

# 深入研究代码

## 数数元素

让我们数一下元素的数量。

```
A.count()
>> 20
```

## “first()”和“take()”运算

记住， **take()** 总是带一个参数。

```
A.first()   #Gives the first element
>> 5A.take(4)   #Gives the first few elements
>> [5, 1, 1, 6]
```

## 使用“distinct()”删除重复项

**注意**:这个操作需要一个**洗牌**来检测跨分区的重复。所以，这是一个缓慢的操作。不要过度。

```
A.distinct().collect()
>> [6, 4, 2, 8, 5, 1, 9, 3, 7]
```

## 将所有元素相加

这里，我们将通过两种方式找到总和:一种使用 **sum()** ，另一种使用 **reduce()** 。注意在后者中使用了[λ函数](https://python-reference.readthedocs.io/en/latest/docs/operators/lambda.html)。

```
A.sum()
>> 97A.reduce(lambda x,y:x+y)
>> 97
```

## 使用“reduce()”查找最大元素

使用 lambda 函数，查找最大值元素。

```
A.reduce(lambda x,y: x if x > y else y)
>> 9
```

## 基本统计

在我们的 RDD 上应用基本统计函数。

```
print("Maximum: ",A.max())
print("Minimum: ",A.min())
print("Mean (average): ",A.mean())
print("Standard deviation: ",A.stdev())>> Maximum:  9
>> Minimum:  1
>> Mean (average):  4.850000000000001
>> Standard deviation:  2.8332843133014376
```

## 在文本块中查找最长的单词

看，我们是如何使用 **reduce()** 让我们的生活变得简单。

```
words = ‘An Oversimplified Introduction to PySpark for Programmers’.split(‘ ‘)
wordRDD = sc.parallelize(words)
wordRDD.reduce(lambda w,v: w if len(w)>len(v) else v)
>> ‘Oversimplified’
```

## 映射操作

**map()** 通过对 RDD 的每个元素应用一个函数来返回一个新的 RDD。

```
B=A.**map**(lambda x:x*x)
B.**collect**()
>> [25, 1, 1, 36, 81, 16, 4, 4, 9, 4, 4, 49, 49, 49, 64, 36, 81, 81, 36, 1]
```

## 使用常规 Python 函数进行映射

以下函数返回奇数元素的平方，并保持偶数参数不变。

```
def square_if_odd(x):
    if x%2==1:
        return x*x
    else:
        return x
A.map(square_if_odd).collect()
>> [25, 1, 1, 6, 81, 4, 2, 2, 9, 2, 2, 49, 49, 49, 8, 6, 81, 81, 6, 1]
```

# 惰性评估(和缓存)

惰性评估是一种评估/计算策略，它为计算任务准备了详细的执行管道的逐步内部图，但将最终执行延迟到绝对需要的时候。这一策略是 Spark 加速许多并行化大数据操作的核心。

让我们在这个例子中使用两个 CPU 内核，

```
sc = SparkContext(master=”local[2]”)
```

用一百万个元素做一个 RDD，

```
%%time
rdd1 = sc.parallelize(range(1000000))
>> CPU times: user 316 µs, sys: 5.13 ms, total: 5.45 ms, Wall time: 24.6 ms
```

现在，创建一个 Python 函数，比如“taketime”，

```
from math import cos
def taketime(x):
    [cos(j) for j in range(100)]
    return cos(x)
```

检查“花费时间”功能花费了多少时间

```
%%time
taketime(2)
>> CPU times: user 21 µs, sys: 7 µs, total: 28 µs, Wall time: 31.5 µs
>> -0.4161468365471424
```

记住这个结果，taketime()函数用了 31.5 us 的墙时间。当然，确切的数字将取决于您正在使用的机器。

现在，对函数执行映射操作，

```
%%time
interim = rdd1.map(lambda x: taketime(x))
>> CPU times: user 23 µs, sys: 8 µs, total: 31 µs, Wall time: 34.8 µs
```

为什么每个 taketime 函数需要 34.8 us，但是一百万元素 RDD 的地图操作也需要类似的时间？

**因为懒评估，即上一步什么都没计算，只做了一个执行计划**。变量 *interim* 并不指向数据结构，相反，它指向一个执行计划，用依赖图表示。依赖图定义了 rdd 如何相互计算。

reduce()方法的实际执行，

```
%%time
print(‘output =’,interim.reduce(lambda x,y:x+y))
>> output = -0.28870546796843666
>> CPU times: user 11.6 ms, sys: 5.56 ms, total: 17.2 ms, Wall time: 15.6 s
```

所以，这里的墙时间是 15.6 秒。还记得吗，taketime()函数的墙时间是 31.5 us？因此，对于一百万个阵列，我们预计总时间约为 31 秒。因为在两个内核上并行操作，所以花费了大约 15 秒。

现在，我们在过渡期间没有保存(物化)任何中间结果，所以另一个简单的操作(例如计数元素> 0)将花费几乎相同的时间。

```
%%time
print(interim.filter(lambda x:x>0).count())
>> 500000
>> CPU times: user 10.6 ms, sys: 8.55 ms, total: 19.2 ms, Wall time: 12.1 s
```

## 缓存以减少类似操作的计算时间(消耗内存)

还记得我们在上一步中构建的依赖图吗？我们可以像以前一样使用 cache 方法运行相同的计算，告诉依赖图规划缓存。

```
%%time
interim = rdd1.map(lambda x: taketime(x)).cache()
```

第一次计算不会改进，但是它缓存了中间结果，

```
%%time
print(‘output =’,interim.reduce(lambda x,y:x+y))
>> output = -0.28870546796843666
>> CPU times: user 16.4 ms, sys: 2.24 ms, total: 18.7 ms, Wall time: 15.3 s
```

现在在缓存结果的帮助下运行相同的过滤方法，

```
%%time
print(interim.filter(lambda x:x>0).count())
>> 500000
>> CPU times: user 14.2 ms, sys: 3.27 ms, total: 17.4 ms, Wall time: 811 ms
```

哇！计算时间从之前的 12 秒下降到不到 1 秒！这样，延迟执行的缓存和并行化是 Spark 编程的核心特性。

# 下一步做什么？

本文介绍了 Apache Spark 的基础知识，以及它如何使用 Python 接口 PySpark 实现核心数据结构 RDD 的一些基本示例。

关于 Apache Spark，还有很多东西需要学习和实验。PySpark 网站是一个很好的参考网站，他们会定期更新和改进，所以请密切关注。

而且，如果您对使用 Apache Spark 进行大规模分布式机器学习感兴趣，那么可以查看一下 [MLlib](https://spark.apache.org/mllib/) 。