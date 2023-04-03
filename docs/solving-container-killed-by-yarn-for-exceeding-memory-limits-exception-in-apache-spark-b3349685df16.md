# 解决 Apache Spark 中的“容器因超出内存限制而被 Yarn 杀死”异常

> 原文：<https://medium.com/analytics-vidhya/solving-container-killed-by-yarn-for-exceeding-memory-limits-exception-in-apache-spark-b3349685df16?source=collection_archive---------0----------------------->

![](img/8d2fe1b9ddc72224e363a4decaffc525.png)

**简介** [Apache Spark](https://spark.apache.org/) 是一个用于分布式大数据处理的开源框架。它最初是用 Scala 编写的，也有 Java、Python 和 R 编程语言的原生绑定。它还支持 SQL、流数据、机器学习和图形处理。

> *总而言之，Apache Spark 通常被称为用于大规模数据处理的* **统一分析引擎。**

如果您使用 Apache Spark 已经有一段时间了，您可能会遇到类似这样的异常:
*容器因超出内存限制而被 YARN 杀死，5 GB 中的 5GB 被使用*

原因可能在驱动程序节点上，也可能在执行器节点上。简而言之，异常表示，在处理时，spark 必须在内存中获取更多的数据，而实际上执行器/驱动程序已经拥有了这些数据。
这可能有几个原因，可以通过以下方式解决:

*   您的*数据是倾斜的*，这意味着您在处理过程中没有正确地对数据进行分区，从而导致需要为特定任务处理更多的数据。在这种情况下，您可以检查您的数据，并尝试使用一个*自定义分区器*对数据集进行统一分区。
*   您的 Spark 工作可能会在网络上传输大量数据。在可供执行器使用的内存中，只有一部分被分配给洗牌周期。尝试使用高效的 Spark API，如 *reduceByKey* 而不是 *groupByKey* 等，如果还没有这样做的话。有时候，洗牌是不可避免的。在这种情况下，我们需要增加内存配置，我们将在后面的内容中讨论

如果以上两点不适用，按顺序尝试以下*直到错误解决。在继续之前，恢复您可能对 spark 配置文件所做的任何更改。*

*   ***增加内存开销** 内存开销是分配给每个执行器的堆外内存量。默认情况下，内存开销被设置为执行器内存的 10%或 384 mb 之间的较高值。内存开销用于 Java NIO 直接缓冲区、线程堆栈、共享本机库或内存映射文件。
    上述异常可能发生在驱动程序或执行程序节点。无论错误在哪里，尝试逐渐增加 ***的开销内存，只存放*** ***(驱动程序或执行程序)*** 并重新运行作业。建议的最大内存开销是执行程序内存的 25%
    ***注意*** *:确保驱动程序或执行程序内存加上驱动程序或执行程序内存开销的总和始终小于****yarn . nodemanager . resource . memory-MB***
    的值，即 `spark.driver/executor.memory + spark.driver/executor.memoryOverhead < yarn.nodemanager.resource.memory-mb`
    您必须通过编辑 *spark-defaults.conf* 文件来更改该属性*

```
*sudo vim /etc/spark/conf/spark-defaults.conf spark.driver.memoryOverhead 1024 
spark.executor.memoryOverhead 1024*
```

*您可以在群集范围内为所有作业指定上述属性，也可以将其作为单个作业的配置进行传递，如下所示*

```
*spark-submit --class org.apache.spark.examples.WordCount --master yarn --deploy-mode cluster --conf spark.driver.memoryOverhead=512 --conf spark.executor.memoryOverhead=512 <path/to/jar>* 
```

*如果这不能解决您的问题，请尝试下一点*

*   ***减少执行核心的数量** 如果执行核心的数量增加，所需的内存量也会增加。因此，尝试减少每个执行器的内核数量，这样可以减少可以在执行器上运行的任务数量，从而减少所需的内存。同样，根据错误所在更改驱动程序或执行器的配置。*

```
*sudo vim /etc/spark/conf/spark-defaults.conf 
spark.driver.cores  3 
spark.executor.cores  3*
```

*与上一点类似，您可以在群集范围内为所有作业指定上述属性，也可以将其作为单个作业的配置进行传递，如下所示:*

```
*spark-submit --class org.apache.spark.examples.WordCount --master yarn --deploy-mode cluster **--executor-cores 5--driver-cores 4** <path/to/jar>*
```

*如果这个不行，看下一点*

*   ***增加分区数量** 如果有更多的分区，每个分区所需的内存量就会更少。内存使用可以由 Ganglia 监控。您可以通过调用*来增加分区的数量。在 RDD 或数据帧上重新分区(< num_partitions > )**

*还没找到吗？增加执行器或驱动程序内存。*

*   ***增加驱动程序或执行程序的内存** 根据错误发生的位置，增加驱动程序或执行程序的内存
    ***注意:*** `spark.driver/executor.memory + spark.driver/executor.memoryOverhead < yarn.nodemanager.resource.memory-mb`*

```
*sudo vim /etc/spark/conf/spark-defaults.conf  
spark.executor.memory  2g 
spark.driver.memory  1g*
```

*就像其他属性一样，这也可以被每个作业覆盖*

```
*spark-submit --class org.apache.spark.examples.WordCount --master yarn --deploy-mode cluster --executor-memory 2g --driver-memory 1g <path/to/jar>*
```

*现在，您很可能已经解决了这个异常。*

*如果没有，那么您的集群可能需要更多的内存优化实例！*

*编码快乐！
参考:[https://AWS . Amazon . com/premium support/knowledge-center/EMR-spark-yarn-memory-limit/](https://aws.amazon.com/premiumsupport/knowledge-center/emr-spark-yarn-memory-limit/)*