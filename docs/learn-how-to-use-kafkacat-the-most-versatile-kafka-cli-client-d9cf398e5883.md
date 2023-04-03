# 了解如何使用 Kafkacat —功能最丰富的 Kafka CLI 客户端

> 原文：<https://medium.com/analytics-vidhya/learn-how-to-use-kafkacat-the-most-versatile-kafka-cli-client-d9cf398e5883?source=collection_archive---------18----------------------->

![](img/658e0ab555a61d2caba690ce698742ce.png)

Kafkacat 是一个非常棒的工具，今天我想向您展示它是多么容易使用，以及您可以用它做一些很酷的事情。

1.5.0 版中提供了下面介绍的所有功能。

> 寻找快速 Kafkacat 参考？下载 [Kafkacat 单页备忘单](https://codingharbour.com/kafkacat-cheatsheet/)

# 安装 Kafkacat

Kafkacat 可以从 Homebrew(最新版本)和一些 Linux 存储库中获得，但 Linux 存储库可能不包含最新版本。如果是这样的话，你可以随时从 [docker](https://hub.docker.com/r/edenhill/kafkacat/tags) 运行最新的 kafkacat。

# 基础知识

Kafkacat 是一个命令行工具，用于生成和使用 Kafka 消息。此外，您可以查看关于集群或主题的元数据。

Kafkacat 有相当多的参数，学习它们可能看起来很可怕，但是(大多数)参数是有意义的，并且容易记住。先说最重要的:**模式**。当调用 Kafkacat 时，您将总是在它拥有的四种模式之一中使用它。所有模式都使用大写字母:

下一个最重要的选项是 **b** roker list (-b)，之后通常是 **t** opic (-t)。

所以你几乎可以把你的命令写成一个故事。以下命令:

```
kafkacat -C -b localhost:9092 -t topic1 -o beginning
```

可以理解为:我想从 **b** roker localhost:9092 和 **t** opic topic1 开始使用 **o** ffset。

好了，现在我已经满怀希望地让您相信所有这些神秘的参数都是有意义的，让我们看看如何使用 Kafkacat 来完成一些常见的任务。

# 生产数据(-P)

我们需要什么才能产生数据？至少，你需要一个经纪人和一个你想写的主题。

**产生数值**

```
kafkacat -P -b localhost:9092 -t topic1
```

输入默认消息分隔符。键入您的消息，并用 Enter 键分隔它们。

**生成密钥和值**

如果你想用 key 产生消息，你需要指定 **K** ey 分隔符(-K)。让我们使用冒号来分隔输入中的键和消息:

```
kafkacat -P -b localhost:9092 -t topic1 -K : 
key3:message3 
key4:message4
```

请注意，该参数使用大写字母 k。

**生成带标题的消息**

如果您想在消息中添加标题，请使用-H 参数，以 key=value 格式添加标题:

```
kafkacat -P -b localhost:9092 \
-t topic1 \
-H appName=kafkacat -H appId=1 
```

如您所见，通过重复-H 标志添加了额外的头。注意，所有生成的消息都有两个用-H 标志指定的头。

**从文件中产生数据**

如果你想用文件产生数据，使用选项-l(比如:fi **l** e)…我说过*大多数参数*很容易记住:)假设我们有一个名为 data.txt 的文件，其中包含由冒号分隔的键-值对:

```
key1:message1 
key2:message2 
key3:message3
```

所以命令应该是:

```
kafkacat -P -b localhost:9092 -t topic1 -K: -l data.txt
```

压缩生成消息

使用(-z)参数可以指定消息压缩:

```
kafkacat -P -b localhost:9092 -t topic1 -z snappy
```

支持的值有:snappy、gzip 和 lz4。

# 消费数据(-C)

# 简单消费者

**消耗一个话题的所有消息**

```
kafkacat -C -b localhost:9092 -t topic1
```

请注意，与 kafka-console-consumer 不同，kafkacat 将默认从主题的开头开始使用消息。这种做法对我来说更有意义，但是 YMMV。

**消费 X 条消息**

您可以使用 **c** ount 参数(-c，小写)来控制使用多少条消息。

```
kafkacat -C -b localhost:9092 -t topic1 -c 5
```

# 从偏移量开始消耗

如果您想从特定的 **o** ffset 中读取数据，您可以使用-o 参数。偏移参数非常通用。您可以:

**从开头或结尾消费消息**

```
kafkacat -C -b localhost:9092 -t topic1 -o beginning
```

使用常量**开始**或**结束**来告诉 kafkacat 从哪里开始消耗。

**从给定偏移量开始消耗**

```
kafkacat -C -b localhost:9092 -t topic1 -o 123
```

对偏移使用绝对值，Kafkacat 将从给定的偏移开始消耗。如果不指定要使用的分区，Kafkacat 将使用给定偏移量的所有分区。

**消耗分区中的最后 X 条消息**

```
kafkacat -C -b localhost:9092 -t topic1 -o -10
```

我们通过使用负偏移值来实现这一点。

# 基于时间戳消费

可以使用-o s@start_timestamp 格式在以毫秒为单位的给定时间戳之后开始消耗。从技术上来说，这是基于偏移量的消耗，不同的是 kafkacat 根据提供的时间戳为您计算偏移量。

```
kafkacat -C -b localhost:9092 -t topic1 -o s@start_timestamp
```

当到达给定的时间戳时，您还可以使用以下命令停止消费:

```
kafkacat -C -b localhost:9092 -t topic1 -o e@end_timestamp
```

当您调试发生的错误时，这非常有用，您有错误的时间戳，但您想检查消息看起来如何。然后，结合开始和结束偏移量，您可以缩小搜索范围:

```
kafkacat -C -b localhost:9092 \
-t topic1 \
-o s@start_timestamp -o e@end_timestamp
```

# 格式化输出

默认情况下，Kafkacat 将只打印出消息有效负载(Kafka 记录的值)，但是您可以打印您感兴趣的任何内容。要定义自定义输出，请指定(-f)标志，如**f**format，后跟一个格式字符串。下面是一个示例，它打印一个包含消息的键和值的字符串:

```
kafkacat -C -b localhost:9092 \
-t topic1 \ 
-f 'Key is %k, and message payload is: %s \n'
```

%k 和%s 是格式字符串标记。输出可能是这样的:

```
Key is key3, and message payload is: message3 
Key is key4, and message payload is: message4
```

那么使用格式字符串可以打印出什么呢？

正如您在上面看到的，您也可以在格式字符串中使用换行符(\n \r)或制表符(\t)。

# 塞尔德斯

如果消息没有写成字符串，您需要使用(-s)参数为键和值配置一个合适的 **s** 德尔。

例如，如果键和值都是 32 位整数，则可以使用以下方式读取:

```
kafkacat -C -b localhost:9092 -t topic1 -s i
```

您可以使用以下命令分别为键和值指定 serde:

```
kafkacat -C -b localhost:9092 -t topic1 -s key=i -s value=s
```

您将在 kafkacat 帮助(kafkacat -h)中找到所有 serdes 的列表。

# Avro serde

Avro 消息有点特殊，因为它们需要一个模式注册表。但是 Kafkacat 也把你包括在内了。使用(-r)指定模式注册表 URL:

```
kafkacat -C -b localhost:9092 \
-t avro-topic \ 
-s key=s -s value=avro \ 
-r [http://localhost:8081](http://localhost:8081)
```

在上面的例子中，我们从一个主题中读取消息，其中键是字符串，但值是 Avro。

# 列表元数据(-L)

列出元数据为您提供有关主题的信息:它有多少个分区，哪个代理是分区的领导者，以及同步副本(isr)的列表。

**所有主题的元数据**

```
kafkacat -L -b localhost:9092
```

只需不带其他参数调用-L 就可以显示集群中所有主题的元数据。

**给定主题的元数据**

如果您只想查看一个主题的元数据，请使用(-t)参数指定它:

```
kafkacat -L -b localhost:9092 -t topic1
```

# 查询模式(-Q)

如果您想根据时间戳查找 Kafka 记录的偏移量，查询模式可以帮助您。只需指定主题、分区和时间戳:

```
kafkacat -b localhost:9092 -Q -t topic1:1:1588534509794
```

# 就这些吗？

我很高兴你问了，因为它不是🙂我创建了一个**单页 Kafkacat cheatsheet** 供你下载。**抢过来** [**这里**](https://codingharbour.com/kafkacat-cheatsheet/) 。😉

*原载于*[*https://codingharbour.com*](https://codingharbour.com/apache-kafka/learn-how-to-use-kafkacat-the-most-versatile-cli-client/)*。*