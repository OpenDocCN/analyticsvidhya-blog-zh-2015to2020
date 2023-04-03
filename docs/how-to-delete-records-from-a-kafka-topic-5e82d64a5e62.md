# 如何从 Kafka 主题中删除记录

> 原文：<https://medium.com/analytics-vidhya/how-to-delete-records-from-a-kafka-topic-5e82d64a5e62?source=collection_archive---------19----------------------->

![](img/014bf85b5230ca696f2078ea1efefcd6.png)

*图片来源:* [*阿德利*瓦希德](https://unsplash.com/@adliwahid?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

我不时会收到同事的请求，他们想删除一些或所有卡夫卡主题的记录。这个请求通常是在有人在测试主题中产生了错误的数据，或者由于生产者代码中的错误而产生的。或者只是因为他们想要一个干净的石板。

不管是什么原因，今天我会告诉你一些方法来删除一些或所有卡夫卡主题的记录。

不言而喻，在生产环境中使用下面描述的方法之前，您应该使用您的最佳判断并(至少)检查两次。

# 卡夫卡-删除-记录

该命令是 Kafka CLI 工具的一部分。它需要两个参数:

*   引导服务器和
*   JSON 文件，描述应该删除哪些记录。

该命令允许您删除从一个分区的开始直到指定偏移量的所有记录。

注意:不能在主题中间删除记录。

JSON 文件指定了一个或多个我们想要从中删除记录的分区。

```
{
    "partitions": [
        {
            "topic": "my-topic",
            "partition": 0,
            "offset": 3
        }
    ],
    "version": 1
}
```

在这里，我们已经指定，对于主题“my-topic”的分区 0，我们希望删除从开始直到偏移量 3 的所有记录。执行该命令后，分区 0 的起始偏移量将是偏移量 3。

# 删除主题中的所有记录

> **注意:这不适用于压缩主题**

如果您想要删除所有消息，另一个选项是将主题的保留减少到一个小值(例如 100ms)，等待代理从主题中删除所有记录，然后将主题保留设置为其原始值。下面是怎么做的。

首先，将 retention.ms 设置为 100 毫秒。

```
kafka-configs --zookeeper localhost:2181 \
--entity-type topics \ 
--entity-name my-topic \ 
--alter --add-config retention.ms=100
```

然后，等待代理删除保留期过期的消息(即所有消息)。要知道该过程是否完成，请检查开始偏移和结束偏移是否相同。这意味着该主题没有更多可用的记录。根据您的设置，Kafka 可能需要几分钟来清理主题，因此请继续检查开始偏移。

使用 GetOffsetShell 类检查主题分区的开始和结束偏移量。为了检查末端偏移，将参数**时间**设置为值-1:

```
kafka-run-class kafka.tools.GetOffsetShell \ 
--broker-list localhost:9092 \ 
--topic my-topic \ 
--time -1
```

为了检查开始偏移，将参数**时间**设置为 **-2** :

```
kafka-run-class kafka.tools.GetOffsetShell \ 
--broker-list localhost:9092 \ 
--topic my-topic \ 
--time -2
```

清除主题后，将 retention.ms 恢复为其原始值:

```
kafka-configs --zookeeper localhost:2181 \ 
--entity-type topics \ 
--entity-name my-topic \ 
--alter --add-config retention.ms=<ORIGINAL VALUE>
```

# 删除一个主题，然后重新创建

不如前两种方法优雅，但在某些情况下可能是更容易的解决方案(例如，如果主题创建是脚本化的)。

```
kafka-topics --bootstrap-server localhost:9092 \ 
--topic my-topic \ 
--delete
```

然后再次创建它:

```
kafka-topics --bootstrap-server localhost:9092 \ 
--topic my-topic \ 
--create \ 
--partitions <number_of_partitions> \ 
--replication-factor <replication_factor>
```

## 使用这种方法时需要注意的事情很少

确保在集群中启用了主题删除。设置 **delete.topic.enable=true** 。在 Kafka 1.0.0 中，该属性默认为真。

确保所有消费者都已停止消费您要删除的主题中的数据。否则，它们会抛出如下错误:

```
Received unknown topic or partition error in fetch for partition my-topic-0
```

或者

```
Error while fetching metadata with correlation id 123 : {my-topic=LEADER_NOT_AVAILABLE}
```

如果您启动并运行了消费者，还有一件事可能会发生:如果集群范围的属性**auto . create . topics . enable**为真(默认情况下为真)，那么主题将会自动创建。本质上还不错，但是它将使用默认的分区数量(1)和复制因子(1)，这可能不是您想要的。

这个故事的寓意是——如果使用这种方法，一定要阻止你的消费者🙂

# 你想了解更多关于卡夫卡的知识吗？

我创建了一个卡夫卡迷你课程，你可以完全免费获得**。[在编码港](https://codingharbour.com/)报名。**

***最初发表于*[*https://codingharbour.com*](https://codingharbour.com/apache-kafka/how-to-delete-records-from-a-kafka-topic/)*。***