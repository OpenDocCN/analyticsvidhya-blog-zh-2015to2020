# 阿帕奇卡夫卡与动物园管理员之旅(上)

> 原文：<https://medium.com/analytics-vidhya/journey-of-apache-kafka-zookeeper-administrator-part-1-d84dfde205b?source=collection_archive---------15----------------------->

![](img/20170be598923af932f670bdabd2e888.png)

我正在写一系列关于我的 Apache Kafka & Zookeeper 管理从零到生产的旅程的文章，我现在有很多时间，为什么不写一篇关于它的文章呢？我将在这里分享我学到的经验。希望它能帮助人们轻松操作阿帕奇卡夫卡和动物园管理员。

**注*** 这大约是一年半的旅程。

【2019 年 5 月
我得到了我需要评估不同 Apache Kafka 发行版的信息。在讨论了不同的选项后，决定使用 Apache Kafka，而不是 Apache Kafka 的专有版本。

【2019 年 6 月(非常非常长的一个月:)
我开始了我的阿帕奇卡夫卡& Zookeeper 过山车之旅，开始寻找关于阿帕奇卡夫卡& Zookeeper 的基本信息比如它是如何工作的，为什么会工作。我开始阅读 LinkedIn & Confluent 的博客，他们正在大规模使用阿帕奇卡夫卡。

*   [https://www.confluent.io/blog/apache-kafka-getting-started/](https://www.confluent.io/blog/apache-kafka-getting-started/)
*   [https://engineering . LinkedIn . com/distributed-systems/log-what-every-a-software-engineer-should-know-on-real-time-data-unified](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying)
*   [https://docs.confluent.io/current/kafka/deployment.html](https://docs.confluent.io/current/kafka/deployment.html#running-ak-in-production)

在阅读了几周并检查了安装和测试文档之后。同时，我请求公司数据中心为测试集群提供资源。经过一周的阅读，我理解了阿帕奇卡夫卡的基本要求，也就是阿帕奇动物园管理员的基本要求，所以现在我不得不考虑阿帕奇动物园管理员 T21 的基本要求，以及它是如何工作的。幸运的是，阿帕奇动物园管理员正在工作！我安装了一个测试节点 zookeeper，并开始在 Apache Kafka 上工作，看看它是如何工作的。我手动启动 Kafka，它开始工作，然后开始做一些基本的操作，比如主题和其他东西，看看它在磁盘上存储了什么，在 zookeeper 上存储了什么。

我很擅长 Ansible Automation，所以决定在这个项目中使用它，然后我开始为 Apache Kafka & Zookeeper 制定安装标准，如在哪里，为什么&如何。

早些时候，我为 Cassandra 做了 Ansible Automation，这启发了我为 Apache Kafka 和 Zookeeper 做更好的安装实践。

**阿帕奇卡夫卡的文件夹结构**

```
root@Davinder:/# tree -L 1 /kafka/
/kafka/
├── kafka -> kafka_2.12-2.6.0/
├── kafka-data
├── kafka-logs
├── kafka_2.12-2.5.1
└── kafka_2.12-2.6.05 directories, 0 files
```

**阿帕奇动物园管理员的文件夹结构**

```
root@Davinder:/# tree -L 1 /zookeeper/
/zookeeper/
├── zookeeper -> zookeeper-3.6.1/
├── zookeeper-3.5.8
├── zookeeper-3.6.1
├── zookeeper-data
└── zookeeper-logs5 directories, 0 files
```

为什么高于结构？

它具有以下优势*

*   ***Zookeeper/Kafka****是一个仅指向活动版本的符号链接。*
*   ***数据**是单独的目录，可以用单独的 SSD 盘备份。*
*   ***日志**是单独的目录，可以用单独的硬盘备份。*
*   ***版本目录**存储给定版本的完整状态，以便在需要时可以轻松回滚，并可用于配置比较目的。*

*阿帕奇动物园管理员的旅程将在下一篇文章开始！*