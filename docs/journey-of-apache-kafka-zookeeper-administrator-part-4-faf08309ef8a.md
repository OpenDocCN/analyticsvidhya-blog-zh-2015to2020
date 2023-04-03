# 阿帕奇卡夫卡与动物园管理员之旅(四)

> 原文：<https://medium.com/analytics-vidhya/journey-of-apache-kafka-zookeeper-administrator-part-4-faf08309ef8a?source=collection_archive---------28----------------------->

![](img/7a868ee7d74ab16f600ca112ffef7ac9.png)

随机照片:)

【2019 年 6 月(续)

[在上一篇文章](/@116davinder/journey-of-apache-kafka-zookeeper-administrator-part-3-e31c2f330895)中，我已经解释了 **Apache Kafka** 的不同方面，而在这篇文章中，我将涵盖 **Apache Kafka** 方面的优化。

在安装了 Apache Kafka 和 Monitoring Setup 之后，我很有信心它会工作，我将能够用它处理好流量。

那一次，我读了 LinkedIn 的一篇博客:[Apache-Kafka-200 万-writes-Second-three-priest-machines](https://engineering.linkedin.com/kafka/benchmarking-apache-kafka-2-million-writes-second-three-cheap-machines)所以我决定用我的设置打破这个数字，老实说，我确实打破了这个数字，我能够在 3 个廉价的 Kafka 节点上达到 230 万/秒的吞吐量。

这是一个关于阿帕奇卡夫卡 1-2 周痛苦而严格的测试/优化的故事。

我知道，我应该在这里分享 Kafka 的测试结果，但现在不可能，因为我已经离开了公司，所以我将在这里分享我的经验。

**测试需要的基本东西**
1。廉价的卡夫卡节点有**卡夫卡 2 . 1 . 1**2
2。Kafka 节点上的详细监控
3。独立测试机器

**便宜的卡夫卡节点** 我有 3 个卡夫卡节点(用我的 ansible-playbook 安装/配置)。配置为
* 6 Vcore
* 10 Gbps 网络
* 12–24 GB RAM
* 100 GB 磁盘(由 **SAN** 支持，超过 10 Gbps 网络)

**JVM 设置(OpenJDK 1.8 )
6 GB 堆大小**

**不同的话题配置**

```
**kafkaperf1R1:** Partition 1 with Replica 1
**kafkaperf1R3:** Partition 1 with Replica 3
**kafkaperf3R1:** Partition 3 with Replica 1
**kafkaperf3R3:** Partition 3 with Replica 3
**kafkaperf6R1:** Partition 6 with Replica 1
**kafkaperf6R3:** Partition 6 with Replica 3
**kafkaperf9R1:** Partition 9 with Replica 1 **kafkaperf9R3:** Partition 9 with Replica 3 **kafkaperf12R1:** Partition 12 with Replica 1
**kafkaperf12R3:** Partition 12 with Replica 3
**kafkaperf15R1:** Partition 15 with Replica 3
**kafkaperf15R3:** Partition 15 with Replica 3
**kafkaperf18R1:** Partition 18 with Replica 3
**kafkaperf18R3:** Partition 18 with Replica 3
```

**如何执行测试** 创建主题
`bin/kafka-topics.sh --create --topic kafkaperf1R1 --bootstrap-server localhost:9092` **现在进行第一轮测试，**

```
bin/kafka-producer-perf-test.sh --topic kafkaperf1R1 --num-records 100000 --record-size 100 --throughput 100000 --producer-props acks=1
```

查结果查卡夫卡监控系统，卡夫卡是不是在什么地方窒息了？监控图表中有任何峰值或异常情况。

**非常重要:**在 **Microsoft One Note** 或类似工具中记录所有测试，以便您可以在以后需要时进行比较或显示/导出为 PDF。

[**OS &网络优化**](https://github.com/116davinder/kafka-cluster-ansible/blob/master/roles/common/tasks/systemTuning.yml)

```
- name: OS Tuning
  sysctl:
    name: "{{ item.key }}"
    value: "{{ item.value }}"
    state: present
  loop:
    - { "key":"vm.max_map_count", "value": "{{ kafkaVmMaxMapCount }}" }- name: Networking Tuning
  sysctl:
    name: "{{ item.key }}"
    value: "{{ item.value }}"
    sysctl_set: yes
    state: present
    reload: true
  loop:
    - { "key": "net.ipv4.tcp_max_syn_backlog", "value": "40000" }
    - { "key": "net.core.somaxconn", "value": "40000" }
    - { "key": "net.ipv4.tcp_sack", "value": "1" }
    - { "key": "net.ipv4.tcp_window_scaling", "value": "1" }
    - { "key": "net.ipv4.tcp_fin_timeout", "value": "15" }
    - { "key": "net.ipv4.tcp_keepalive_intvl", "value": "60" }
    - { "key": "net.ipv4.tcp_keepalive_probes", "value": "5" }
    - { "key": "net.ipv4.tcp_keepalive_time", "value": "180" }
    - { "key": "net.ipv4.tcp_tw_reuse", "value": "1" }
    - { "key": "net.ipv4.tcp_moderate_rcvbuf", "value": "1" }
    - { "key": "net.core.rmem_default", "value": "8388608" }
    - { "key": "net.core.wmem_default", "value": "8388608" }
    - { "key": "net.core.rmem_max", "value": "134217728" }
    - { "key": "net.core.wmem_max", "value": "134217728" }
    - { "key": "net.ipv4.tcp_mem", "value": "134217728 134217728 134217728" }
    - { "key": "net.ipv4.tcp_rmem", "value": "4096 277750 134217728" }
    - { "key": "net.ipv4.tcp_wmem", "value": "4096 277750 134217728" }
    - { "key": "net.core.netdev_max_backlog", "value": "300000" }
```

[**卡夫卡参数调谐**](https://github.com/116davinder/kafka-cluster-ansible/blob/master/inventory/development/group_vars/all.yml#L42-L66)

```
### Production Optimization Parameters
### if nothing is set then it will use default values.
kafkaDefaultReplicationFactor: 3
kafkaMinInsyncReplicas: 2
kafkaBackgroundThread: 10
kafkaMessagesMaxBytes: 1000012 # 1MB approx
kafkaReplicaFetchMaxBytes: 2000000 # this should be higher than kafkaMessagesMaxBytes
kafkaQuededMaxRequests: 500
kafkaNumReplicaFetchers: 1
kafkaNumNetworkThreads: 6
kafkaNumIoThreads: 8
kafkaSocketSendBufferBytes: 102400
kafkaSocetReceiveBufferBytes: 102400
kafkaSocetRequestMaxBytes: 104857600
kafkaNumPartitions: 1
kafkaNumRecoveryThreadsPerDataDir: 1
kafkaOffsetsTopicReplicationFactor: 3
kafkaTransactionStateLogReplicationFactor: 3
kafkaTransactionStateLogMinIsr: 3
kafkaLogFlushIntervalMessages: 10000
kafkaLogFlushIntervalMs: 1000
kafkaLogRetentionHours: 168
kafkaLogSegmentBytes: 2073741824 # need to ask expert kafka, should we use default 1GB here or 2GB or more
kafkalogRetentionCheckIntervalMs: 300000
kafkaGroupInitRebalanceDelayMs: 3
```

**现在是第二轮测试，**我知道重新进行所有测试是一个痛苦的过程，但必须有人去做，否则，没有人会知道当有人对卡夫卡施加压力时会发生什么，卡夫卡在同样的环境下会有什么表现。再次将所有测试记录到一个笔记或类似的工具中。

**现在是第三轮测试，**
首先，检查之前运行的测试，您将会知道，一旦副本 3 达到分区 6 和更高，您将获得非常好的延迟和 100 万吞吐量，让我们将吞吐量进一步提高到 200 万，现在只在分区 6 和更高的配置上运行测试。**再次将所有测试记录到一个笔记或类似的工具中。**

**现在是第四/第五/第六轮测试，** 首先检查第三轮测试中的最佳案例，并根据需要减少表现不佳的主题数量。让我们将吞吐量增加到 2.1 / 2.2 / 2.3，并在系统阻塞+任何异常峰值的情况下密切关注监控系统。再次将所有测试记录到一个笔记或类似的工具中。

这里是最后一个场景，
尝试将记录大小增加到 512 KB / 1 MB，而不是增加吞吐量。看看网络会发生什么，你能达到你的网络极限吗？这将是我们**第 7&第 8 轮测试**，**再次将所有测试记录到一个笔记或类似的工具中。**

**有趣的是**，CPU，内存& JVM 像 GC 时间+使用的堆之类的东西，即使在我达到 230 万的目标时也从未改变过。230 万之后，我在看那个 Avg。99%的延迟是以秒为单位的，这对于我的要求来说有点过了。

旅程将在下一个话题继续(**雅虎卡夫卡经理**又名 **CMAK** )！