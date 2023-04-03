# 替换卡珊德拉和惊喜中的死节点

> 原文：<https://medium.com/analytics-vidhya/replacing-a-dead-node-in-cassandra-and-surprises-4681287eeddf?source=collection_archive---------4----------------------->

在替换 Cassandra 集群中的死节点时会出现一些意外。

![](img/6dbe9dfc1dd88da0390d2934a52e9fd0.png)

抄送:pixabay

这些意外的各种原因描述如下:

**问题 1:** 由于节点故障，如果集群的其余部分重新启动，则无法替换。

假设集群中有 3 个节点:节点 1、节点 2 和节点 3。你的 node3 崩溃了，你想按照[文档](https://docs.datastax.com/en/archived/cassandra/3.0/cassandra/operations/opsReplaceNode.html)中描述的标准流程用全新的节点 node3_replace 替换 node3。

因此，您仔细遵循了每个步骤，并更新了 node3_replace 的 cassandra-env.sh

```
JVM\_OPTS="$JVM\_OPTS -Dcassandra.replace_address=<address of dead node node3>
```

启动 ndoe3_replace 节点。

现在，在这一点上，你会得到你的第一个惊喜，在节点 1 和节点 2 在节点 3 失败后以某种方式重新启动的情况下，出现了下面的异常。

```
ERROR \[main\] 2017-11-15 13:57:59,043 CassandraDaemon.java:583 - Exception encountered during startup
java.lang.RuntimeException: Cannot replace_address /<node3 ip> because it doesn't exist in gossip
```

**问题 2 :** 无法替换 as 在解决上述问题 1 时，您忽略了提供的解决方案中的一个词。

您的 node3_replace 节点无法启动，您感到困惑。所以你谷歌了一下，在 Stackoverflow [link](https://stackoverflow.com/questions/23982191/cant-replace-dead-cassandra-node-because-it-doesnt-exist-in-gossip) 上找到了一个很好的解决方案。此链接解释了集群丢失死节点 3 的八卦信息的原因，并建议了一个 3 步解决方案:

1.  发现故障节点的节点 ID(从集群中的另一个节点)节点工具状态| grep DN
2.  并将其从集群中删除:nodetool removenode(节点 ID)
3.  现在，您可以清除故障节点的数据目录，并将其作为一个全新的节点进行引导。

所以你很快就按照上面的步骤做了，因为你想尽快解决这个问题，如果你在解决步骤的第 3 步中忽略了“全新”这个词，你会有另一个惊喜:)

```
java.lang.RuntimeException: cassandra does not use new-style tokens!
```

你现在越来越困惑，但这次即使花了一些时间，你也找不到任何谈论这个问题的网站，现在你的困惑变成了沮丧。

实际上，这里出错的地方是您忘记在 node3_replace 节点的 cassnadra-env.sh 中注释掉 replace_address。

**问题 3:** 问题 2 +您所在节点群集中的另一个节点出现故障

这是问题 2 的第二个版本，但是这次 node1 也关闭了，所以您忽略了 node1，您对启动 node3_replace 更感兴趣。

但是在这里，您会惊讶于在 node 启动期间遇到的另一个 Cassandra 异常

```
java.lang.RuntimeException: A node required to move the data consistently is down (/<node1 ip>). If you wish to move the data from a potentially inconsistent replica, restart the node with -Dcassandra.consistent.rangemovement=false
```

这一次，您没有使用 google，因为异常消息很能说明问题。因此，您检查了节点 1 是否关闭，并启动了节点 1，以为这样可以解决问题，但现在您又回到了原因 2

经验教训:在执行管理步骤时不要忽略一个单词:)

**总结:**替换一个死节点和添加一个全新的节点是有概念上的区别的。在替换中，令牌将被分配给新节点，同时还将进行令牌洗牌。

添加新节点的管理步骤与 replace_address 无关(这纯粹是为了替换死节点),不幸的是，如果在添加新节点的过程中添加，cassandra 不会忽略这个参数。

**最后一点:**添加新节点后，不要忘记运行 nodetool clean。