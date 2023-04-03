# Cassandra 中读取延迟的 15 个原因

> 原文：<https://medium.com/analytics-vidhya/15-reasons-of-read-latency-in-cassandra-8d965f18f85c?source=collection_archive---------1----------------------->

![](img/393f36a603f5d9107f97d12b9484132e.png)

斯坦尼斯拉夫·康德拉蒂耶夫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

以下是 Cassandra 读取延迟的一些可能原因:

**1。现代医学之父希波克拉底曾经说过“所有的疾病都始于肠道”，同样，我相信《卡桑德拉》中的所有主要问题都始于广泛的隔离。因此，这可能是你慢读的一个好理由。nodetool tablehistogram“分区大小”可以给出一个思路，理论上分区大小不应该超过 100 mb。实际上，它应该远小于 100 mb。**

**修复:**尝试对表的分区键进行分桶，使其具有瘦分区。例如:对于一个雇员表，组织名称上的分区键可能是一个坏的选择，但是(组织，部门)可能是一个修正。

**2。错误查询:**下面是一些会导致协调器级延迟的错误查询的示例。

*   全表扫描:例如 select * from employeeselect count(*)from employee；
*   使用允许过滤。
*   使用 IN 子句。

**修复:**对于全扫描，您可以使用 spark。如果那是不可能的，你可以试试

```
cqlsh -e "copy keyspace.table_name (first_partition_key_name) to '/dev/null'" | sed -n 5p | sed 's/ .*//'
```

对于其他人，尽量避免运行它们，因为它们不可扩展，并且会随着数据的增长而面临挑战。如果您发现任何特定的查询需要时间，那么您可以在 cqlsh 中进行“跟踪”,然后运行一个有问题的查询或“node tool settrace probability 0.001”(这里，要小心将该值设置得太高的含义)并检查跟踪以查看发生了什么。

**3。触摸的表的数量:**如果在读取过程中触摸了更多的表(例如:95%触摸 2 个以上的表),那么您可以期待 cassandra 集群有所改进。您可以使用 nodetool 表/cf 直方图来检查这一点

**修复:**确保你的压实没有滞后。尝试调整压缩策略或 concurrent _ compactors 或 compaction_throughput 选项。

**4。阻塞读取修复:**由于读取期间的摘要不匹配，将触发阻塞读取修复。您可以检查调试日志 digestmismatch 异常(该异常的某些数字不是预期的，但不应该以吨为单位)

**修复:**通过定期运行反熵修复来保持集群的一致性和健康性。写完数据后立即读取数据也可能导致这种情况，请检查延迟的可能性。

**5。推测性重试:** Cassandra 被设计成容错的，因此可以很容易地处理故障节点，但是处理故障节点可能很困难。换句话说，Cassandra 节点完全关闭是没问题的，但如果节点状态波动(经常无响应)，则会由于推测性重试而导致读取延迟，因为节点没有完全关闭，而是变得。

**修复:**需要识别故障节点并排除故障。还要检查您的应用程序是否均匀地分配了负载。确保你的分区键是均匀分布的。也试着禁用 cassandra.yaml 中的 dynamic _ snitch 并测试。

**6。次要索引:**如果选择不当，将导致协调员级别的延迟。如果你在非常高的基数字段(例如:电子邮件 id 或任何唯一的 id)或非常低的基数字段(例如:性别:男/女)上创建 SI，那么你就有麻烦了。

**7。一致性级别和跨 DC 流量:**随着一致性级别的提高，读取操作将会变慢。在多 dc 群集中使用 QUOURM 一致性会导致跨 DC 流量，从而导致读取速度变慢。

**修复:**如果可能，在多 dc 集群的情况下，尝试使用 LOCAL_QUORUM。如果您不想要很强的一致性，您可以使用小于 QUORUM 的读取一致性。避免在生产中使用所有一致性。

**8。墓碑:**如果你的表和范围中有很多墓碑，那么读取将会很慢，如果在新版本的 Cassandra 中越过 tombstone_warn_threshold，你将会看到警告。

```
WARN org.apache.cassandra.db.ReadCommand Read 0 live rows and 87051 tombstone cells for query SELECT * FROM example.table
```

您还可以检查 sstable 元数据来检查逻辑删除率。

**修复:**如果可能的话，首先尽量避免创建墓碑。也可以查看我的第一篇[文章](/analytics-vidhya/how-to-resolve-high-disk-usage-in-cassandra-870674b636cd)如何通过调优来照顾墓碑。

**9。每秒巨大的读取数(rkB/s):** 查看 nodetool tpstats 的输出以了解情况。大量挂起/数据块读取阶段会导致读取延迟增加。

**修复:**通常通过添加节点或减少客户端的读取吞吐量来解决。

10。缓存不足:没有足够的 ram 会降低读取速度，因为内核必须从磁盘读取，而不是从页面缓存读取。所以有足够的页面缓存是好的。Cassandra 还提供了一些缓存，如键缓存、行缓存，这些缓存只能在非常特殊的情况下进行调整(例如:频繁读取一些行)

11。硬件问题:Cassandra 中建议使用本地 ssd 进行快速读取。**修复:**避免使用 NAS 或 SAN。

**12。布隆过滤器:**检查节点工具表状态，大量的布隆过滤器误报计数会导致读取延迟。修复:调整 bloom_filter_fp_chance:根据可用 ram 和表的数量进行调整，对于读取速度慢和大量表，降低 fp 更改以减少磁盘 io。请注意，这将增加内存的使用。

**13。网络延迟和吞吐量:**您可以检查 nodetool proxyhistogram 和，在系统级，您可以使用 ping、traceroute 或 mtr 检查延迟，使用 iftop 命令检查网络吞吐量

**14 垃圾收集:**检查 gc 日志和 nodetool gcstas 以查看 gc 暂停。垃圾收集可能是由多种原因造成，如错误的数据模型、没有足够的最大堆大小(至少从 8G 开始)或未调优的其他垃圾收集参数等。

15。资源受限:检查您的系统在读取期间是否不受 cpu 或 io 的限制。您可以使用 top/htop 或 iostat 这样命令。页面缓存可以使查询更快，因此您还可以使用 free 命令检查是否有足够的页面缓存可用。较高的预读值(> 128 kb)或默认的 chunk_length_in_kb (256 kb)可能会导致较高的 i/o 和读取延迟。
**修复:** Apache Cassandra 有一篇关于相同的[文章](http://cassandra.apache.org/doc/4.0/troubleshooting/use_tools.html)。另外，使用 *blockdev — report* 命令检查预读值，并使用 *blockdev— setra* 命令将其设置为 8 kb。尝试使用 4 kb 的 chunk_length_in_kb 在读取时一次读取较少字节的块，并观察性能。确保逐个调整参数，以确定每个变化的影响。

**结论:**只有拥有合适的数据模型和访问模式，Cassandra 才能很好地适应您的应用。Cassandra 调优是复杂的，在这篇博客中，我给出了解决 Cassandra 中读取延迟时需要检查的要点的高度概述。

如果你遇到任何问题，请随时在评论中添加更多观点。