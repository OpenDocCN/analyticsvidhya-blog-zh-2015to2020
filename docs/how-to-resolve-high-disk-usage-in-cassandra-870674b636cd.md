# 如何解决 Cassandra 中的高磁盘使用率

> 原文：<https://medium.com/analytics-vidhya/how-to-resolve-high-disk-usage-in-cassandra-870674b636cd?source=collection_archive---------1----------------------->

![](img/b52b11b22c12136ccfa8f7277837101b.png)

马库斯·斯皮斯克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

以下是 Cassandra 集群中高磁盘使用率的一些故障排除步骤。

找到根本原因:维度不合适，写入的数据比预期的多，ttl 或其他值较高。

# 选项:

## A.添加容量-

1.  添加磁盘:垂直缩放
2.  使用 JBOD 为表添加第二个位置文件夹，移动其中一些，然后重新启动 Cassandra。数据文件目录:

*   /var/lib/cassandra/data
*   /var/lib/Cassandra/data-额外注意:JBOD 功能在急需磁盘空间的紧急情况下会很有用，但是在这种情况下使用它应该是暂时的。

3.添加新节点:水平缩放

## B.减少使用的磁盘空间。

## 不要:

1.  直接删除表
2.  运行修复:它不驱逐墓碑
3.  正在执行删除操作。

## Do 的

1.  快照:使用“nodetool listsnapshot”检查快照是否已经存在于 cassandra 数据目录中，如果该选项不存在，则检查目录。如果存在，则在备份后使用“nodetool clearsnapshot”将其清除。
2.  Cassandra 磁盘中存在的其他文件修复:删除任何堆转储或其他可能存储在那里的文件
3.  放大后忘记运行“节点工具清理”。
4.  请检查“nodetool compactionstats”。通过列出临时表，压缩的当前开销: *tmp* Data.db. fix:禁用压缩。删除它们。逐一进行压实。
5.  修复期间的反压缩还会导致临时的磁盘使用高峰

日志:9e 09 c 490-f1be-11e 7-b2ea-b 3085 f 85 ccae 维修后防碰撞货物事件数据 147.3 GB 158.54 GB 字节 92.91%

147 GB *[压缩比]。例如，如果压缩率为 0.3，那么在反压缩结束后不久就会回收 44GB。

修复:使用子范围修复

6.修复过程中出现的一些溢出。请检查“nodetool netstats”。

修复:节流蒸。nodetool setstreamthroughput。流吞吐量出站兆比特每秒:200 mbps

7.Tombstones:当 tombstones 被逐出时:7.1 如果 tombstones 创建时间跨越了 GCGP 7.2 如果在某个其他文件中不存在具有相同分区的旧数据。(重叠表格)

8.关于压缩战略的思考:STCS vs LSCS。稍后可能会有所缓解，但请注意，空间不应该只是改变压缩策略的理由。

9.TWCS:直接从 cassandra 中删除数据可能会导致数据丢失，但如果使用 twcs，如果稳定的创建时间与 ttl 冲突，可以小心操作。否则可以使用 unsafe _ aggressive _ sstable _ expiration = true，这样在移除之前不会检查重叠的 ss table 文件。

10.注意，在 STCS，较大的桌子不参与压实，除非它属于同一铲斗。

修复:设置“uncheck _ tombstone _ compaction:true”通常会有所帮助。

如果不可用，请使用 sstablemetadata 和用户定义的压缩，如下所示。

使用表元数据:

```
for f in *.db; do meta=$(/home/user1/tools/bin/sstablemetadata $f); echo -e "Max:" $(date --date=@$(echo "$meta" | grep Maximum\ time | cut -d" "  -f3| cut -c 1-10) '+%Y/%m/%d %H:%M:%S') "Min: " $(date --date=@$(echo "$meta" | grep Minimum\ time | cut -d" "  -f3| cut -c 1-10) '+%Y/%m/%d %H:%M:%S') $(echo "$meta" | grep droppable) $(ls -lh $f | awk '{print $5" "$6" "$7" "$8" "$9}'); done | sortMax: 2017/12/07 10:06:46 Min:  2017/01/10 11:40:11 Estimated droppable tombstones: 0.524078481591479 4.4K Jan 21 07:07 ks-table-jb-2730206-Statistics.db
Max: 2018/01/16 17:02:09 Min:  2017/01/10 09:52:06 Estimated droppable tombstones: 0.4882548629950296 4.5K Jan 21 07:07 ks-table-jb-2730319-Statistics.db
Max: 2018/02/01 14:13:56 Min:  2017/01/10 08:15:39 Estimated droppable tombstones: 0.0784809244867484 5.9K Jan 21 07:07 ks-table-jb-2730216-Statistics.db
Max: 2018/04/03 06:21:42 Min:  2017/01/10 08:13:04 Estimated droppable tombstones: 0.17692511128881072 5.9K Jan 21 07:07 ks-table-jb-2730344-Statistics.db
```

用户定义的压缩:

```
SSTABLE_LIST="/Users/tlp/.ccm/2-2-9/node1/data0/ks/table-b260e6a0a7cd11e9a56a372dfba9b857/lb-46464-big-Data.db,\
/Users/tlp/.ccm/2-2-9/node1/data0/ks/table-b260e6a0a7cd11e9a56a372dfba9b857/lb-46464-big-Data.db"JMX_CMD="run -b org.apache.cassandra.db:type=CompactionManager forceUserDefinedCompaction ${SSTABLE_LIST}"
echo ${JMX_CMD} | java -jar jmxterm-1.0-alpha-4-uber.jar -l localhost:7100
#calling operation forceUserDefinedCompaction of mbean org.apache.cassandra.db:type=CompactionManager
#operation returns:
null
```

或者 nodetool garbagecollect 如果有 cassandra 版本也可以有效。注意:这是 io 密集型操作。

11.检查集群是否不平衡:“节点工具状态”负载。修复:确保没有手动错误，如每个节点的 num_token 不相等或 init_token 设置不正确。没有正确选择分区键。

12.磁盘空间超过 90%时的预防措施。由于没有足够的空间修复，您的压缩可能会失败:通过将“未使用的”表移动到其他地方或截断它们来立即获得一些空间，以便为其他表的压缩提供一些空间。

13.可能是你的 Cassandra 版本的一个错误:例如:过时的压缩文件没有被删除。滚动重启后，在修复运行时登录 Cassandra 2.0.14:

*   `INFO [CompactionExecutor:1957] 2020-01-20 06:44:56,721 CompactionTask.java (line 120) Compacting [SSTableReader(path='/var/lib/cassandra/data/keyspace/columnfamily/keyspace-columnfamily-jb-123456-Data.db'), SSTableReader(path='/var/lib/cassandra/data/keyspace/columnfamily/keyspace-columnfamily-jb-234567-Data.db'), SSTableReader(path='/var/lib/cassandra/data/keyspace/columnfamily/keyspace-columnfamily-jb-345678-Data.db')] INFO [CompactionExecutor:1957] 2020-01-20 12:45:23,270 ColumnFamilyStore.java (line 795) Enqueuing flush of Memtable-compactions_in_progress@519967741(0/0 serialized/live bytes, 1 ops) INFO [CompactionExecutor:1957] 2020-01-20 12:45:23,502 CompactionTask.java (line 296) Compacted 3 sstables to [/var/lib/cassandra/data/keyspace/columnfamily/keyspace-columnfamily-jb-456789,]. 136,795,757,524 bytes to 100,529,812,389 (~73% of original) in 21,626,781ms = 4.433055MB/s. 1,738,999,743 total partitions merged to 1,274,232,528\. Partition merge counts were {1:1049583261, 2:309997005, 3:23140824, }`

修复:删除过时的文件，重启并尽快升级到稳定版本。

参考和信用:自己的经验和 Cassandra 用户邮件列表。