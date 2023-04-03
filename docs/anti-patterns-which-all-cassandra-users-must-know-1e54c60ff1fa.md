# 所有 Cassandra 用户都必须知道的反模式

> 原文：<https://medium.com/analytics-vidhya/anti-patterns-which-all-cassandra-users-must-know-1e54c60ff1fa?source=collection_archive---------1----------------------->

再多的性能调优也无法减轻已知的反模式。当你在谷歌上搜索“Cassandra 中的反模式”时，你会找到很多信息。Datastax 做了大量的工作，列出了其中的许多，但这还不是全部。我的目标是在一个地方记下所有列出的反模式。这对每个 Cassandra 用户来说都是非常重要的。

1.写前读:写前读模式两个主要缺点是 a .性能影响 b .如果多个客户端应用程序并行访问相同的记录，您将无法实现原子比较和设置。为了解决后一个问题，用户最终在 Cassandra 中使用 LWT(轻量级交易)。但是这恶化了延迟方面的性能。
改变数据模型的最佳方式是避免这种模式。[这个](https://easyprograming.com/posts/2017/08/cassandra-deep-dive-read-before-write-evil/)是一个非常好的帖子，其中有一些数据模型变化的好例子。

2.集合意味着存储/反规范化相对少量的数据。使用(单个)集合来存储大量数据是一种反模式。尽管地图和列表的最大大小是 2 GB，但用户应该尽量保持比 MBs 小得多的值。

3.宽分区是 Cassandra 中多个问题的根本原因。一个经验法则是不要超过 100 MB，然而一个好的数据模型设计应该使它更小。小贴士:桶化在这里可以有所帮助。例如，存储物联网应用的传感器事件，而不是将主键保持为(sensor_id，insert_time ),这将导致每个传感器的宽分区，可能将其更改为((sensor_id，date) insert_time)更有意义。事件虽然[CASSANDRA-11206](https://issues.apache.org/jira/browse/CASSANDRA-11206)(3.5+版本)在一定程度上移动了宽分区的屏障，但仍然建议不要有太宽的分区。

4.将大的有效负载存储为数据类型为 text 或 blob 的列是不明智的。建议的实际大小小于 1 MB，但尽量保持在 Kbs。小贴士:把大文件放在像 S3 这样的存储桶里，把元数据/引用链接保存在 Cassandra 表中，或者如果 Cassandra 是你唯一的选择，把一个大的 blob 分成几个小块，放在不同的行里。

5.没有分区键的 Select all 或 select count 将导致全表扫描，不应在大型数据集上运行。

6.物化视图是一个实验性的特性，应该在生产中避免。

7.非常小心地使用 Cassandra 二级索引。SI 在高或低的肉欲场上不是一个明智的决定。

8.队列类型数据结构(一旦使用就删除) :队列反模式提醒人们，任何依赖于数据删除的设计都可能是性能很差的设计。Cassandra 提供了 *tombstone_warn_threshold* 配置参数来警告您是否已经这样设计了您的数据模型。

9.“允许过滤”基本上是指需要扫描一堆数据来返回其中的一部分，相当于 sql 数据库中的全表扫描。我读到的最有趣的定义之一是“允许过滤”实际上意味着“允许我销毁我的应用程序和集群”这意味着数据模型不支持查询，并且不能伸缩。话虽如此，如果你真的知道你在做什么，你仍然可以使用它，例如通过指定分区键来允许查询过滤。

10.多分区批处理:如果我们使用批处理查询，并且它的大小很大，这将导致协调器级别的延迟。来自 RDBMS 的人应该记住这一点，多分区批处理不会提供更好的性能。多分区批处理是 Cassandra 中的一种反模式，所以要避免。Cassandra 中的 Batch (Logged)应该用于在多个非规范化表中保持写原子性。以下是与批处理相关的一些配置参数。

> batch _ size _ WARN _ threshold _ in _ kb
> (默认值:每批 5KB)使 Cassandra 在任何批大小超过此千字节值时记录一条警告消息。
> 注意:
> 增加这个阈值会导致节点不稳定。
> batch _ size _ fail _ threshold _ in _ kb
> (默认值:每批 50KB)Cassandra 会使任何大小超过此设置的批处理失败。默认值是批处理大小警告阈值以 kb 为单位的值的 10 倍

11.在生产中不使用推荐的[设置](https://docs.datastax.com/en/dse/6.7/dse-admin/datastax_enterprise/config/configRecommendedSettings.html)运行肯定是“否”。

12.最大堆大小太小:根据我的经验，Cassandra 在小于 8GB 的堆上生产看起来太小了。所以 8 GB 是绝对的最小值，除非你很少阅读。如果您有 8GB MAX_HEAP_SIZE，请继续使用 CMS。对于 G1，从 16 GB 或至少 12 GB 开始。

13.最大堆大小太大:即使使用 G1，也不建议超过 32GB。它将产生递减的结果。MAX_HEAP_SIZE 的最佳值将取决于多种因素，如访问模式、数据模型、每个节点的数据等，因此尝试调整它，看看哪个值最适合您。

13.不建议使用 IN 子句的 Multi-get，因为它会给单个节点带来压力，因为通常必须查询许多节点，而并发的多个单独选择更可取。

14.列表与集合:列表上的一些操作确实在本地存储上执行写前读。此外，一些列表操作本质上不是等幂的，这使得它们在超时的情况下重试是有问题的。因此，建议尽可能选择集合而不是列表。

15.使用 SAN 或 NAS 成为 IO 和网络路由方面的瓶颈，因此避免使用 SAN 或 NAS，而是使用本地磁盘 SSD 或旋转磁盘。

17.如果由于 io 模式冲突而使用旋转磁盘，则保持 commitlog 和 ssd 在同一个磁盘中。如果是 SSD 或者 EC2 就可以了。

16.在 C*前面放一个负载均衡器是完全没有必要的，只会增加另一个故障点。而是使用具有不同类型负载平衡策略的驱动程序。

18.如果您在 AWS 上运行 C*，EBS 卷是有问题的。它是不可预测的，并且在许多情况下吞吐量是有限的。请使用临时设备，而不是将其分条。

19.行缓存:行缓存有一个更加有限的用例。行缓存将整个分区拉入内存。如果该分区的任何部分被修改，则该行的整个缓存都将失效。对于大分区，这意味着缓存可能会频繁地缓存和失效大块内存。因为您确实需要大部分静态分区，所以对于大多数用例，建议您不要使用行缓存。

21.绑定空值:将空值绑定到准备好的语句参数会生成 tombstones (java 驱动示例:*boundstatements . add(prep stmt . bind(1，null)) )* ，但是在 Cassandra 2.2+中保留未设置的绑定参数结合 DataStax Java 驱动 3.0.0+不会创建 tombstone。因此，如果值为空，则不要添加绑定值。

22.同一列上的密集更新:这将在读取期间导致性能影响，因为需要多个表扫描。为了避免这种情况，如果您的 I/O 能够跟上，请使用分级压缩。否则，请更改您的数据模型，参见下面的示例:

> 设计不好:
> 创建表 sensor_data ( id long，value double，PRIMARY KEY(id))；
> 更新传感器 _ 数据集值=？其中 id =？—这是对相同 id 的频繁操作
> 从 sensor_data 中选择值，其中 id =？—这里您将获得读取延迟(最坏的情况:超时)
> 
> 更好的设计:
> 创建表 sensor_data ( id 长，日期时间戳，值双精度，主键(id，date))带聚类顺序(日期 desc)；
> 插入 sensor_data (id，date，value)值(？, ?, ?);
> 从 sensor_data 中选择值，其中 id =？极限 1；—您将获得最新的传感器值

23.并发模式和拓扑变化:这种设计的一个例子是每天创建/删除表(单独的表用于每日数据)。当您尝试更改拓扑(例如添加/删除节点)时，问题出现了，并且在您的拓扑更改正在进行时，您的应用程序尝试将 DDL 作为日常事务的一部分来执行。并发的模式更改和拓扑更改是一种反模式。

24.混合版本群集中的 DDL 或 DCL 操作:请在升级过程中避免它们..应该在步骤前或步骤后完成。

25.在更新期间，任何涉及流的活动，如修复、扩大或缩小，都会给你带来麻烦，所以要避免。混合版本集群只支持跨版本读写。

26.不要试图在不同版本的 Cassandra 中保留同一个集群的两个数据中心。这仅适用于您正在升级集群的情况。

27.并发引导多个节点:并发引导可能会导致节点之间的令牌范围移动不一致。要安全引导每个节点，请尝试顺序引导。添加单独的 dc 节点时，可以通过设置 *auto_bootstrap=false* 并遵循新 dc 中两次连续启动之间的 2 分钟暂停规则来添加。

28.在所有节点上运行完全修复(不带任何参数的 nodetool repair)以避免副本的冗余修复:尝试使用子范围修复(最好使用 reaper 之类的工具)或在每个节点上顺序运行 nodetool repair -pr。请记住，Cassandra 4.0 在修复和固定增量修复方面有了很大的改进，值得一试。

29.表太多:集群中的表太多可能会导致高内存使用率和压缩问题导致性能大幅下降。尽量保持计数少于 200。

30.使用字节序分区器:不推荐使用字节序分区器(BOP)。请改用虚拟节点(vnodes)。

31.CPU 频率缩放:正如 Datastax 提到的，最近的 Linux 系统包括一个称为 CPU 频率缩放或 CPU 速度缩放的特性。它允许动态调整服务器的时钟速度，以便当需求或负载较低时，服务器可以以较低的时钟速度运行。这降低了服务器的功耗和热量输出(这会显著影响冷却成本)。不幸的是，这种行为对运行 DataStax 产品的服务器有不利影响，因为吞吐量可能会被限制在较低的速率。

32.不熟悉 linux:操作员必须了解一些基本的故障排除 Linux 命令来指出问题。例如:top、dstat、iostat、mpstat、iftop、sar、lsof、netstat、htop、vmstat 等。需要一个单独的博客来涵盖所有这些。

33.测试不足:确保在规模和生产负荷下进行测试。这是确保您的系统在应用程序上线时正常运行的最佳方式。要正确测试，请参见生产前测试集群的数据表[提示](https://docs.datastax.com/en/dse-planning/doc/planning/planningTesting.html)。

34.多次准备同一个查询通常是一种反模式，可能会影响性能。Cassandra 也为 same 生成警告消息“重新准备已经准备好的查询”。

35.用 RF=1 运行 Cassandra。你为什么要这么做？你没有使用 Cassandra 来避免单点故障吗？

36.在生产中使用简单策略。简单的复制策略可用作测试集群。网络拓扑系统策略应优先于简单策略，以便于以后向集群添加新的物理或虚拟数据中心

37.如果使用 GossipingPropertyFileSnitch 和 Cassandra-topology.properties 仍然存在于服务器中，可能会导致一些不必要的行为，例如:在节点工具状态输出中显示不同的节点已关闭，或者当重新启动一个节点而其他节点已关闭时，节点工具状态以错误的 dc 名称显示所有已关闭的节点。当文件存在时，*GossipingPropertyFileSnitch*总是加载 Cassandra-topology.properties。从任何新集群或从 *PropertyFileSnitch* 迁移的任何集群上的每个节点中删除该文件。

38.不要在相同记录上混合普通写入和 LWT 写入，以避免并发执行期间的不一致性。此外，并行对同一记录进行正常读取(一致性仲裁)的 LWT 写入可能会显示过时数据，因此为了避免这种不一致性，请使用具有一致性的串行/本地串行读取。

参考文献:
1。[https://docs . datas tax . com/en/DSE-planning/doc/planning/planning anti patterns . html](https://docs.datastax.com/en/dse-planning/doc/planning/planningAntiPatterns.html)
2 .[https://strange-loop-2012-notes . readthe docs . io/en/latest/Tuesday/Cassandra . html](https://strange-loop-2012-notes.readthedocs.io/en/latest/tuesday/Cassandra.html)
3 .[https://www . slide share . net/doanduyhai/Cassandra-nice-use-cases-and-worst-anti-patterns](https://www.slideshare.net/doanduyhai/Cassandra-nice-use-cases-and-worst-anti-patterns)