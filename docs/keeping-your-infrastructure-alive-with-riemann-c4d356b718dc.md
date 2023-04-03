# 借助 Riemann 让您的基础设施保持活力

> 原文：<https://medium.com/analytics-vidhya/keeping-your-infrastructure-alive-with-riemann-c4d356b718dc?source=collection_archive---------0----------------------->

黎曼( [http://riemann.io](http://riemann.io) )是一个很牛逼的工具。这是毫无疑问的。然而，僵硬的学习曲线和对 Clojure 知识的需求使得它非常不受欢迎。

在 [Mintly](https://mintly.eu) ，我们成功地将它集成为我们的监控、警报和监视框架，它运行得非常完美。然而，大量的血汗被用来创造它。这个博客的目的是引导潜在用户。

# 获得关于黎曼的知识

您将遇到的第一个主要问题是缺少文档。我是说严重缺乏。

我发现 https://www.artofmonitoring.com 的《T4》这本书非常有用。所以对于任何考虑使用黎曼的人来说，这是必读的。

在(也只有在)黎曼的介绍之后，观看创作者自己的视频是有用的:[https://vimeo.com/131385889](https://vimeo.com/131385889)。我建议以后再看的原因是因为 Kyle 说话很快，那里有很多信息(被非常漂亮的图片覆盖)。并且在没有实际背景的情况下，不可能正确地使用这些信息。

# 黎曼有什么不同？

第一个也是最常见的问题。在进入这个问题之前，需要先提出一些理论。

有三种类型的监控[1]

1.  无监控/手动监控。当你通过手工工具检查系统是否有效时
2.  反应监测。当你的服务周期性地轮询和检查具体的服务器是否还活着的时候。Nagios 是最受欢迎的例子之一
3.  主动监控。它是指每个服务定期向某个中央机器报告自己的状态

监控策略取决于以下因素的综合成熟度:

*   产品
*   组
*   公司

只有当所有这些部分都成熟时(公司关心、开发者关心并且产品本身足够有价值)，才使用主动监控解决方案。这些现在非常流行——大多数开发人员现在都听说过 ELK stack 和沃尔夫斯堡。

有趣的是(很少有人知道)，警报共享相同的规则。虽然有很多工具用于主动监控，但只有少数工具用于实际的主动警报。它们都很复杂。也许其中一个更为人所知(但特性本身大多不为人所知)——log stash 能够通过提供一些额外的设置来做到这一点。从这个角度来看，黎曼能够带来灵活快速的主动预警。

# 主动和被动警报有何不同？

简而言之，有了反应式警报，您就能够只根据趋势发出警报。你不能(在很大程度上)对单一事件做出反应。这是因为您正在处理的数据通常是聚合的。关于这个话题可以做很多讨论，对于大多数情况来说，这已经足够了。你想得到警告的通常数据都是关于趋势的。例如，硬盘空间不足—警报，服务器负载过高—警报，服务 99%响应时间过高—警报。但是，如果你每秒有 1000 个事件，那么第 99 个百分位数会丢失 10 个事件。即使是 99.99 百分位，您也可能会丢失事件。对于大多数情况来说，这可能已经足够好了，但是如果**每个**事件都是业务关键型的呢？我猜银行不会同意偶尔忽略一些事件。

这里需要主动预警。他们像过滤器一样工作，每个事件都要经过**。**

# 是时候驱散困惑的迷雾了

黎曼本身不是另一个监控框架。实际上，它是事件处理-路由工具，通过站在巨人(ELK、Graphite 等)的肩膀上，可以创建监控基础架构，并能够对原始警报发出警报，再次通过站在巨人的肩膀上，向 Slack/Hipchat 写入消息，发送电子邮件，或触发 pagerduty。所以它通常是对系统的补充而不是改变。

# 集成到现有环境中的问题

所以主要的问题是——把它放在哪里。如果基础设施是新鲜的，那么在每个软件中添加 Riemann 客户端并直接发送所有内容应该没有问题。然而，更有可能的情况是，您有一个在 StatsD 协议下工作的聚合器。Riemann 是 StatsD 的一种替代(它做 StatsD 所做的一切以及更多)，但是它不接受以这种方式格式化的消息(接受 graphite 格式)。如果你不怕引入 JVM 栈， [Riepete](https://github.com/simao/riepete) 可以是一个解决方案。

# 使用被动(拉)型指标

在一些工具中，将度量推向黎曼比在其他工具中更难。例如 MySQL、PostgreSQL、Redis，但这些指标实际上非常有价值。对于这些情况，黎曼本身并不适合(石墨和 ELK 也是如此)。这个问题的解决方案可能比您想象的更简单——您只需要数据收集器。CollectD 做得很棒。对于你不能从 CollectD 获得的东西(例如，插件丢失，或者不能以你想要的方式工作)，有 [Riemann Tools](https://github.com/riemann/riemann-tools) ，它基本上是一个 Ruby 应用程序，从各种来源查询指标。

# 化繁为简

上面提到的大部分东西可能(也应该)听起来有点可怕。对小项目来说开销太大了(还记得成熟度规则吗？项目也需要成熟)。为了使它更简单(并避免 JVM + Ruby 栈仅用于监控)，我们创建了 [Oshino](https://github.com/CodersOfTheNight/oshino) 。

Oshino 是 StatsD + CollectD + Riemann tools(虽然 CollectD 还是和硬件度量有关)。它能够接受推指标(以 StatsD 格式)和拉指标(查询各种东西)。

这可能有点自以为是，因为(目前)没有外部的贡献者。

# 黎曼装置

这是最有争议的部分，因为众所周知，配置是通过 Clojure 代码完成的。然而，这并不像听起来那么糟糕，因为总的来说 Clojure 语法并不比大多数 DSL 大，所有的魔法都发生在逻辑中。作者做了大量工作使逻辑流程尽可能简单。

完整的设置如下:

```
(let [host "0.0.0.0"]
  (tcp-server {:host host})
  (udp-server {:host host})
  (ws-server  {:host host})
  (repl-server {:host "127.0.0.1"}))
; Expire old events from the index every 15 seconds.
(periodically-expire 15 {:keep-keys [:host :service :tags :metric]})
(require '[mintly.etc.influxdb :refer :all])
(require '[mintly.etc.email :refer :all])c(let [index (index)](streams
    (default :ttl 60
      index(where (tagged "deploy")
              index
              #(info %)
              persist-influxdb
      )(where (tagged-any ["statsd", "riepete", "sincity", "view", "collectd", "db", "auth", "api"])
         index
         persist-influxdb
      )(where (tagged "traffic")
         index
         persist-influxdb-w-tags
      )(where (tagged "discovery")
        (changed :state
            (rollup 2 3600
                (email "[alerts@mintly.eu](mailto:alerts@mintly.eu)")
            )
        )
      )(where (tagged-all ["oshino", "heartbeat"])
         (changed :state
           (email "[alerts@mintly.eu](mailto:alerts@mintly.eu)")
         )
      )
    )
  )
)
```

是的，我知道，乍一看很混乱。所以让我们来分析一下。

首先

```
(let [index (index)]
  (streams
     ...
))
```

就当这是个圈套。就像编程语言中的“主”函数。

另一件事，每个表情看起来像这样

```
(<function> <arg1> <arg2> ... <arg#>)
```

因此，默认表达式实际上是一个函数“default ”,如果参数“ttl ”(生存时间)未设置，它会将其设置为 60

```
(default :ttl 60 ...)
```

实际的逻辑从“where”表达式开始，此时您将事件分支到单独的流中，在流中它将被相应地处理。

“where (tagged …”表达式似乎是不言自明的。更有趣的一个是:

```
(changed :state ...)
```

使用“:state”，您通常传递与该对象状态相关的信息，例如，服务可以是“关闭”或“启动”，或者您可以测量请求/响应时间的平均时间，并将其分为“错误”、“警告”、“正常”。当状态保持不变时，实际上不会触发任何事件，但是当它改变时，您会得到通知。通过这种方式，您可以看到服务何时停止，何时恢复。对我们来说，它被证明是极其有用的机制。

这些警报会很快导致垃圾邮件。为了缓解这种情况，有两个很棒的功能“节流 <events_count><time_span>”和“汇总 <events_count><time_span>”</time_span></events_count></time_span></events_count>

这两个函数都允许在定义的时间段内发送 X 个事件。差别很小:节流完全丢弃溢出事件，汇总它们，然后作为列表发送。Rollup 听起来更好，但是有代价——每个存储的事件都要消耗一些 RAM。更高级的场景结合了这两者，允许聚合一定数量的事件(例如 1000 个)并且不会溢出内存。

```
(require '[mintly.etc.influxdb :refer :all])
(require '[mintly.etc.email :refer :all])
```

是我们的自定义代码导入。基本上，influxdb 部分由写入数据库的函数组成，而 email one 使用我们的外部 API 来发送电子邮件。

注意:这里使用 InfluxDB 而不是 Graphite，这是因为一些其他的原因，这些原因目前并不相关。它们可以以相同的方式使用。

# 自我修复基础设施

黎曼的一个独特之处在于，你可以用它来修复你的基础设施。在某些情况下，只需从外部 JAR 执行命令就可以实现。这个 JAR 可以用任何基于 JVM 的语言创建，但是当然 Clojure 或 Java 产生的问题最少。

可以添加额外的 jar 来更新 Riemann 的 sysconfig 文件，如下所示:

```
EXTRA_CLASSPATH=/path/to/your/jar
```

然后通过 Clojure 的 import 语句包含这段代码，并作为普通函数执行。

一个实际的用例可能是在高峰时间产生额外的 AWS 实例来处理负载。只需对 API 稍加修改，就很容易做到，并且带来了很大的价值。但是，请注意，阻塞程序可以停止当前的黎曼事件循环。为了防止这种情况，您需要编写以异步方式处理这种情况的代码。

# 标度黎曼

很明显，单个节点并不能完全解决问题。没有人想要 nexus point。然而，黎曼并没有为此提供任何琐碎的路径，它只给你自己找到它的能力。这种能力来自于黎曼能够在 TCP 端口上接收和发送。从稍微不同的角度看，你可以理解没有什么可以阻止你从**节点 A** 向**节点 b**发送相同的事件

一般来说，你希望有主从式结构。您在每台机器上添加从实例，让它们对本地问题做出反应(例如，该节点上的 HDD 空间不足)，并将更多的全局实例发送到主节点。为了使它更有弹性，你实际上可以制造几个主节点，对它们进行负载平衡，在它们之间设置复制，等等。这取决于你的知识和想象力

配置通常是这样的:

```
(let [index (index)downstream (async-queue!
                    :agg-forwarder        ; A name for the forwarder
                    {:queue-size     1e4  ; 10,000 events max
                     :core-pool-size 4    ; Minimum 4 threads
                     :max-pools-size 100} ; Maxium 100 threads
                    (forward
                      (riemann.client/tcp-client :host "<address of riemann master node>")))      ]
```

所有这些“异步队列！”事情有点混乱。其结构如下所示:

```
(async-queue! :<name of this queue> {<queue parameters>} 
  (<code to execute>))
```

稍后，您只需执行(或者在主代码分支中，或者在一些“where”语句之后):

```
(batch 100 1/10 downstream)
```

命令“下游”本身足以将事件传递给主设备，但是为了更有效地完成它，我们将事件分组到批处理中。

# 安全性

此时，有些人可能会开始担心安全性，因为您可能会发送一些有价值的信息。黎曼在这里的策略很简单:接受 TCP、UDP 和 WebSocket。每个协议都允许您创建 TLS 加密连接。仅此而已。

# 基准又名我想拍摄我所有的事件黎曼

我还没有看到更好的优化 JVM 使用。它确实从 Netty 框架中榨取了很多。然而，这归结于硬件、操作系统和 JVM 本身的限制。此外，这取决于您的配置设置。半官方的基准可以在这里找到【https://aphyr.com/posts/279-65k-messages-sec 。传说在生产层硬件上，单个节点每秒可以处理几百万个请求。我无法证明也无法否认，直到我亲眼看到。

直截了当地说——如果每个节点都有 Riemann 实例，那么您不太可能比您的服务更快地达到峰值。不过，如果能获得这方面的经验就好了。

# 总结

总的来说，黎曼是非常灵活和有用的工具，但它需要一些爱。基本设置非常简单，但是对于复杂的设置，您可能需要更深入的 Clojure 知识。

在确实需要确保每个事件都得到正确处理的情况下，我会推荐这种方法。没有多少产品要求这样。然而，当它发生时，你可能会受到极大的挑战。

[1]《监控的艺术》，詹姆斯·特恩布尔