# 利用 Amazon Redshift 实现数据仓库现代化

> 原文：<https://medium.com/analytics-vidhya/modernize-data-warehouse-with-amazon-redshift-6b65c04da886?source=collection_archive---------14----------------------->

数字数据正以不可思议的速度增长，企业发现很难在保持低成本的同时快速接收、存储和分析数据，因此他们正在将数据仓库迁移到云中。在这篇文章中，我将讨论亚马逊红移；最流行的基于云的数据仓库。

![](img/534cd4ae64459740661f659128e25f2e.png)

来源:https://www.classcentral.com/

# **什么是亚马逊红移？**

Amazon Redshift 是一个快速、完全托管的 Pb 级数据仓库，使使用现有商业智能工具分析所有数据变得简单且经济高效。它允许从拥有数百千兆字节数据的小型数据集开始，扩展到数千兆字节甚至更多。如果数据仓库因 OLAP 事务而过载，Redshift 是一个很好的选择，因为它是为 OLAP 设计的，允许您轻松地组合多个复杂的查询来提供答案。因为 Amazon Redshift 是基于 ANSI SQL 的，所以它允许我们几乎不用修改就可以运行现有的查询。此外，amazon redshift 的运营成本比任何其他数据仓库解决方案都低，但却提供了高质量的性能，而且它是最快的数据仓库解决方案。因此，它已经成为当今数据企业数据仓库和数据集市的流行选择。

# **优势**

**性能**

亚马逊红移的一个关键优势是它的高性能。Amazon Redshift 使用列数据存储以及并行和分布式查询，为几乎任何数据大小提供快速查询和 I/O 性能。大多数常见的管理任务已经在 redshift 中实现了自动化。因此管理起来既简单又便宜。使用 amazon redshift，您可以在几分钟内构建一个 Pb 级的数据仓库，但对于同样的事情，传统的内部实现需要几周或几个月的时间。

**现收现付**

亚马逊红移是一项低成本服务，因此与其他数据仓库解决方案相比，以非常低的成本购买一个 Pb 级数据仓库是非常可能的。价格可能从每小时 0.25 美元开始，然后以每年每 TB 1000 美元的价格扩展到 Pb。红移集群的大小可以根据您的性能规格来确定，只需为您使用的产品付费，并且您可以通过可预测的每月成本来了解您将使用多少产品。新的托管存储可自动扩展您的数据仓库存储容量，而无需您添加和支付额外的计算实例。

**可扩展性**

如果想增加红移中数据库的节点，可以根据需要增加。这是动态的，您不必等待任何硬件或基础架构的采购。无论何时需要，您还可以缩减资源。所以可扩展性成为了亚马逊红移的关键资产之一。此外，由于它可以跨多个可用区域访问，这使得该服务成为高可用性服务。每当你创建或访问红移时，你就在它上面创建了一个集群。您可以为您的集群定义自己的虚拟私有云，并且可以创建自己的安全组来连接到您的集群。这意味着您可以根据自己的需求定义安全参数，并且可以将数据保存在安全的地方。因此，amazon redshift 提供的安全性也是它的一个主要优势。

# **迁移到红移**

每当您需要从传统数据仓库迁移到 amazon redshift 时，您可以根据几个因素来选择迁移策略，例如，数据库及其表的大小、源服务器和 AWS 之间的网络带宽、源系统中的数据更改率、迁移期间的转换和您计划用于迁移和 ETL 的合作伙伴工具，以及迁移切换的步骤数。有几种工具可用于从传统数据仓库到 redshift 的数据迁移。AWS 数据库迁移服务，Attunity，Informatica，SnapLogic，Talend，Bryte 就是其中的几个。

# **理想的时候**

如前所述，amazon redshift 非常适合使用现有 BI 工具进行在线分析处理(OLAP)。组织使用它来运行企业和报告，分析多种产品的全球销售数据，存储历史股票交易数据，分析广告展示和点击，汇总游戏数据，以及分析社会趋势。它还用于衡量医疗保健的临床质量、运营效率和财务绩效。

# 不理想的时候

Amazon Redshift 不适合小数据集(小于 100)，因为它是一个并行处理集群，所以如果数据集很小，您将无法获得它的所有好处。此外，也不建议 OLTP 使用它，因为它是为数据仓库工作负载设计的，因此不适合快速事务。此外，amazon redshift 不适合非结构化数据和 blob 数据(数字图像、视频和音乐)。对于非结构化和 blob 数据，amazon 有更好的解决方案，如 Amazon EMR 和 Amazon s3。

参考资料:

 [## 反模式

### Amazon Redshift 有以下反模式:

docs.aws.amazon.com](https://docs.aws.amazon.com/whitepapers/latest/big-data-analytics-options/anti-patterns-5.html) [](https://aws.amazon.com/redshift/?whats-new-cards.sort-by=item.additionalFields.postDateTime&whats-new-cards.sort-order=desc) [## 亚马逊红移-云数据仓库-亚马逊网络服务

### 了解亚马逊红移云数据仓库。亚马逊红移是一个快速，简单，经济高效的数据仓库…

aws.amazon.com](https://aws.amazon.com/redshift/?whats-new-cards.sort-by=item.additionalFields.postDateTime&whats-new-cards.sort-order=desc)