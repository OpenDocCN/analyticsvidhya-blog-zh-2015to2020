# Azure Synapse Analytics —构建数据仓库时的关键考虑事项

> 原文：<https://medium.com/analytics-vidhya/azure-synapse-analytics-key-considerations-while-building-your-data-warehouse-a54ad1804139?source=collection_archive---------14----------------------->

Azure Synapse Analytics 的最佳实践集合在一个位置进行汇编，以供快速参考。

![](img/725cc63ef293ee80e38be2472e0a4f23.png)

图片由 [Ag Ku](https://pixabay.com/users/myrfa-3126475/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1614223) 来自 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1614223)

Azure Synapse Analytics(以前的 SQL Datawarehouse)提供了数 Pb 的扩展能力，并统一了企业数据仓库和大数据分析。

Synapse SQL 使用 MPP(大规模并行处理)架构，利用 SQL 池，SQL 池是分析资源的集合，池的大小由 DWU(数据仓库单位)决定，这也决定了定价。SQL Synapse 还提供了暂停计算并仅在需要时保持存储活动的能力。

一个表中的数据被分成 60 个分布，分布策略可以是**循环、散列分布**或**复制。**然后，根据所需的性能水平，计算节点的数量可以从 **1** 到 **60** 不等。随着计算节点或 DWU 数量的增加，每个计算节点的分配会减少，从而提高整体性能。

Synapse SQL 的文档非常详尽，可以在 [h *ere*](https://docs.microsoft.com/en-us/azure/synapse-analytics/sql-data-warehouse/) *，*中找到，完整的 pdf 文件超过 450 页。我整理了下面的链接，这些链接帮助我理解了设计数据仓库时的最佳实践和考虑事项，作为快速参考指南。

# 备忘单

*   [https://docs . Microsoft . com/en-us/azure/SQL-data-warehouse/cheat-sheet](https://docs.microsoft.com/en-us/azure/sql-data-warehouse/cheat-sheet)

# 最佳实践

*   [https://docs . Microsoft . com/en-us/azure/SQL-data-warehouse/SQL-data-warehouse-best-practices](https://docs.microsoft.com/en-us/azure/sql-data-warehouse/sql-data-warehouse-best-practices)

# 设计表格

*   [https://docs . Microsoft . com/en-us/azure/SQL-data-warehouse/SQL-data-warehouse-tables-overview](https://docs.microsoft.com/en-us/azure/sql-data-warehouse/sql-data-warehouse-tables-overview)

# 索引表

*   [https://docs . Microsoft . com/en-us/azure/synapse-analytics/SQL-data-warehouse/SQL-data-warehouse-tables-index](https://docs.microsoft.com/en-us/azure/synapse-analytics/sql-data-warehouse/sql-data-warehouse-tables-index)

# 很

*   [https://docs . Microsoft . com/en-us/SQL/relational-databases/indexes/heaps-tables-without-clustered-indexes？view=sql-server-ver15](https://docs.microsoft.com/en-us/sql/relational-databases/indexes/heaps-tables-without-clustered-indexes?view=sql-server-ver15)

# 聚集列存储索引

*   [https://docs . Microsoft . com/en-us/SQL/relational-databases/indexes/column store-indexes-overview？view=sql-server-ver15](https://docs.microsoft.com/en-us/sql/relational-databases/indexes/columnstore-indexes-overview?view=sql-server-ver15)
*   [https://docs . Microsoft . com/en-us/SQL/relational-databases/indexes/column store-indexes-overview？view = SQL-server-ver 15 # can-I-combine-row store-and-column store-on-same-table](https://docs.microsoft.com/en-us/sql/relational-databases/indexes/columnstore-indexes-overview?view=sql-server-ver15#can-i-combine-rowstore-and-columnstore-on-the-same-table)

# 设计分布式表——循环法、散列法和复制法

*   [https://docs . Microsoft . com/en-us/azure/SQL-data-warehouse/SQL-data-warehouse-tables-distribute](https://docs.microsoft.com/en-us/azure/sql-data-warehouse/sql-data-warehouse-tables-distribute)
*   [https://docs . Microsoft . com/en-us/azure/SQL-data-warehouse/SQL-data-warehouse-tables-overview # common-distribution-methods-for-tables](https://docs.microsoft.com/en-us/azure/sql-data-warehouse/sql-data-warehouse-tables-overview#common-distribution-methods-for-tables)

> 哈希分布—对具有聚集列存储索引的大型事实数据表使用这种分布方法。
> 
> 复制—对较小的表使用这种分发策略(<2GB), which can be replicated to each compute node, typically used for Dimension Tables.
> 
> Round-Robin — Use this strategy for staging tables or for loading data or when there is no clear choice for distribution.

# Load data from external tables using polybase

*   [https://docs . Microsoft . com/en-us/azure/SQL-data-warehouse/load-data-from-azure-blob-storage-using-poly base # load-the-data-into-your-data-warehouse](https://docs.microsoft.com/en-us/azure/sql-data-warehouse/load-data-from-azure-blob-storage-using-polybase#load-the-data-into-your-data-warehouse)

我希望这篇文章是有用的，能够为数据工程师和架构师在处理 Synapse SQL 仓库时提供一个有用的起点。