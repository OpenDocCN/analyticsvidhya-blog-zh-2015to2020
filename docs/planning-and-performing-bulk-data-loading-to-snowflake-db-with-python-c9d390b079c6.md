# 使用 python 规划和执行向雪花数据库的批量数据加载

> 原文：<https://medium.com/analytics-vidhya/planning-and-performing-bulk-data-loading-to-snowflake-db-with-python-c9d390b079c6?source=collection_archive---------6----------------------->

![](img/d5cf5b7f49bc7d96f5ad8cf7b4c2884f.png)

# 使用雪花时为什么要使用批量数据加载

所有雪花成本都基于对[数据存储、计算资源和云服务](https://docs.snowflake.com/en/user-guide/admin-usage-billing.html)的使用，因为计算因素可能是最重要的因素。在雪花中，当我们执行查询、加载和卸载数据或执行 DML 时，我们使用[虚拟仓库](https://docs.snowflake.com/en/user-guide/warehouses.html)