# 如何从非规范化的原始数据创建事实表和维度表

> 原文：<https://medium.com/analytics-vidhya/how-to-create-fact-and-dimension-tables-from-denormalized-raw-data-e26127df2249?source=collection_archive---------2----------------------->

在数据仓库领域，有时开发人员必须从平面 csv 文件中逆向工程模型。本文用一个简单的例子解释了这个过程。

**Source**:5 行& 7 列
**CSV 数据库** : PostgreSQL
**托管在** : AWS RDS 自由层

**需求** : psql cli ( [更多信息](http://postgresguide.com/utilities/psql.html))

![](img/ff11e06a85862c9161c0546bf7faab20.png)

Jan Antonin Kolar 在 [Unsplash](https://unsplash.com/s/photos/database?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片