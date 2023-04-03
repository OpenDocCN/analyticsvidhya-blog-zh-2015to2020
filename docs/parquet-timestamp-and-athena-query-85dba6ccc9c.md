# 拼花时间戳和雅典娜查询

> 原文：<https://medium.com/analytics-vidhya/parquet-timestamp-and-athena-query-85dba6ccc9c?source=collection_archive---------1----------------------->

在这篇博客中，我将向您介绍时间戳在 Parquet 文件版本 1.0 和 2.0 中的存储方式，如何在 Athena 中显示每个版本的时间戳列数据，以及如何在 Athena 中转换时间戳列来查看版本 2.0 Parquet 文件的时间戳。

使用 AWS Glue crawler，我抓取了由 RDS 快照到 S3 功能创建的存储在 S3 的几个拼花文件。在 crawler 完成并添加新表之后，我使用 AWS Athena 查询数据，时间戳列显示数据如下