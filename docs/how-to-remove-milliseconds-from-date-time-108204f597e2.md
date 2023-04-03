# 如何从日期时间中删除毫秒

> 原文：<https://medium.com/analytics-vidhya/how-to-remove-milliseconds-from-date-time-108204f597e2?source=collection_archive---------3----------------------->

让我们使日期时间分析数据库友好。

![](img/bca7f4625028fc5bd19aae29220bacac.png)

img src: unsplash

从日期时间戳中删除毫秒可能有几个原因。

第 101 个原因可能是，您的分析软件(如“Clickhouse ”)在尝试创建分区时不喜欢毫秒。