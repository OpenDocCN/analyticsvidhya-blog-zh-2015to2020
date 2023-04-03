# 云中的查询:通过 R 和 Python 使用 BigQuery

> 原文：<https://medium.com/analytics-vidhya/querying-in-the-cloud-using-bigquery-with-r-and-python-576541554967?source=collection_archive---------6----------------------->

![](img/f36235c47f9cc7e7433dc02abd155fbc.png)

在这里了解更多关于 Google 的无服务器、高可伸缩、快如闪电的数据仓库: [**BigQuery**](https://cloud.google.com/bigquery/what-is-bigquery) 。

在本文中，我们将研究如何使用 R 和 Python 访问和查询存储在 BigQuery 中的数据。

# 使用 R

我们将使用由 Hadley Wickham 创建的`[bigrquery](https://github.com/r-dbi/bigrquery)`包，它提供了一个非常简单的…