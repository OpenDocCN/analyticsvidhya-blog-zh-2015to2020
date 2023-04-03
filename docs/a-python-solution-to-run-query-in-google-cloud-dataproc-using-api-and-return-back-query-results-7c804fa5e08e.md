# 使用 API 在 Google Cloud Dataproc 中运行查询并返回查询结果的 Python 解决方案

> 原文：<https://medium.com/analytics-vidhya/a-python-solution-to-run-query-in-google-cloud-dataproc-using-api-and-return-back-query-results-7c804fa5e08e?source=collection_archive---------8----------------------->

本文原载于[大数据日报](https://burntbit.com/a-python-solution-to-run-query-in-google-cloud-dataproc-using-api-and-return-back-query-results/)，也可以在 [Linkedin](https://www.linkedin.com/pulse/python-solution-run-query-google-cloud-dataproc-using-boning-zhang/?published=t) 上找到。源代码见 [Git](https://github.com/BoningZhang/Learning_Airflow/blob/master/common/utils/run_hive_query_gcp.py) 。

在某些情况下，我们需要从本地运行 Google Cloud Dataproc 中的查询并返回查询结果，例如，我们需要根据 Dataproc 表中的查询结果向用户发送一些消息，或者我们需要根据 dataproc 表中的查询结果更新 mysql 表。

如果我们的 hive 表驻留在本地 hadoop 集群中，我们可以使用 python 中的 subprocess 运行 hive/spark-sql 查询，查询结果将以字符串形式从 subprocess 命令返回。因此，我们可以处理字符串结果。该解决方案可以按如下方式实施:

但是没有简单的解决方案可以从 Google Cloud Dataproc API 触发的查询中获取结果，因为 API 调用不会返回查询结果。假设如果我们通过调用 dataproc API 运行“`SELECT * FROM table_a`”，我们无法在 dataproc 作业的输出中获得查询结果。

所以这篇博客将实现一个使用 API 和 output 来运行 dataproc 查询的解决方案。至少有三种解决方案可以做到这一点。

(1)从 hive 表的 HDFS 文件创建外部大查询表，因为大查询交互地运行，这意味着查询结果将包括在大查询作业的输出中。详见[https://cloud.google.com/bigquery/docs/running-queries](https://cloud.google.com/bigquery/docs/running-queries)

(2)如果我们的目标是基于 dataproc 的查询结果更新 mysql 表，那么我们可以通过`JDBC`将 pyspark 数据帧转储到 mysql 表中，并使用 API 将这个 pyspark 作业提交给 dataproc。详见[https://stack overflow . com/questions/46552161/write-data frame-to-MySQL-table-using-py spark](https://stackoverflow.com/questions/46552161/write-dataframe-to-mysql-table-using-pyspark)

(3)使用云存储作为临时空间来转储查询结果。默认情况下，每个 dataproc 作业都会将其日志和输出转储到云存储的文件中。但是由于 dataproc 中的 hive 查询使用的是 Beeline，所以输出文件中会有一些其他的日志信息，因此不容易得到清晰的查询结果。然而，我们可以使用类似的想法，将查询结果覆盖到云存储中的指定文件，然后从云存储中获取内容。本博客的以下部分将详细解释这个解决方案。假设查询结果相对较小，否则我们将不希望运行查询并以字符串形式返回结果，即使在本地 hadoop 集群中也是如此。

我这里用的 python 是`python2.7`，用的是`google.cloud`、`apilcient`、`oauth2client`

`run_hive_query_gcp()`是我们可以使用的函数 run hql，它将以字符串形式返回查询结果。

Google Cloud 文档显示使用环境变量 export `GOOGLE_APPLICATION_CREDENTIALS="/home/user/Downloads/[FILE_NAME].json"`来获得认证

([https://Cloud . Google . com/docs/authentic ation/production # getting _ and _ provide _ service _ account _ credentials _ manually](https://cloud.google.com/docs/authentication/production#obtaining_and_providing_service_account_credentials_manually))，但是如果我们有几个不同的 Google Cloud 项目，就不太方便了，因为我们可能需要为不同的项目设置`GOOGLE_APPLICATION_CREDENTIALS`为不同的 json key 文件。相反，这里我使用了另一个包来设置身份验证，我们可以使用 json 密钥文件作为身份验证函数 ServiceAccountCredentials()的输入参数。更多信息请参见[https://developers . Google . com/analytics/dev guides/config/mgmt/v3/quick start/service-py](https://developers.google.com/analytics/devguides/config/mgmt/v3/quickstart/service-py)

然后，我们向 dataproc 集群提交一个 hive 查询，并等待它的执行。在“`SELECT * FROM table_a`”子句之前，我们添加了 sql 来将查询结果覆盖到云存储中的一个文件。

在执行 dataproc 作业之后，我们使用云存储 API 将结果 blobs 作为字符串下载到云存储中。