# 气流子 Dag 将数据从本地 Hadoop 集群同步到 Google 云存储

> 原文：<https://medium.com/analytics-vidhya/an-airflow-sub-dag-to-sync-data-from-on-premise-hadoop-cluster-to-google-cloud-storage-230783fcf80a?source=collection_archive---------6----------------------->

这个博客也可以在 [Linkedin](https://www.linkedin.com/pulse/airflow-sub-dag-sync-data-from-on-premise-hadoop-cluster-boning-zhang) 找到。

我们现在正在将数据从本地 Hadoop 集群迁移到谷歌云平台(GCP)。由于我们的团队中有许多数据管道，以及对上游数据的许多内部和外部依赖，我们无法避免在内部 Hadoop 集群和 Google 云存储(GCS)之间来回复制数据的情况。例如，我们有一个非常关键和时间敏感的管道，向搜索引擎提交投标，我们希望将整个管道迁移到 GCP。但是，它有一些上游依赖项，还没有迁移到 GCP。因此，一旦新填充的上游数据准备就绪，我们需要首先将它们定期从本地集群同步到 GCS，这样我们的投标渠道就可以完全在 GCP 运行。还有许多其他使用案例，我们需要定期将数据从本地集群同步到 GCP。这个博客为此引入了一个气流子图，它封装了细节，可以很容易地被其他团队成员使用。

让我首先提供一个使用这个子 dag 来加载数据以显示其接口的例子。整个例子可以在 [Github](https://github.com/BoningZhang/Learning_Airflow/blob/master/DAGS/sync_data_to_gcp/dag.py) 中找到。

在本例中，我们将两个表从本地集群导出到 GCP。它们在字典中有定义。

这里的“hql_dict”是我们需要提供给 sub-dag 的查询，它们将在那里运行。稍后我会详细介绍它们。

这个“export_table_params”是我们将在“hql_dict”的查询中使用的 jinja 参数。

export_table_dataproc_config 是 gcp 的 dataproc 的信息:“dataproc_cluster”是我们将用于在 GCP 创建表和加载数据的 dataproc 的名称，“region”是 dataproc 的区域，“gcp_conn_id”是与 GCP 项目的气流连接，可以按照[指令](https://airflow.apache.org/docs/stable/howto/connection/gcp.html)进行设置。

这是为了创建任务，通过应用我们定义的子 dag export_to_gcp_dag 来同步这两个表。如您所见，这三个参数(export _ table _ dict[op _ name][" hql _ dict "]，export_table_params，export_table_dataproc_config)在子 dag 中传递。

现在，让我向您展示这个子 dag 是如何实现的，以及它将如何将数据从本地数据集群同步到 GCP。sub-dag 的实现可以在 [Github](https://github.com/BoningZhang/Learning_Airflow/blob/master/common/operators/export_on_premise_to_gcp_dag.py) 中找到。

此子 dag 主要运行三个查询，“export_data_hql”在本地 Hadoop 集群中运行。它在位于 Google 云存储中的本地集群中创建一个表，并将原始表中的查询结果插入到这个新表中。完成此操作后，本地集群中的数据将加载到 GCS 中。请注意，这里我们使用 GCP 的服务帐户授权本地集群访问 GCS。此外，我们在表名中使用{ { params . gen _ date _ str _ nodash(execution _ date)} }，这样我们可以为同一个表运行多个具有不同日期分区的实例。

第二步，“add_dataproc_partition_hql”将在 GCP 的 dataproc 集群中运行，以在 GCP 创建目标表，并将 GCS 中的数据加载到该表中。

之后，我们运行第三个查询“drop_tmp_table_hql”来删除本地 Hadoop 集群中的临时表。

如果进行了配置，将调用 bash 命令为 GCS 中的分区生成 stamp 文件。