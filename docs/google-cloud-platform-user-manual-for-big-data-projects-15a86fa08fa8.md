# 谷歌云平台大数据项目用户手册

> 原文：<https://medium.com/analytics-vidhya/google-cloud-platform-user-manual-for-big-data-projects-15a86fa08fa8?source=collection_archive---------9----------------------->

![](img/272bc815ff704759d9c46e75cece53d5.png)

凯利·西克玛在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

> 这个博客也可以在 [Linkedin](https://www.linkedin.com/pulse/google-cloud-platform-user-manual-big-data-projects-boning-zhang) 找到。这个博客最初是供我们团队内部使用的，介绍了我们的谷歌云平台(GCP)项目的架构，我们使用 GCP 的最佳实践，以及获得访问和设置 GCP 项目的指导。现在公司的相关资料被删除了，但我保留了主要内容。

通过这篇博客，您将了解如何访问和创建 GCP 项目，如何管理 GCP 的生产和开发环境，如何创建 dataproc 集群，向 dataproc 集群提交作业，运行大查询作业，将数据从云存储加载到大查询，如何使用 GPU 为深度学习模型创建计算机引擎，以及将数据从本地 Hadoop 集群复制到 google 云存储。

## 1、访问 GCP

首先要[创建一个 GCP 项目](https://cloud.google.com/resource-manager/docs/creating-managing-projects)。

## 2、我们的 GCP 建筑

1.  GCP 项目

根据 GCP [文件](https://cloud.google.com/storage/docs/projects)。
“一个项目组织你所有的谷歌云平台资源。一个项目由一组用户组成；一组 APIs 以及这些 API 的计费、身份验证和监控设置。因此，例如，您的所有云存储桶和对象，以及访问它们的用户权限，都驻留在一个项目中。你可以有一个项目，也可以创建多个项目，用它们把你的谷歌云平台资源，包括你的云存储数据，组织成逻辑组。”

在我们的 GCP 客户中，我们有三个云项目，每个团队成员都应该至少可以访问 dev 和 stg 项目。

gcpExploration，项目 ID: xxxx-stg。这个项目是用来探索 GCP 的新特色。

GCP 开发，项目 ID: xxxx-dev。这是我们的开发项目，用于数据管道和机器学习模型的开发和测试，也用于运行临时作业。

GCP 生产，项目 ID: xxxx-prd。这是我们的生产项目，所有的生产管道和模型都应该在这个项目中运行。

2.访问控制

我们的团队中有两个 gcp 组:gcp-user 和 gcp-admin。gcp 项目的访问控制是基于 gcp 组来操作的:gcp -admin 中的团队成员拥有对三个项目的所有访问权限，而 gcp-user 中的团队成员没有对生产项目的访问权限。生产管道和机器学习模型被设置为通过服务帐户而不是每个单独的团队成员的帐户来访问生产项目。

在这里，拥有对项目的访问权意味着您拥有对项目中所有已启用组件的完全访问权，比如您可以创建和删除 dataproc 集群/计算机引擎，创建和删除大型查询数据集和表，以及对与该项目相关联的云存储执行写/读操作。

3.数据存储

我们在 GCP 有 4 个数据库，它们的云存储路径应该在例程中，如 GS://GCP _ project _ id-warehouse/database . db/。
a. project1:用于 project1 相关表，其在生产项目中的云存储路径为:GS://xxxx-prd-warehouse/project 1 . db/

b.project2:用于 project2 相关表，其在生产项目中的云存储路径为:GS://xxxx-prd-warehouse/project 2 . db/

c.cmn:用于常用表，如所有 project1 和 project2 项目都使用的 uber 表，其在生产项目中的云存储路径为:GS://xxxx-prd-warehouse/cmn . db/

在 gcp 中，建议将所有数据和 hdfs 文件存储在相应项目的云存储中，即生产数据应驻留在生产项目的云存储中:gs://xxxx-prd-warehouse/。和测试/特别数据应该驻留在开发项目的云存储中。我们项目中的所有 dataproc Hadoop 集群共享同一个元存储，这意味着在不同项目的不同 Hadoop 集群中定义的表是跨项目共享的。dev 和 stg 项目可以对 prod 项目中的云存储进行读访问。因此，dev/stg 项目中的测试/特别作业可以使用 prod 项目中定义/驻留的表和数据。但是，不允许 dev/stag 项目中的 test/adhoc 作业写入 prod 项目中的云存储，这些测试表的数据应该驻留在 stag/dev 的云存储中。

使用云存储中存储的数据创建表的示例:

4.访问其他团队的项目

在我们协作团队的 gcp 项目中，我们的开发和生产项目都具有对云存储桶的读取权限。这意味着我们可以在我们的项目中的大查询/hadoop 集群中操作他们云存储中的数据。他们项目中的 dataproc 集群也使用与我们相同的元存储，因此我们可以像在本地集群中一样直接读取他们项目中的表。

5.从内部访问 GCP

本地集群需要使用服务帐户连接到 GCP，然后我们可以在本地集群中运行作业并将结果转储到云中，或者我们可以创建和处理数据实际驻留在本地集群的云存储中的表。

## 3、安装 GCP SDK 并设置 gcloud config

为了在终端中使用命令行连接到 gcp 项目，我们需要安装 GCP SDK 并设置配置。实际上，使用 GCP 控制台对用户来说更方便，所以这一步是不必要的。在开始下面的步骤[安装 GCP SDK](https://cloud.google.com/sdk/docs/quickstart-macos) 之前，我们需要确保 python2.7 已经安装在我们的笔记本电脑上。

1.下载并安装 Google SDK

a.点击下载 mac os 64 位的 SDK[b .解压 tar.gz 文件并运行。/谷歌-云-sdk/install.sh](https://cloud.google.com/sdk/docs/quickstart-macos)

2.设置第一个 GCP 项目配置

a.运行命令:gcloud init
b .在如下提示选择云项目时，选择[1]然后进入 gcpStaging

c.选择默认地区/区域:region=us-central1 和 zone=us-central1-a

3.设置多个项目配置

运行“gcloud config 配置列表”,您将看到只有 stg 项目已配置，我们需要[为其他两个项目设置配置](https://cloud.google.com/sdk/gcloud/reference/topic/configurations)。
a .“g cloud config configuration create dev-config”创建一个新的空配置并将其激活
b .“g cloud init”通过选择“[1]使用新设置重新初始化此配置[dev-config]”来设置配置
c .新配置设置完成后，我们可以使用“g cloud config configuration activate stag-config”在不同项目之间切换
d .我们还可以使用“g cloud config configuration delete[config name]”来删除项目配置

## 4、云存储

云存储独立于任何计算机引擎，这意味着删除计算机引擎不会删除云存储中的数据。云存储也是 Hadoop 兼容的文件系统，可以作为 Hadoop 集群的存储。

至少有三种方法可以访问云存储:

1.GCP 控制台

我们可以创建/删除存储桶，上传/删除文件

2.gsutil 命令行

参见:[https://cloud.google.com/storage/docs/quickstart-gsutil](https://cloud.google.com/storage/docs/quickstart-gsutil)
a . gsutil MB，创建存储桶
b. gsutil cp，将本地文件复制到存储
c. gsutil ls，列出存储桶中的文件
d. gsutil cat，显示存储文件

## 5、使用计算机引擎的最佳实践

因为计算机引擎是根据我们请求的资源计费的，即使我们不使用虚拟机，我们也会不断地被收费。要停止被收费，我们需要删除计算机引擎。所以建议**我们把重要的输入输出数据和代码保存在云存储**里。建议直接从云存储桶读取输入，在代码中直接写到云存储。如果不可能，我们至少应该每天备份云存储/笔记本电脑中的数据和代码，尤其是在删除虚拟机之前。

1.使用 GPU 设置深度学习计算机引擎的步骤

a.首先使用“g cloud config configuration activate[config _ name]”切换到托管计算机引擎的项目

b.使用 gcloud 命令行创建计算机引擎，您可能需要为您公司的项目指定子网

c.转到 GCP 控制台->计算机引擎获取虚拟机的内部 IP，并将您的 ssh 密钥添加到虚拟机。

d.设置 jupyter 笔记本:

## 6、GCP Dataproc 上的 Hadoop 集群

GCP 的 Hadoop 集群使用计算机引擎作为主节点和工作节点，因此 Hadoop 集群在请求的资源上收费，即 CPU、内存和磁盘。所以建议使用按需 Hadoop 集群。对于一些临时作业，生产项目现在托管一个小型永久 Hadoop 集群，如果需要，可以进行扩展(如果需要，还可以在开发项目中创建永久 Hadoop 集群)。对于生产管道/模型，实践是在 airflow dags 中使用 gcloud api 创建 Hadoop 集群，然后在这个 Hadoop 集群中运行作业。当这些作业完成后，应该会删除 airflow dags 中的 Hadoop 集群。在 airflow dag 中使用按需 hadoop 集群的示例可以在[这里](https://github.com/BoningZhang/Learning_Airflow/blob/master/DAGS/example_on_demand_dataproc/dag.py)找到。使用这种实践，我们观察到由 GCP dataproc 引起的花费的快速减少。

1.向 Hadoop 集群提交作业

向 Hadoop 集群提交临时作业至少有三种方式。

a.在 GCP 控制台，选择 Dataproc →选择作业→点击提交作业
地区:us-central1
集群:选择我们的集群名称，即 hadoop-cluster
作业类型:可以是 hadoop、pyspark、spark-sql、hive 等等

b.使用 gcloud SDK，确保激活正确的 GCP 项目，即 Hadoop 集群应该在激活的项目配置中

示例:gcloud dataproc 作业提交配置单元—集群 hadoop 集群—执行“显示数据库”—美国中部地区 1

c.登录到 Hadoop 集群中的虚拟机实例，然后在那里提交作业。不建议这样做。

2.在气流中运行 Hadoop 集群

Airflow 使用 gcloud python 包和服务帐户与 GCP 的 Dataproc Hadoop 集群进行交互，API 调用封装在 operators 中。我们经常使用的 Dataproc 操作符主要有三种:DataprocClusterCreateOperator、DataProcHiveOperator、DataprocClusterDeleteOperator。这些操作符经过修改后用于我们的 GCP 项目，我们的 GCP 项目和我们的气流平台之间的连接需要使用 GCP 服务帐户进行设置。

a.DataprocClusterCreateOperator

b.DataProcHiveOperator

c.DataprocClusterDeleteOperator

您可能会注意到，我使用“Hadoop-cluster-KPI-pacing-ratio-{{ ds }}”作为 cluster_name，这是因为我们需要确保 cluster_name 在不同 dag 之间以及同一 dag 的不同 dag 运行之间是唯一的，示例 Dag 是每日计划的，因此{ { ds } }是可以的。否则，airflow 任务实例可能会删除正在其他 dag 运行中运行作业的群集。我们也可以使用 uuid 来生成唯一的 dataproc 名称，但是在 dataproc 名称中包含 dag 信息将有助于监控作业。

3.在 Hadoop 集群中调试作业

在 Dataproc 中，选择 Jobs，您将会看到这个项目中完整的作业列表。通过单击作业，我们可以看到它的日志和输出。

## 7、大查询

对于临时作业，建议使用 Big Query 运行查询，它非常快。大查询中有两种类型的表，一种是原生表，另一种是外部表。本地表的数据驻留在 Big Query 的存储中，这与云存储不同。外部表的数据驻留在其他外部源中，比如云存储。本机表的处理性能比外部表快。但是外部表的优点是它不需要额外的存储资源，更重要的是查询结果总是基于外部源数据的最新结果。但是，如果我们将 HDFS 文件作为原生表加载到 Big Query 的云存储中，那么一旦云存储中的数据更新，我们就需要重新加载它。

实际上，我们使用 parquet 格式将数据存储在云存储中，这样可以节省大量的存储空间。而目前大查询不支持 parquet 格式作为外部数据源，因此我们必须在大查询中创建原生表，并将云存储中的数据加载到原生表中。我们应该记住，一旦数据更新，我们需要从云存储中重新加载数据。

大查询中的分区概念与 Hive 中的不同。此外，由于 Big Query 针对处理大型数据进行了优化，因此我们可以在一个大型查询表中加载多个分区的数据。例如，dctr 表现在是基于每日分区的，但是在大型查询中，我们可以在单个大型查询表中加载一个月甚至更多天的数据。请注意，在加载到大查询表之后，分区字段将不在大查询的表模式中，因为在 hive 表的 HDFS 路径中没有分区列。

大查询至少有三种操作方式，其中使用 GCP 控制台更方便。目前，我们在开发和生产项目中有四个数据集。请在相应的数据集中创建大的查询表。大查询中的数据集类似于 Hive 中的数据库。

1.通过 GCP 控制台操作大型查询

a.将一个月的 dctr 数据从云存储加载到大查询，可能需要大约 2 分钟

创建表来自:Google 云存储
从 GCS 桶中选择文件:gs_path
文件格式:Parquet
项目名称:wmt-customer-tech-case-sci-dev
数据集名称:casesci_sem
表类型:原生表

b.在大查询中处理数据，大查询的 SQL 与 Hive 中的非常相似

请注意“CAST(date_string AS STRING)”，因为在 parquet 格式中字符串类型存储为二进制以节省空间。在执行细节中，我们可以看到大查询处理 150MB 数据只需要 1.2 秒。

c.写入新表，在 query_settings 中，选择“为查询结果设置目标表”

2.通过 bq 命令行操作大型查询

a.bq 命令。例如，bq ls project_id，bq show project _ id:data _ set . table _ name，bq rm -t mydataset.mytable

b.将一个月的 dctr 数据从云存储加载到大查询，确保在运行命令之前使用正确的项目配置

c.处理大型查询表

d.写入新表

3.通过 Google Cloud API 操作大查询

## 8、将数据从本地 Hadoop 集群中的 BFD 集群迁移到 GCP

目前，我们不打算将大量历史数据从本地 Hadoop 集群迁移到 Google 云存储，这可以通过在 BFD 集群中运行分布式拷贝作业来完成。但是从本地集群到 GCP 的连接需要使用 GCP 的服务帐户来设置，并且我们每天都将一些最终生产表从本地集群迁移到 GCP。

1.将每日分区导出到 GCP

a.通过将查询结果转储到指向 Google 云存储的外部表中，导出本地集群中的数据

b.在 GCP 创建指向 Google 云存储数据的表

2.将大量历史数据迁移到 GCP

目前，我们不打算将大量历史数据从 CDC 迁移到 Google 云存储，这可以通过在 BFD 集群中运行分布式复制作业(distcp)来完成。我将构建一个仪表板，允许用户将大量历史数据从本地 Hadoop 集群复制到 Google 云存储中。