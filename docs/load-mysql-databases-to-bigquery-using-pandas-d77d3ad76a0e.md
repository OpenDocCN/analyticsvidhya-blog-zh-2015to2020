# 使用 Pandas 将 MySQL 数据库加载到 BigQuery 中

> 原文：<https://medium.com/analytics-vidhya/load-mysql-databases-to-bigquery-using-pandas-d77d3ad76a0e?source=collection_archive---------3----------------------->

## 简单的方法，避免模式错误、数据类型问题和长期头痛

![](img/db2c7bacaf3323d608b6ed4d75dfe1f3.png)

熊猫在吃东西(鸣谢:[https://unsplash.com/@mana5280](https://unsplash.com/@mana5280)

# 问题是

您可能尝试过将 MySQL 或 PostgreSQL 之类的关系数据库加载到 BigQuery 之类的列式数据库系统中，即使这是一种标准且被广泛采用的格式，过程也并不明显。

如果您遇到类似以下的错误:

> 读取数据时出错，错误消息:CSV 表遇到太多错误，放弃。行数:x；错误:1。有关更多详细信息，请查看 errors[]集合。

或者

> CSV 表引用了 x 列位置，但从位置:0 开始的行只包含 x 列。(错误代码:无效)

或者再一次…

> 分析从位置 x 开始的行时检测到错误。错误:右双引号(")和字段分隔符之间的数据。

你绝对是来对地方了！😎

有些解决方案是存在的，你可以在 Stackoverflow 或互联网上找到它们。他们中的大多数建议您使用封闭方法在 CSV 中进行特定的导出，或者在换行符分隔的 JSON 中转换所有内容。

Google 还提供了将 MySql 加载到 BigQuery 的文档，但是这个过程需要为每个表提供一个模式，并使用数据流来执行管道。另一个解决方案是使用 Cloud Dataprep，它可以在接收数据之前帮助转换和清理数据。

这些方法效果很好，但有以下限制:

*   当您有数百个表时，为每个表提供一个模式可能会成为一个耗时的编写和更新过程。此外，如果关系数据库模式发生变化，动态管理会更加困难。
*   使用纯 SQL 命令导出 CSV 格式的数据需要开发一个 bash 脚本，并且获取表头通常会变得很复杂。
*   将 CSV 表转换为 NDJSON 可以修复字段分隔符，但不能修复数据类型错误。
*   使用数据流意味着设置许多步骤，使用谷歌存储，提供模式，然后映射/连接和转换每个字段。这个过程更加精确，但是也非常耗时，并且需要 Apache Beam Python SDK 方面的知识。

以下是已知的现有解决方案列表，请随意尝试并使用最适合的解决方案:

*   [https://cloud . Google . com/solutions/performing-ETL-from-relational-database-into-big query](https://cloud.google.com/solutions/performing-etl-from-relational-database-into-bigquery)
*   [https://stack overflow . com/questions/41774233/best-practice-to-migrate-data-from-MySQL-to-big query](https://stackoverflow.com/questions/41774233/best-practice-to-migrate-data-from-mysql-to-bigquery)
*   [https://medium . com/Google-cloud/loading-MySQL-backup-files-into-big query-straight-from-cloud-SQL-d40a 98281229](/google-cloud/loading-mysql-backup-files-into-bigquery-straight-from-cloud-sql-d40a98281229)

# 解决方案

使用 *Pandas* (一个用于数据分析的开源 python 库)可以很容易地解决这个问题，而且最棒的是，它甚至不强制提供模式，你也不需要使用谷歌存储。用几行代码编写脚本、运行和部署也很容易。

更重要的是，这种技术允许您加载到 BigQuery，或者使用头文件、特定分隔符、编码等创建 CSV 转储文件！

下面的脚本中列出了这两种方法！😇

假设您已经:

*   在 BigQuery 中配置的项目和数据集(您需要您的项目 ID 和数据集 ID)
*   直接连接到您的 MySql 数据库
*   你的。用于访问 GCP 的 json 服务帐户文件

以下是 gist 的完整代码，我们将一步一步解释:

[https://gist . github . com/Romain 9292/89 ba 7 cbef 0715d 07 c 8198 a5ff 1347805](https://gist.github.com/romain9292/89ba7cbef0715d07c8198a5ff1347805)

# 步骤 1:声明变量

```
# Service account file for GCP connection
credentials = service_account.Credentials.from_service_account_file('key.json')#  BigQuery Variables
PROJECT_ID = 'your_project_ID'
DATASET_ID = 'your_data_set_ID'#  MySql Variables
MYSQL_USERNAME = 'root'
MYSQL_PASSWORD = 'root'
MYSQL_HOST = '127.0.0.1'
MYSQL_DATABASE = 'your_table'# Path for the dump directory
DIRECTORY = 'dump'
```

确保您的 *service-account.json* 文件在您的 python 项目目录中。

然后，您可以提供您的 BigQuery 和 MySql 连接细节。最后但并非最不重要的一点是，需要一个特定的目录来将所有表写成 CSV 文件。

# 步骤 2:列出数据库中的所有表

```
tables_query = 'SELECT table_name ' \
               'FROM information_schema.tables ' \
               'WHERE TABLE_TYPE = "BASE TABLE" ' \
               'AND TABLE_SCHEMA = "{}";'.format(MYSQL_DATABASE)
```

使用 SQL 查询，我们可以从数据库中选择所有表的名称。

# 步骤 3:上传到 BigQuery

```
for index, row in list_tables.iterrows(): table_id = '{}.{}'.format(DATASET_ID, row['TABLE_NAME']) print 'Loading Table {}'.format(table_id)
    df = pd_mod.read_sql_table(row['TABLE_NAME'], engine)

    pd_gbq.to_gbq(df, table_id,
                  project_id=PROJECT_ID,
                  if_exists='replace',
                  chunksize=10000000,
                  progress_bar=True)
```

这是脚本中最有趣的部分(您可以在几行代码中看到它)。首先，我们迭代我们的数据框架以获得存储在*行['TABLE_NAME']* 中的每个表的名称，然后我们使用 *pd_mod.read_sql_table* 函数，它的作用类似于*SELECT * FROM TABLE _ NAME；*获取完整的表格数据。

我们使用*熊猫摩丁*来加速这部分过程。这个库允许我们透明地分发数据并并行运行计算。

此时，SQL 表存储在名为 *df* 的数据框中，该数据框将使用 pandas-gbq 模块加载到 BigQuery，该模块是 Google big query 的包装器，简化了检索/推送数据的过程。您还可以微调区块大小参数，以加快进程并优化 IO 成本。

# 步骤#4:转储到 CSV

```
for index, row in list_tables.iterrows(): table_id = '{}.{}'.format(DATASET_ID, row['TABLE_NAME']) print 'Loading Table {}'.format(table_id)
    df = pd_mod.read_sql_table(row['TABLE_NAME'], engine) pd_gbq.to_gbq(df, table_id,
                  project_id=PROJECT_ID,
                  if_exists='replace',
                  chunksize=10000000,
                  progress_bar=True) print 'Exporter les  tables en csv'
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY) df.to_csv('{}/{}.csv'.format(DIRECTORY, row['TABLE_NAME']),
              index=None,
              header=True)
```

正如你所看到的，这是与第 3 步相同的代码，但是我们添加并使用了这个 *to_csv* 方法，它将把本地存储中的每个表写入 csv 文件。由于这种方法，您可以利用 *Pandas* 库的所有功能和提供的所有特性，如编码、压缩、块大小、索引、缺失数据等…

# 现在，快乐的 SQL！

这种简单的方法可以集成到完全生产就绪的数据接收管道中。

我希望这种方法可以帮助您节省一些时间，并修补您与 BigQuery 中的关系数据库之间的问题！

我还要感谢我非常欣赏的同事和朋友 Adrien，他在这个问题上与我一起工作。

是时候深入研究一些数据了！🤓