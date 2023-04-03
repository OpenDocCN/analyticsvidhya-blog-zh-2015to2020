# 大查询分区和聚类

> 原文：<https://medium.com/analytics-vidhya/bigquery-partitioning-clustering-9f84fc201e61?source=collection_archive---------1----------------------->

![](img/d705cf44476d4c42c66884bcb1cd155d.png)

在这篇博文中，我将解释 BigQuery 中的分区和集群特性，以及如何提高查询性能和降低查询成本。

# 分割

对表进行分区可以使查询运行得更快，同时花费更少。直到 2019 年 12 月，BigQuery 才支持仅使用 **date 数据类型**的表分区。现在，你也可以在整数范围内这样做。如果您想了解更多关于这种表分区方式的信息，请查看 Guillaume Blaquiere 的[博客文章](/google-cloud/partition-on-any-field-with-bigquery-840f8aa1aaab)。

在这里，我将重点介绍**日期类型**分区。您可以使用两种主要策略对数据进行分区:一方面，您可以使用表列，另一方面，您可以使用数据摄取时间。

当您有非常大的多年前的数据集时，这种方法特别有用。事实上，如果您希望只在特定时间段运行分析，那么按时间对表进行分区可以让 BigQuery 只读取和处理特定时间段的行。因此，您的查询将运行得更快，而且因为它们读取的数据更少，所以成本也更低。

创建分区表是一项简单的任务。在创建表时，您可以指定哪个列将用于分区，否则，您可以在[摄取时间](https://cloud.google.com/bigquery/docs/creating-partitioned-tables)设置分区。因为可以用与未分区的表完全相同的方式查询该表，所以不必更改现有查询的任何一行。

```
select
  day,
  count(*)
from full_history
where sampling_date >= ‘2019–08–05’
and sampling_date < ‘2019–08–06’
group by 1
```

假设“ **sampling_date** ”是分区列，现在 BigQuery 可以使用“where 子句”中指定的值来只读属于正确分区的数据。

## 奖金金块

您可以使用**分区装饰器**来更新、删除和覆盖整个单个分区，如下所示:

```
# overwrite single partition loading from file
bq load —-replace \
  project_id:dataset_name.table_name$20190805 \
  gs://my_input_bucket/data/from/20190805/* ./schema.json
```

和

```
# overwrite single partition from query results
bq query —- replace --use_legacy_sql=false \
  —-destination_table project_id:dataset.table$20190805 \
  ‘select * from project_id:dataset.another_table’ 
```

在上述情况下，加载的数据和查询结果都必须属于被引用的分区，否则作业将失败。

# 使聚集

聚类是组织数据的另一种方式，它将在所选的聚类列中共享相似值的所有行一个接一个地存储。这个过程提高了查询效率和性能。注意，BigQuery 只在分区表上支持这个特性。

BigQuery 可以利用聚簇表只读取与查询相关的数据，因此它变得更快、更便宜。

在创建表时，您可以在逗号分隔的列表中提供最多 4 个聚类列，例如“ **wiki** ”、“ **title** ”。你也应该记住它们的顺序是最重要的，但是我们一会儿就会看到。

在这一节中，我们将使用来自 [Felipe Hoffa](https://twitter.com/felipehoffa) 的公共数据集的“ *wikipedia_v3* ”，该数据集包含维基百科页面浏览量的年度表格。这些按“**日期时间**列划分，并聚集在“**维基**和“**标题**列上。单行可能如下所示:

```
datehour, language, title, views
2019–08–10 03:00:00 UTC, en , Pizza, 106
...
```

以下查询统计了从 2015 年 1 月 1 日到 2015 年 1 月 1 日意大利维基的所有页面浏览量。

```
select
  _table_suffix as year,
  wiki,
  sum(views) / pow(10, 9) as Views
from `fh-bigquery.wikipedia_v3.pageviews_*`
where wiki = ‘it’and datehour >= ‘2015–01–01’
group by 1,2
order by 1 asc
```

如果您在 BigQuery UI 中编写这个查询，它将估计 4.5 TB 的数据扫描。但是，如果您实际运行它，最终扫描的数据将只有 160 GB。

这怎么可能？

当 BigQuery 读取时，只读取属于包含意大利语 wiki 数据的集群的行，而丢弃其他所有内容。

为什么列顺序在集群中如此重要？

这很重要，因为 BigQuery 将根据创建表时指定的列顺序分层组织数据。

让我们用下面的例子:

```
select
  wiki,
  sum(views) / pow(10, 9) as Views
from `fh-bigquery.wikipedia_v3.pageviews_2019`
where title = ‘Pizza’
and datehour >= ‘2019–01–01’
group by 1
order by 1 asc
```

这个查询需要访问所有的“wiki”集群，然后它可以使用“title”值跳过不匹配的集群。

这导致扫描比按相反顺序" **title** "、" **wiki** "的聚类列多得多的数据。

在撰写本文时，上面的查询估计扫描成本为 1.4 TB，但它实际上只扫描了 875.6 GB 的数据。

现在让我们颠倒集群列的顺序，将第一个" **title** "和第二个" **wiki** "放在一起，您可以使用以下命令来完成:

```
bq query --allow_large_results --nouse_legacy_sql \
  --destination_table my_project_id:dataset_us.wikipedia_2019 \
  --time_partitioning_field datehour \
  --clustering_fields=title,wiki \
'select * from `fh-bigquery.wikipedia_v3.pageviews_2019`'
```

在我们的新表上运行“Pizza”查询“**my _ project _ id:dataset _ us . Wikipedia _ 2019**”应该会便宜很多。事实上，虽然估计值仍为 1.4 TB，但实际读取的数据仅为 26.3 GB，少了 33 倍。

作为最后的测试，让我们试着过滤一下“ **wiki** 专栏:

```
select
  wiki,
  sum(views) / pow(10, 9) as Views_B
from `my_project_id:dataset_us.wikipedia_2019`
where wiki = ‘it’ and title is not null
and datehour >= ‘2019–01–01’
group by 1
order by 1 asc
```

数据读取估计值始终相同，但现在实际数据读取量跃升至 1.4 TB(整个表)，而在第一个示例中，实际数据读取量仅为 160 GB。

**注意**:由于 BigQuery 使用列存储，“**title not null**”确保我们在每个查询中总是引用相同数量的列。否则，从最后一个查询中读取的数据会更少，因为我们引用的列更少。

显然，选择正确的聚类列及其顺序会产生很大的影响。您应该根据您的工作量来计划。

记住，永远**分区**和**簇**你的表！这是免费的，它不需要改变你的任何查询，它会使他们更便宜，更快。

作者的 [Github](https://github.com/alepuccetti) 和 [Twitter](https://twitter.com/alepuccetti) 。