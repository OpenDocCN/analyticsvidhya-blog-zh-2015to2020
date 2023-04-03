# 弹性搜索中的数据搜索基础

> 原文：<https://medium.com/analytics-vidhya/basics-of-data-search-in-elasticsearch-cda7fde50849?source=collection_archive---------1----------------------->

2018 年 8 月 4 日上午 7:02:21 | 5 分钟

![](img/775112f0fa321c90aa2eb537d6f490ba.png)

随着我们迈向数字时代，将事物转变为物联网，数据量与日俱增。举一个简单的智能手表的例子，它测量步数、血压、心率等，并推送到服务器，从那里我们可以访问我们的健康相关指标。同样，有不同的智能设备不断发送存储在服务器上的常规数据。我们在服务器上倾倒大量的数据，这些数据可以帮助我们找到趋势，通过数据科学分析它们以解决一些严重的问题，或者应用机器学习算法来预测未来的趋势。

因此，简单地说，我已经解释了数据的重要性，我们应该定期获取数据，从中获取有意义的信息。现在问题来了。搜索数据的最佳方式是什么？传统上，我们将数据存储在 RDBMS 系统中，并通过直接应用 SQL 查询来获取数据，但现在情况发生了变化，因为我们需要快速的搜索响应，我们没有时间通过查看加载图标的移动方向来等待搜索结果。我们面临的另一个问题是数据格式的不确定性，对于这些类型的数据，我们在 RDBMS 系统中遇到了瓶颈。

现在转到搜索部分，因为这个博客在这里向你介绍基本的 Elasticsearch 查询结构，这样一个新手就可以安装、索引和搜索 Elasticsearch 集群中的数据。如今，Elasticsearch 主要用于其搜索功能和 ELK 堆栈，后者可以应用于任何应用程序集，以提高性能和监控功能。所以让我们开始这个过程，学习 Elasticsearch 中的基本搜索查询构造。

在 Elasticsearch 中，我们基本上有两种类型的搜索 API:“基于请求 URI”和“基于请求正文”。在 REST 请求 URI 中，我们使用 URL 本身来传递搜索标准，例如:

```
GET /blogs/technical/_search?q=topic:kibana
```

在 REST 请求体中，我们使用来构造搜索块，并在 Elasticsearch 的查询块中编写搜索查询，如下所示:

因此，基于 URI 的搜索是一个非常基本的搜索，我们只想搜索一个关键字，而在请求正文中，我们可以构建复杂的查询。所以我们有了查询语言来处理基于请求体的搜索。在这篇博客中，我不打算深入细节，以保持简单，让每个人都可以理解正在发生的事情。

默认情况下，在 Elasticsearch 中文本字段上禁用 Fielddata，因此我们需要启用它来构造查询。

```
PUT blogs/_mapping/technical?update_all_types
{
 "properties": 
  {
     "topic": 
     { 
       "type": "text",
       "fielddata": true
     }
  }
}
```

现在让我们了解查询语言的基础，首先是 match_all 查询:

在 match_all 查询中，Elasticsearch 返回所有文档。所以这个 Elasticsearch 查询基本上就像 SQL“select * from technical”查询。

**极限:**

现在，我们将在查询中设置偏移和限制来限制记录，如下所示:

在上面的查询中，我从第二个开始获取 5 个文档。同样，我们可以在任何 Elasticsearch 查询中设置偏移和限制。

在 Elasticsearch 中，我们可以按照我们的要求对文档进行分类，例如:

在上面的表达式中，我们对字段主题应用了排序。

我们限制 SQL select 查询中的列数，同样，我们也可以在 Elasticsearch 查询中这样做，比如:

```
GET /blogs/technical/_search
{
   "query": { "match_all": {} },
   "_source": ["category"]
}
```

在上面的查询中，我们将只在搜索结果中得到类别字段，而主题字段不会显示。

我们可以对字段名运行匹配查询，比如:

在上面的查询中，我们可以传递文本来搜索主题字段。

在这篇博客中，我解释了 Elasticsearch 查询构造的基础。在下一篇博客中，我将介绍过滤器、布尔查询、通配符查询等，然后解释聚合及其用法。

你可以在推特上关注我:[https://twitter.com/anubioinfo](https://twitter.com/anubioinfo)

**关于 Elastic stack 的其他博客:** [Elastic Search 简介](https://bqstack.com/b/detail/31/Introduction-to-Elasticsearch)
[Elasticsearch 在 Ubuntu 14.04 上的安装和配置](http://bqstack.com/b/detail/52/Elasticsearch-Installation-and-Configuration-on-Ubuntu-14.04)
[使用 Elastic Stack 的日志分析](http://bqstack.com/b/detail/1/Log-analysis-with-Elastic-stack) [Elastic Search Rest API](https://bqstack.com/b/detail/83/Elasticsearch-Rest-API)
[Elastic Search 中数据搜索的基础知识](https://bqstack.com/b/detail/84/Basics-of-Data-Search-in-Elasticsearch)
[Elastic Search Rest API](https://bqstack.com/b/detail/83/Elasticsearch-Rest-API)
[Elastic Search](https://bqstack.com/b/detail/87/Wildcard-and-Boolean-Search-in-Elasticsearch)通配符和布尔搜索

*如果你觉得这篇文章很有意思，那么你可以探索“* [*【掌握基巴纳 6.0】*](https://www.amazon.com/Mastering-Kibana-6-x-Visualize-histograms/dp/1788831039/ref=olp_product_details?_encoding=UTF8&me=)*”、“* [*基巴纳 7 快速入门指南*](https://www.amazon.com/Kibana-Quick-Start-Guide-Elasticsearch/dp/1789804035) *”、“* [*学习基巴纳 7*](https://www.amazon.com/Learning-Kibana-dashboards-visualization-capabilities-ebook/dp/B07V4SQR6T) *”、“* [*Elasticsearch 7 快速入门指南*](https://www.amazon.com/gp/product/1789803322?pf_rd_p=2d1ab404-3b11-4c97-b3db-48081e145e35) *”等书籍，了解更多*

*原载于*[*https://bqstack.com*](https://bqstack.com/b/detail/84/Basics-of-Data-Search-in-Elasticsearch)*。*