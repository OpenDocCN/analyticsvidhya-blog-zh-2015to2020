# 为 Elasticsearch 编写应用程序，这里有一些你应该注意的事情。

> 原文：<https://medium.com/analytics-vidhya/writing-apps-for-elasticsearch-here-are-some-things-you-should-take-care-of-828135c2995b?source=collection_archive---------19----------------------->

![](img/fe23c7b06fa8e4f976c753bfdcc715ae.png)

图片来自[percona.com](http://www.percona.com)

我最近一直在用`Python`编写一个应用程序，它与一个 Elasticsearch 集群通信，从/向它读取/写入数据(`document`)。让我们以下面的代码为例，它与我们的 Elasticsearch 集群进行通信。

```
from elasticsearch import Elasticsearch def add_document(i_index, i_doc_type, i_id, i_body):
    es = es_connect()
    try:
        res=es.index(index=i_index, doc_type=i_doc_type, id=i_id, body=i_body)
        count_success=res.get("_shards",{}).get("successful")
        return count_success
    except Exception as ex:
        raise Exception(ex)def es_connect():
    try:
        es = Elasticsearch(hosts=[{"host": es_host, "port": es_port}])
        return es
    except Exception as ex: 
        raise Exception("Failed connecting to Elasticsearch")
```

您可能已经发现，可以调用`add_document`来向指定的 Elasticsearch `index`和`doctype`添加一个`document`，最终调用`es_connect`来使用主机:`es_host`和端口:`es_port`连接到 Elasticsearch 集群。

```
def get_document(i_index, i_doc_type, i_id):
    es = es_connect()
    try:
        res=es.get(index=i_index, doc_type=i_doc_type, id=i_id)
        return res 
    except Exception as ex: 
        raise Exception("Document ID not found")will be setup Thu
```

现在假设您的应用程序有两个组件/服务`componentA`和`componentB.` `componentA`向 Elasticsearch 集群添加一个文档，而`componentB`尝试获取相同的文档，更新它，然后将其添加回 Elasticsearch 集群。

我在这个场景中遇到的问题是，有时当`componentB`试图获取 es 文档时，操作会因`Document not found`而失败，即使`componentA`写了文档，而我们试图获取具有相同文档 id 的文档。现在，如果我们看到官方的弹性搜索文档

> 默认情况下，Elasticsearch 每秒钟都会定期刷新索引，但仅限于在过去 30 秒内收到一个或更多搜索请求的索引。

因此，如果执行`componentA`和`componentB`之间的时间差小于`1 second`，则由`componentA`编写的文档可能无法在所有节点上进行搜索。换句话说，我们可以说,`componentA`编写的文档在编写后不能立即用于搜索。为了解决这个问题，我们将研究一个名为`refresh`的参数。

# 恢复精神

我们可以在索引级别设置刷新值，更改索引的设置，或者您可以在使用`.index`函数/API 存储文档时提供刷新设置。您可以查看此[文档](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-refresh.html)了解如何设置步进设置。

对于下面的另一个选项，您可以更改调用`.index`函数的代码

```
def add_document(i_index, i_doc_type, i_id, i_body):
    es = es_connect()
    try:
        res=es.index(index=i_index, doc_type=i_doc_type, id=i_id, body=i_body, refresh='true')
        count_success=res.get("_shards",{}).get("successful")
        return count_success
    except Exception as ex: 
        raise Exception(ex)
```

如果你注意到我们已经引入了`refresh=’true’`，现在将使用`add_document`功能添加到 Elasticsearch 集群的文档，将在添加到集群后立即可供搜索。

T 以下是在性能部分使用`refresh=’true’`的一些后果，因此请确保您仅在应用程序因添加文档后立即不可用而出现功能问题时才使用它。关于这一点的更多信息可以在[这个链接](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-refresh.html)中阅读。

# VersionConflictEngineException

`VersionConflictEngineException`通常在我们试图更新 Elasticsearch index 的文档时会遇到，顾名思义，版本存在一些问题。

对于每个 Elasticsearch 文档，它都维护该文档的一个版本。例如，如果我们只是创建一个索引，并尝试将一个文档放入该索引中

```
POST /index_name/_doc
{
    "user" : "Vivek",
    "post_date" : "2009-11-15T14:12:12",
    "message" : "trying out Elasticsearch"
}#returns
{
    "_index": "index_name",
    "_type": "_doc",
    "_id": "7kRpUW4BJVpxMvy30qnq",
    "_version": 1,
    "result": "created",
    "_shards": {
        "total": 2,
        "successful": 1,
        "failed": 0
    },
    "_seq_no": 0,
    "_primary_term": 1
}
```

可以清楚的看到文档是被创建的，创建文档的版本是`“_version”:1`。现在让我们假设我们正试图从我们的 python 代码中更新这个文档，并且`update_document`的定义如下所示

```
def update_document(i_index, i_doc_type, i_id, i_body):
        es= es_connect()
        try:
                result =  es.update(index=i_index, doc_type=i_doc_type, id=i_id, body=i_body)
                return result
        except Exception as e:
                print("There was an exception updating the document", e)
```

现在考虑这个场景，在更新文档之前，我们得到 Elasticsearch 文档(用`version 1`)，然后尝试用一些修改过的数据更新它。但是在我们从 Elasticsearch 获得数据并更新它的过程中，一些其他程序或过程改变了同一个文档，最终改变了那个文档的版本(making is `version 2`)。现在，如果我们的程序试图更新同一个文档，它将看到当它获得该文档时，它有`version 1`，但现在它已被更改为 2，更新该文档可能会覆盖其他人对该文档所做的更改，这就是为什么我们得到`VersionConflictEngineException`。

换句话说，我们可以说，如果我们从 Elasticsearch 集群获得文档的`version n`，我们将尝试更新相同的版本，如果具有相同 ID 的文档现在具有`version n+x`(当我们尝试更新它时)，这仅仅意味着同时其他人对该文档进行了一些更改，这将导致`VersionConflictEngineException`。

我们通常通过重试更新相同的文档，但使用更新的版本号来处理这个问题。在得到这个错误后，我们可以做的是发出另一个 get 请求，获取文档的更新文档(最新版本),然后用我们的更改更新该文档。

现在出现的重要问题是，我们应该重试多少次来更新文档？根据我的个人经验，我认为如果`n`进程试图同时更新你的文档，最好重试`n-1`次。

作为手动重试更新文档的一部分，我们可以在调用`update` API 时传递一个参数(`retry_on_conflict`，如果`VersionConflictEngineException` 发生，API 将负责重试更新。以下是`elasticsearch`包中该参数的描述。

```
:arg retry_on_conflict: Specify how many times should the operation be
            retried when a conflict occurs (default: 0)
```

我很乐意听到你对此的反馈，如果你对这篇文章有任何问题/建议，你可以在推特上联系我。