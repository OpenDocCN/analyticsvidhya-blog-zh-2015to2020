# ElasticBERT:使用 BERT 和 ElasticSearch 的信息检索

> 原文：<https://medium.com/analytics-vidhya/elasticbert-information-retrieval-using-bert-and-elasticsearch-51fef465b9ae?source=collection_archive---------2----------------------->

![](img/bf74408483046da02e5fb6ab5aa29bdb.png)

我建立的 hat 是一个简单的信息检索系统，使用预先训练的 BERT 模型和弹性搜索。最近，elasticsearch 宣布在这个[帖子](https://www.elastic.co/jp/blog/text-similarity-search-with-vectors-in-elasticsearch)中使用向量进行文本相似性搜索。我们将文本转换成固定长度的向量，该向量将被保存到弹性搜索索引中。然后我们使用余弦相似性度量来计算出索引中最相似的内容。这是…