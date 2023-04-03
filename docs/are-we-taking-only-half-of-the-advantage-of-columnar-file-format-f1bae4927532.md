# 我们是否只利用了分栏文件格式的一半优势？

> 原文：<https://medium.com/analytics-vidhya/are-we-taking-only-half-of-the-advantage-of-columnar-file-format-f1bae4927532?source=collection_archive---------6----------------------->

( **原发于*[*LinkedIn*](https://www.linkedin.com/pulse/we-taking-only-half-advantage-columnar-file-format-eric-sun/)*2018 年*)

列文件格式已经成为大数据系统的主要存储选择，但当我本周末搜索相关主题时，我发现大多数文章都在谈论简单的查询基准和特定列格式与行格式之间的存储足迹比较。排序也是分栏格式的一个关键特性，但是它的好处和有效实践到目前为止还没有被强调或详细解释。IMHO，使用分栏格式…