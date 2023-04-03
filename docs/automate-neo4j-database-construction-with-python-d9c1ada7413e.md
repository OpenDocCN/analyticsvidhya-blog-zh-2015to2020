# 使用 Python 自动构建 Neo4j 数据库

> 原文：<https://medium.com/analytics-vidhya/automate-neo4j-database-construction-with-python-d9c1ada7413e?source=collection_archive---------4----------------------->

## 详细介绍如何通过 Python 脚本导入数据和创建图形数据库的教程。

![](img/96b53937c9221714363d58b7ca9dbf20.png)

由[克里斯托弗·高尔](https://unsplash.com/@cgower?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

我最近的一个项目涉及在 Neo4j 中创建一个**大规模**(超过一百万个节点和关系)图形数据库。当有那么多要处理的时候，拥有一个自动化的脚本就成了关键。这篇文章概述了我给处于类似情况的人的一些建议。

代码链接贯穿始终，请评论任何额外的提示！

# 先决条件

1.  需要安装 Python(最好是 2.7 或更高版本)
2.  应该创建 Neo4j 数据库，并设置所需的任何插件或设置
    (我使用 APOC 和 GDSL 插件，并将最大堆大小增加到 3GB)
3.  Cypher 查询将按照您的意愿构建数据库

# 设置:Python 目录和虚拟环境

1.  导航到图表的目录。我 Mac 上的目录看起来像`/Library/Application Support/Neo4j Desktop/Application/neo4jDatabases/database-asdfjkl-1234-etc/installation-4.1.1/`。我把它钉在我的边栏上以便于访问。
2.  在`installation`目录中创建一个`python`目录。
3.  可选:在`python`目录中，创建一个虚拟环境。我真的很喜欢使用 venv，因为我可以很容易地与队友或其他项目分别共享或转移所有必需的包的`requirements.txt`。[该资源](https://docs.python-guide.org/dev/virtualenvs/)帮助我在 Mac 和 PC 上成功设置了虚拟环境。
4.  一旦创建了虚拟环境，就可以通过`pip`安装`py2neo`包。
5.  如果您的 Cypher 查询从任何文件导入数据，请将这些文件放在`import`目录中(应该在同一个`installation`目录中)。

# 编码时间

*注意:这个项目的完整代码在下面的存储库中，所以你可以参考下面的 gists 的上下文或者检查我正在导入的数据。*

[](https://github.com/mgipson/Text-Similarity) [## mgi pson/文本相似性

### GitHub 是超过 5000 万开发人员的家园，他们一起工作来托管和审查代码、管理项目和构建…

github.com](https://github.com/mgipson/Text-Similarity) 

## 脚本框架

我们导入数据或构建节点和关系的主要 Python 脚本将存在于这个方便的 Python 文件夹和虚拟环境中。

这个代码块做了最少的工作:
1。导入`py2neo`
2。建立数据库连接
3。运行单个查询

使用 [Py2Neo](https://py2neo.org/2020.1/) ，您可以按原样运行 Cypher 查询，或者使用 Py2Neo 函数来执行类似的操作。我的个人偏好是按原样运行 Cypher 查询，因为我可以简单地在 Neo4j 浏览器中复制和粘贴代码。

如果您想要一些超级简单的东西，您可以用您不同的查询重复`graph.run()`调用，它将完成工作。然而，我在开发自己的脚本时发现了一些非常有用的东西；如果你想知道这些对你是否有用，请继续阅读。

## 批处理和自动提交

我利用了[定期提交(也称为批处理](https://neo4j.com/labs/apoc/4.1/graph-updates/periodic-execution/))，因为我的大型查询经常遇到内存问题。

如果您确实在批处理中运行，您将希望使用一个开放的事务(通过下面代码中的`begin(), run(), commit()`序列完成)而不是一个关闭的事务(独立的`run()`调用)。
这是因为关闭的事务会自动提交，这意味着如果您使用批处理和关闭的事务，它将执行并提交您的第一个批处理，但不会继续执行其他批处理。肯定不是我们想要的。

```
#Without Periodic Commitsgraph.run("""LOAD CSV WITH HEADERS
FROM 'file:///womeninstem.csv' AS line
MERGE (person:Person { name: line.personLabel })
ON CREATE SET person.birth = line.birthdate, person.death = line.deathdate""")#With Periodic Commitsquery_list = []query_list.append("""CALL apoc.periodic.iterate("
LOAD CSV WITH HEADERS
FROM 'file:///womeninstem.csv' AS line
RETURN line
","
MERGE (person:Person { name: line.personLabel })
ON CREATE SET person.birth = line.birthdate, person.death = line.deathdate", {batchSize:1000, parallel:false, retries: 10}) YIELD operations""")#would have more queries in query_listfor query in query_list:        
   g = graph.begin() # open transaction        
   result = g.run(query).to_data_frame() # execute query
   g.commit() # close transaction
```

## 查询失败警报

通过 Python 代码而不是在浏览器中运行 Cypher 查询的问题是，它并不总是告诉您查询是否失败——它会提醒您语法错误，但仅此而已。

因此，如果一个查询没有完全完成，实现一些提示是很重要的。

```
query_list = []query_list.append("""CALL apoc.periodic.iterate("
LOAD CSV WITH HEADERS
FROM 'file:///womeninstem.csv' AS line
RETURN line
","
MERGE (person:Person { name: line.personLabel })
ON CREATE SET person.birth = line.birthdate, person.death = line.deathdate", {batchSize:1000, parallel:false, retries: 10}) YIELD operations""")#would have more queries in query_listfor query in query_list:        
   g = graph.begin() # open transaction        
   result = g.run(query).to_data_frame() # execute query
   try:            
      if result['operations'][0]['failed'] > 0: #means batch failed
         print('Could not finish query')
      else: #means batch succeeded                
         g.commit() # close transaction
         print('Completed query ')
   except Exception as e: #means there's a syntax or other error         
      g.commit() # close transaction  
      print('Completed query')
```

## 参数化查询

我就此写了一篇单独的短文——长话短说，您可以在 Cypher 查询的末尾添加简单的`.format()`替换。

[](https://madisongipson.medium.com/running-a-cypher-query-with-parameters-through-a-python-script-f0089e245b4d) [## 通过 Python 脚本运行带参数的密码查询

### 教程详细介绍了如何使用 Python 库“Py2Neo”来动态参数化和执行 Cypher 查询。

madisongipson.medium.com](https://madisongipson.medium.com/running-a-cypher-query-with-parameters-through-a-python-script-f0089e245b4d) 

我还没有写这些，但是我有一些构建 Neo4j 数据库的技巧(并行查询、显示脚本进度的简单 GUI 等)。);然而，他们应该有自己的文章，我还没有写那些。我计划一旦我做了，就把它们链接到这里！

如果您想查看从中提取的更大的项目，请查看这个资源库。

[](https://github.com/mgipson/Text-Similarity) [## mgi pson/文本相似性

### GitHub 是超过 5000 万开发人员的家园，他们一起工作来托管和审查代码、管理项目和构建…

github.com](https://github.com/mgipson/Text-Similarity)