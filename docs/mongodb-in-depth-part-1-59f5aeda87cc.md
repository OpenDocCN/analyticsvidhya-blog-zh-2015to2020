# MongoDB 深度—第 1 部分

> 原文：<https://medium.com/analytics-vidhya/mongodb-in-depth-part-1-59f5aeda87cc?source=collection_archive---------13----------------------->

![](img/f1a5e0a152d2700b0fc0ac255a11b128.png)

由[米卡·鲍梅斯特](https://unsplash.com/@mbaumi?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

我们将介绍一些与 NoSQL 数据库 MongoDB 相关的高级概念，主要集中在运行大规模应用程序上。我们将主要讨论 MongoDB 的内部机制，当你需要使用 MongoDB 来支持一个大规模的应用时，这些机制将会对你有所帮助。

在继续之前，让我们重温一下 NoSQL 数据库，尤其是 MongoDB 的概念。本文(第 1 部分)将只涉及基础知识。我们将从[第 2 部分](/@rohan36/mongodb-in-depth-part-2-5e0d37c03853)开始学习高级概念。

NoSQL 数据库本质上是一个没有模式的数据库。我们将插入本质上与关系无关的数据。NoSQL 数据库旨在跨许多服务器水平扩展，这使得它们对于超过单个服务器容量的大数据量或应用程序负载很有吸引力。

有四种类型的 NoSQL 数据库:键值存储、基于文档、基于列和基于图形。MongoDB 是一个基于文档的 NoSQL 数据库。数据以文档的形式存储在 MongoDB 中。MongoDB 文档类似于 JSON 对象。字段的值可以包括其他文档、数组和文档数组。

词典:

1.  集合—存储逻辑上相似的文档。例如，用户集合
2.  数据库—存储来自特定域的集合。目录数据库
3.  mongod——MongoDB 系统的主要守护进程。`[mongod](https://docs.mongodb.com/manual/reference/program/mongod/#bin.mongod)`进程将 MongoDB 服务器作为守护进程启动。MongoDB 服务器管理数据请求和格式化，并管理后台操作。
4.  节点—物理机器。多个 mongod 守护进程可以在一台机器上运行。一个 mongod 守护进程就是一个 MongoDB 服务器。所以一个节点可以运行多个 MongoDB 服务器。如果这很难理解，那就把它想象成在一部智能手机上运行的两个 Whatsapp 信使。

在生产中运行 MongoDB 有三种不同的方式:

1.  单独的
2.  副本集
3.  串

简单说一下吧。
独立模式只是运行一个 MongoDB 服务器。对于生产环境，这是绝对不推荐的。如果这台服务器崩溃，所有数据都将丢失。这在本地或测试环境中探索 MongoDB 时非常有用。

顾名思义，副本集是一组包含相同数据的 MongoDB 服务器。该集有一个主服务器，其余都是辅助服务器。辅助服务器通常与主服务器具有相同的数据。所以我们有复制数据的服务器。当主服务器崩溃时，这很有用。辅助服务器不应与主服务器位于同一节点上。如果节点崩溃，我们将失去所有服务器和所有数据，我们将无法实现容错。

集群模式运行 MongoDB，其中数据被划分为多个逻辑分区，称为碎片，单个 MongoDB 服务器并不包含所有数据。当数据太多而无法存储在一台服务器上时，这很有用。想想数十亿字节的数据。碎片被部署为副本集。所以一个集群有许多副本集。

阅读本系列的第 2 部分 ，我们将讨论在最后两种模式下运行 MongoDB 的技术细节。独立非常简单。我们将从 [**第二部分**](/@rohan36/mongodb-in-depth-part-2-5e0d37c03853) 开始深入研究副本集和分片。