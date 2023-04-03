# 了解 OpenTSDB——一个分布式和可伸缩的时序数据库

> 原文：<https://medium.com/analytics-vidhya/understanding-opentsdb-a-distributed-and-scalable-time-series-database-e4efc7a3dbb7?source=collection_archive---------6----------------------->

学习 OpenTSDB 的基础知识，了解它的架构，并学习如何使用它。

OpenTSDB 代表开放时间序列数据库。OpenTSDB，顾名思义，是建立在 HBase 之上的时序数据库。它具有出色的读写性能，可以很好地扩展，并且是分布式的。在进入 OpenTSDB 的核心之前，我们需要理解几个概念和术语，在进入它的体系结构之前，我们将浏览它们。最后，我们将看到如何…