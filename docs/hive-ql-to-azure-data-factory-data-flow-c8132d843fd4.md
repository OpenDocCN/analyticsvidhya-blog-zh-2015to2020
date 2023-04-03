# Hive QL 到 Azure 数据工厂数据流

> 原文：<https://medium.com/analytics-vidhya/hive-ql-to-azure-data-factory-data-flow-c8132d843fd4?source=collection_archive---------27----------------------->

这里考虑一个包含 5 个连接的 sql select 查询。我有一个事实和 4 个维度要加入

*   事实人口
*   维度状态
*   维度 Countyname
*   维度竞赛
*   维度性
*   首先创建一个管道
*   创建新的数据流
*   首先连接到事实作为源
*   选择联接
*   将状态配置为另一个源
*   连接联接列
*   对其他尺寸做同样的操作，如下图所示。

![](img/31adc6e4e26da1646454409e67092833.png)

*   创建连接后，保存并发布
*   然后转到管道并触发一次

![](img/5d1ff0f6132c0d38c29b352dd791742c.png)![](img/1c5096dfc02022d9b8281b33d01aa62e.png)![](img/d8f81d07b9570d70882429eb0cf58b41.png)![](img/9035468538567e36a78dfbd20f151a0b.png)![](img/e47c9e952a352589ebed4d76e54a5421.png)![](img/46052b04904f0d54b686f68abceee038.png)![](img/20979d43acd2d961dd3be626972a596e.png)

*最初发表于*[*【https://github.com】*](https://github.com/balakreshnan/Accenture/blob/master/cap/adfmultijoin.md)*。*