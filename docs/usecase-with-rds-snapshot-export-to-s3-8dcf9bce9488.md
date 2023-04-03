# 将 RDS 快照导出到 S3 的用例

> 原文：<https://medium.com/analytics-vidhya/usecase-with-rds-snapshot-export-to-s3-8dcf9bce9488?source=collection_archive---------1----------------------->

AWS 最近宣布了“[亚马逊 RDS 快照导出到 S3](https://aws.amazon.com/about-aws/whats-new/2020/01/announcing-amazon-relational-database-service-snapshot-export-to-s3/) ”功能，现在您可以将亚马逊关系数据库服务(亚马逊 RDS)或亚马逊 Aurora 快照导出到亚马逊 S3 作为 Apache Parquet，这是一种用于分析的高效开放式列存储格式。

我有一个用例，每天用帐户 B(us-east-1)中的全部数据集刷新 Athena 表，这些数据集来自帐户 A (us-west-2)的私有子网下运行的 Aurora MySQL 数据库。我能想到的两个解决方案是-

1.  让 EC2 实例在公共子网中运行，作为通向 Aurora 的桥梁…