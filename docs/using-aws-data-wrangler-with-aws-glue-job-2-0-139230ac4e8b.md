# 将 AWS Data Wrangler 与 AWS Glue Job 2.0 和 Amazon Redshift 连接一起使用

> 原文：<https://medium.com/analytics-vidhya/using-aws-data-wrangler-with-aws-glue-job-2-0-139230ac4e8b?source=collection_archive---------7----------------------->

我承认，AWS Data Wrangler 已经成为我开发提取、转换和加载(ETL)数据管道和其他日常脚本的首选包。AWS Data Wrangler 与 S3、Glue Catalog、Athena、数据库、EMR 等多种大数据 AWS 服务的集成让工程师的生活变得简单。它还提供了导入像 Pandas 和 PyArrow 这样的包来帮助编写转换的能力。

在这篇博文中，我将带你通过一个假想的用例来读取 glue 中的数据…