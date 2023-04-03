# 使用 AWS Lambda 读取拼花文件

> 原文：<https://medium.com/analytics-vidhya/reading-parquet-files-with-aws-lambda-487a4eff20da?source=collection_archive---------0----------------------->

我有一个用例，每次上传文件时，从存储在 S3 的 parquet 文件中读取数据(几列),并写入 DynamoDB 表。在考虑使用 AWS Lambda 时，我一直在寻找如何在 Lambda 中读取 parquet 文件的选项，直到我偶然发现了 [**AWS Data Wrangler**](https://github.com/awslabs/aws-data-wrangler) 。

从文件上看-

> ***什么是 AWS 数据牧马人？***
> 
> *一个开源的 Python 包，将 Pandas 库的功能扩展到 AWS，连接 DataFrames 和 AWS 数据相关服务(Amazon Redshift、AWS Glue、Amazon Athena、Amazon EMR 等)。*