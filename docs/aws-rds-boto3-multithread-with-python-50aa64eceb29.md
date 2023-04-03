# AWS —使用 Python 节省您的 RDS 资金

> 原文：<https://medium.com/analytics-vidhya/aws-rds-boto3-multithread-with-python-50aa64eceb29?source=collection_archive---------9----------------------->

了解如何通过轻松管理 RDS 数据库来节省资金

![](img/38b20763e0956086b6a3d10ba11745d5.png)

图片来自[推特:](https://unsplash.com/@jankolar?utm_source=medium&utm_medium=referral) [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的@jankolario

AWS 的一个大问题是如何管理您的基础设施，以始终支付更少的费用。在本文中，我们将讨论如何在 RDS 数据库方面节省一些成本，因为 RDS 数据库很快就会变得非常庞大。

> 一个 Postgres on demand DB，m5.large，有 100GB 的存储空间，将花费你…