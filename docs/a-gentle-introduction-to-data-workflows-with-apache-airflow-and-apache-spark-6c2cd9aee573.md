# 关于 Apache Airflow 和 Apache Spark 的数据工作流的简明介绍

> 原文：<https://medium.com/analytics-vidhya/a-gentle-introduction-to-data-workflows-with-apache-airflow-and-apache-spark-6c2cd9aee573?source=collection_archive---------1----------------------->

假设您在一个本地 Spark 中开发了一个转换流程，并希望对其进行调度，这样一个简单的 Cron 作业就足够了。现在想想，在这个过程之后，您需要启动许多其他的过程，如 python 转换或 HTTP 请求，这也是您的生产环境，因此您需要监控每个步骤
这听起来很困难吗？只有火花和克朗工作，是的，但感谢我们有[阿帕奇气流](https://airflow.apache.org/)。

> Airflow 是一个平台，可编程地创作、安排…