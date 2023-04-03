# 如何用 Jaeger 可视化 Sanic Python 微服务

> 原文：<https://medium.com/analytics-vidhya/how-to-visualize-sanic-python-micro-services-with-jaeger-b7943938dbcb?source=collection_archive---------17----------------------->

![](img/0de25f31dd9a48387fa1b421db0cc462.png)

特雷文·鲁迪在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

本文将描述使用 Sanic 和多个 workers 创建两个简单的微服务，以及有助于可视化流程的 Jaeger 配置。

来自 Sanic 官方文件:

> “Sanic 是一个 Python 3.6+ web 服务器和 web 框架，旨在快速运行。它允许使用 Python 中添加的 async/await 语法…