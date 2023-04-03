# 芹菜节流—为队列设置速率限制

> 原文：<https://medium.com/analytics-vidhya/celery-throttling-setting-rate-limit-for-queues-5b5bf16c73ce?source=collection_archive---------4----------------------->

## 在本文中，我将展示如何在基于分布式任务队列的系统中控制队列的吞吐量，或者用更简单的语言来说，如何设置其速率限制。作为一个例子，我将使用 python 和我最喜欢的 **Celery + RabbitMQ** 工具包，尽管我使用的算法不依赖于这些工具，并且可以在任何其他栈上实现。