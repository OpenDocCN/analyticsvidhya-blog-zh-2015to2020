# 创建 Python API 包装器(Ally Invest API)

> 原文：<https://medium.com/analytics-vidhya/creating-a-python-api-wrapper-ally-invest-api-568934a1411c?source=collection_archive---------1----------------------->

对于我的一些投资账户，我使用的是 Ally Invest，它有一个很好的 API，可以让你查询股票和账户数据，创建订单，管理观察列表和数据流。API 文档中有一些用 Java、Node 编写的示例程序。JS、PHP、R 和 Ruby。然而，我有几个想用 Python 实现的应用程序的想法。

GitHub 上已经有几个开源项目实现了 Ally Invest API 的包装器，但是没有一个实现了我想要做的一切。灵感来自 [Discord.py](https://github.com/Rapptz/discord.py) 和…