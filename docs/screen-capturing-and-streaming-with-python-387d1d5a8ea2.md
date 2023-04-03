# 使用 Python 进行屏幕捕捉和流式传输

> 原文：<https://medium.com/analytics-vidhya/screen-capturing-and-streaming-with-python-387d1d5a8ea2?source=collection_archive---------3----------------------->

最近我一直在考虑开发人工智能来玩我的 steam 库中的一些游戏。大多数游戏，不像[星际争霸 2](https://github.com/deepmind/pysc2) ，缺乏一个 API 让开发者挂钩游戏观察环境并做出决定。因此，这两个操作必须使用更原始的方法。在这篇文章中，我讨论了前者，即从应用程序中收集数据，就我而言，可以用来训练机器学习模型，但我相信这种功能还有许多其他用途。

# 属国