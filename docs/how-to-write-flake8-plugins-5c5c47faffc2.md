# 如何编写 Flake8 插件😍

> 原文：<https://medium.com/analytics-vidhya/how-to-write-flake8-plugins-5c5c47faffc2?source=collection_archive---------4----------------------->

## 自动化您的代码评审

![](img/94405d47e7a0671048478c2ee7bdbdd9.png)

摄影爱好在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上

代码评审中真正重要的部分几乎不可能自动化:架构决策和逻辑错误。它们对你的代码库来说太定制化了；对于拉取请求太具体。

但是，代码评审中的很多注释并不是这样的。它们是关于简单的风格决定、常见的小错误和误解。他们是…