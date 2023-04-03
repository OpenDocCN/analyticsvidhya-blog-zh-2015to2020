# 异步 Python:很容易做错

> 原文：<https://medium.com/analytics-vidhya/async-python-its-easy-to-do-it-wrong-9cfacb9a946e?source=collection_archive---------18----------------------->

![](img/0b9df11596c4f4c04d64e65f44766d80.png)

艾米丽·莫特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

编写不像你想象的那样运行的异步 Python 代码**很容易**。我最近遇到了一个情况，一个 Sanic web 应用程序经常在 Kubernetes 中崩溃/重启，但是没有任何痕迹。

为什么它在没有任何日志的情况下崩溃/重启？我一无所知。我想到的唯一可能的原因是我在其上配置的就绪性/活性探测器…