# 如何轻松地跨 Docker 容器持久化和共享数据

> 原文：<https://medium.com/analytics-vidhya/how-to-persist-and-share-data-across-docker-containers-easily-5bf681e61956?source=collection_archive---------23----------------------->

本文讨论了在 docker 中跨不同容器/映像共享数据的简单技术。

在学习如何分享之前，让我们看看 Docker Volume 是什么意思..

卷是跨 docker 容器持久化和共享数据的首选机制。

简单来说什么是卷？

> 指向主机中存储容器数据的物理位置的指针。