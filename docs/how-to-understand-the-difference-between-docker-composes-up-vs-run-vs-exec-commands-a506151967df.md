# 如何理解 docker-compose 的 up vs run vs exec 命令之间的区别

> 原文：<https://medium.com/analytics-vidhya/how-to-understand-the-difference-between-docker-composes-up-vs-run-vs-exec-commands-a506151967df?source=collection_archive---------1----------------------->

# 问题

您开始使用容器，并注意到 Docker Compose 提供了多种运行容器的方式，即`[up](https://docs.docker.com/compose/reference/up/)`、`[run](https://docs.docker.com/compose/reference/run/)`和`[exec](https://docs.docker.com/compose/reference/exec/)`。两者的区别是什么？你什么时候想用其中一个而不是另一个？

# 解决办法

大多数情况下，您很可能希望启动您的`docker-compose.yml`中列出的所有服务，并让容器运行它们的默认命令，因此您可能希望使用`[up](https://docs.docker.com/compose/reference/up/)`。