# 部署您的第一个 Docker 容器—第 3 部分

> 原文：<https://medium.com/analytics-vidhya/deploy-your-first-docker-container-part-3-bd54e0bc62b?source=collection_archive---------18----------------------->

在上一篇文章中，我们学习了如何运行 Redis 的特定实例。现在，我们将进一步了解部署您第一个 docker 容器的步骤:

## 步骤 5:保存数据

在使用容器几天后，Jane 意识到当她删除和重新创建容器时，存储的数据会不断被删除。Jane 需要在重新创建容器时持久化和重用数据。

容器被设计成无状态的。使用选项*-v<host-dir>:<container-dir>*来绑定目录(也称为卷)。装载目录时，容器可以访问主机上该目录中存在的文件，并且容器内目录中更改/写入的任何数据都将存储在主机上。这允许您在不丢失数据的情况下升级或更改容器。

通过使用 Redis 的 Docker Hub 文档，Jane 发现正式的 Redis 映像将日志和数据存储在 a /data 目录中。

任何需要保存在 Docker 主机上而不是容器中的数据都应该存储在 */opt/docker/data/redis* 中。

解决任务的完整命令是`docker run -d --name redisMapped -v /opt/docker/data/redis:/data redis`

## 步骤 6:在前台运行容器

Jane 一直在 Redis 做后台处理。Jane 想知道容器如何与前台进程一起工作，比如 *ps* 或 *bash* 。

之前，Jane 使用 *-d* 在分离的后台状态下执行容器。如果不指定这一点，容器将在前台运行。如果 Jane 想要与容器交互(例如，访问 bash shell)，她可以包含选项 *-it* 。

除了定义容器是在后台还是前台运行，某些图像还允许您覆盖用于启动图像的命令。能够替换默认命令使得拥有一个可以以多种方式重新利用的单一图像成为可能。例如，Ubuntu 映像既可以运行 OS 命令，也可以使用 */bin/bash* 运行交互式 bash 提示符

## 例子

命令`docker run ubuntu ps`启动一个 Ubuntu 容器，执行命令 *ps* 查看容器中运行的所有进程。

使用`docker run -it ubuntu bash`允许 Jane 访问容器内部的 bash shell。