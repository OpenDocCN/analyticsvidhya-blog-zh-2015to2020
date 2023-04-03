# 部署您的第一个 Docker 容器—第 1 部分

> 原文：<https://medium.com/analytics-vidhya/deploy-your-first-docker-container-part-1-db2dfaaa4734?source=collection_archive---------21----------------------->

**步骤 1:运行容器**

第一个任务是识别 Docker 映像的名称，该映像被配置为运行 *Redis* 。使用 Docker，所有容器都是基于 Docker 映像启动的。这些图像包含启动该过程所需的一切；主机不需要任何配置或依赖性。

Jane 可以在 registry.hub.docker.com/的[找到现有的图片，或者使用命令](https://registry.hub.docker.com/)

> docker 搜索<name></name>

例如，要查找 *Redis* 的图像，可以使用 docker search redis。

使用 search 命令，Jane 发现 *Redis* Docker 映像名为 *redis* ，并希望运行*最新的*版本。因为 *Redis* 是一个数据库，Jane 想在继续工作的同时，把它作为后台服务来运行。

要完成这一步，在后台启动一个容器，运行基于官方映像的 Redis 实例。

Docker CLI 有一个名为 *run* 的命令，它将启动一个基于 Docker 映像的容器。其结构是

> *docker 运行<选项>image-name>*

默认情况下，Docker 会在前台运行一个命令。要在后台运行，需要指定选项 *-d* 。

> docker run -d redis

默认情况下，Docker 将运行最新的版本。如果需要特定的版本，可以将其指定为标记，例如，版本 3.2 将是

> *docker run -d redis:3.2*

由于这是 Jane 第一次使用 Redis 映像，它将被下载到 Docker 主机上。

**第二步，我们将继续第二部分**