# 如何理解用 docker-compose 运行容器

> 原文：<https://medium.com/analytics-vidhya/how-to-understand-running-containers-with-docker-compose-a8160b487a98?source=collection_archive---------3----------------------->

# 问题

现在[我们已经下载了一个图像](/@zhao.li/how-to-understand-downloading-images-with-docker-compose-236e323e541)到我们的本地开发机器(又名主机)上。我们可以用这个图像做什么？

# 解决办法

我们有一个可能的选择，就是基于这个图像运行一个容器。

本教程将从 [Docker Hub](https://hub.docker.com/) 运行 Docker 镜像，然后理解发生了什么。

以下是我们将采取的步骤:

1.  结合使用我们现有的`docker-compose.yml`文件…