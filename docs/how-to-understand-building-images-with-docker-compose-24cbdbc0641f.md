# 如何理解用 docker-compose 构建图像

> 原文：<https://medium.com/analytics-vidhya/how-to-understand-building-images-with-docker-compose-24cbdbc0641f?source=collection_archive---------0----------------------->

# 问题

当我们[下载图片](/@zhao.li/how-to-understand-downloading-images-with-docker-compose-236e323e541)然后[将图片作为容器](/@zhao.li/how-to-understand-running-containers-with-docker-compose-a8160b487a98)运行时，我们使用的是其他人通过 Docker Hub 提供的图片。我们如何创建一个为我们自己定制的 Docker 映像？

# 解决办法

为了创建满足我们自己需求的定制图像，我们需要构建一个包含我们所需要的内容的图像。

本教程将通过建立一个 Docker 图像，然后解释发生了什么。