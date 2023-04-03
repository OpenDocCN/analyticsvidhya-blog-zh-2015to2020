# 使用 Nginx、Gunicorn 和 AWS ECS 设置 Django

> 原文：<https://medium.com/analytics-vidhya/setting-up-django-with-nginx-gunicorn-and-aws-ecs-e1b279c7ae8?source=collection_archive---------0----------------------->

在我之前的博客中，我解释了如何使用 ***Docker*** 在生产环境中运行 Django 应用程序。

在这篇博客中，我将使用 AWS 弹性容器服务(ECS ),这是一个完全托管的服务，用于运行 Docker 容器，我们将部署示例 Django 应用程序。

你可以从我的 GitHub 库下载[源代码](https://github.com/harshvijaythakkar/harsh-django-docker)。

# **什么是** AWS 弹性容器服务(ECS) **？**