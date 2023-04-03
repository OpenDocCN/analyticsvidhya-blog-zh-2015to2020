# RESTful API，如何|第 4 部分—部署

> 原文：<https://medium.com/analytics-vidhya/restful-api-how-to-part-4-deployment-2347d44e4aad?source=collection_archive---------24----------------------->

![](img/45375d3f7f32820ae279c163f95b96da.png)

设计和实施服务是我日常工作的一部分，我想分享一些最佳实践和技巧，可以帮助你的工作。

在这个关于 RESTful API 的系列文章中，我将讨论几个主题:

*   设计
*   履行
*   测试
*   **部署**

# 一些信息

> *我们将使用*[***swagger***](https://editor.swagger.io/)*来设计我们的 API，*[***python***](https://www.python.org/)*语言来创建微服务，最后*[***Docker***](https://www.docker.com/)*来交付最终的解决方案。所有的代码都在这个* [***回购***](https://github.com/dandpz/restfulapi-howto) *中。*

我们现在到了本系列的最后一篇文章。这里我们将讨论应用程序的部署。对于这个范围我们将使用 Docker 技术，如果你不知道它，去[官方网站](https://www.docker.com/)，在那里你会找到很多关于它的信息，并且[文档](https://docs.docker.com/)是一个宝贵的资源。

我将不涉及 docker 的安装，它取决于您的平台，所以我将直接跳到应用程序部署。

# Dockerfile 文件

首先我们必须填充 Dockerfile，在生成的代码中 Dockerfile 已经存在，但是我们应该稍微改变一下内容。在文件的新版本下面。

```
FROM pypy:3-7.3.0-slim

# add a not privileged user
RUN useradd base_user

# create the application folder and set it as the working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# upgrade pip and install requirements
RUN pip install --upgrade pip
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

# copy the application to the wd
COPY . /usr/src/app

# switch to not privileged user
USER base_user

EXPOSE 8080

ENTRYPOINT ["gunicorn"]

CMD ["wsgi:app"]
```

正如您所看到的，文件已经发生了很大的变化，我想重点介绍文件的不同部分:

## 安全性

在网上你会发现大量的 Dockerfile 例子，但是它们经常会带来安全漏洞。让我们来谈谈 docker 用户拥有的用户特权:

基本映像几乎总是以 *root* 权限运行，这是因为他们需要安装包来构建基本映像，但对于最终应用程序来说，通常不再需要 root 用户，相反它可能会使您的系统暴露在危险的攻击之下。

为了降低这种风险，我们创建一个没有 root 权限的用户，然后在配置完映像后，我们切换到这个**用户**。这是运行我们的容器的更安全的方式。

## 隐藏物

在构建阶段，缓存层非常有用，不会浪费太多时间等待，当我们编写 Dockerfile 时，明智地编写每个指令非常重要，例如，我们首先复制需求文件并安装它们，然后才复制应用程序文件夹。这样，如果我们更改代码中的任何内容，我们不会使需求无效，映像将使用缓存层并跳过需求的安装。

## 图像尺寸

谈到图层，我们还应该考虑图像的大小，包括基本图像和我们用 Dockerfile 添加的图层。

始终选择符合您的**要求**的基础映像！显然，从两个方向来看，浪费几个小时的工作来从一个 *scratch* 映像构建您的映像，以使最终的映像大小减少 4 MB 是没有意义的。

还要考虑编辑 Dockerfile 文件时添加的所有层。

最后但同样重要的是，给你的项目添加一个**是非常重要的。dockerignore** 文件，其工作方式与已知的*完全相同。git 忽略*文件，避免将*无用的*文件和目录复制到您的映像中。

## 建设

为了构建映像，我们运行以下命令:

```
docker build -t test_app:1.0 .
```

使用这个命令，我们用标签 *test_app:1.0 来标记我们的图像。*

> 这里，我们在本地环境中构建应用程序的 docker 映像，在真实的生产环境中，强烈建议将映像推送到 docker 注册表中。
> 
> 要了解更多信息，请参见[文档](https://docs.docker.com/registry/)。

# 生产喜欢

因此，我们现在已经构建了 docker 映像，为了模拟类似生产的环境，我们将使用 [docker compose](https://docs.docker.com/compose/) 。请参阅文档，以便在您的系统上安装它。

在 docker-compose 文件下面。

```
version: '3'
services:
  api:
    image: test_app:1.0
    ports:
    - 80:8080
    env_file:
      - .env
    restart: always
```

在 docker-compose 文件中，我们描述了应用程序部署的配置，我们在*端口* 80 上公开应用程序，沿着 docker-compose 文件我们将提供一个**。env** 文件包含所有对配置应用程序有用的环境变量。*重启*指令指示发动机始终重启**容器**已经停止。

让我们运行它:

```
docker-compose up --scale api=3
```

使用 *scale* 标志，我们告诉 docker-compose 创建 3 个服务实例，这样我们就可以处理大量的流量。

在这篇文章中，我们看到 hot to **配置**一个 docker 文件并**运行**带有 docker compose 的容器。我们没有讨论像 Docker Swarm 或 Kubernetes 这样的*编排*工具，因为它们很复杂，不在本文的讨论范围之内。

关于 Docker，我想列出一些有用的链接:

*   [https://docs . docker . com/develop/develop-images/docker file _ best-practices/](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
*   【https://docs.docker.com/engine/security/security/ 
*   [https://www.docker.com/play-with-docker](https://www.docker.com/play-with-docker)

**就这些！**我们终于到了这个系列的最后一篇文章，感谢来到这里的每一个人，**感谢你们**在这次冒险中跟随我，并且**永远保持学习**！

> **提醒:**你可以在[**这个 GitHub 库找到所有更新的代码！**](https://github.com/dandpz/restfulapi-howto)
> 
> 链接往期文章:[https://medium . com/analytics-vid hya/restful-API-how-to-part-3-testing-8 FD 3 fac 4 E1 CD](/analytics-vidhya/restful-api-how-to-part-3-testing-8fd3fac4e1cd)