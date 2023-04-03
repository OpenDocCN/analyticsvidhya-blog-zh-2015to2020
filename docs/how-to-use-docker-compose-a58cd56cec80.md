# Docker 撰写简介

> 原文：<https://medium.com/analytics-vidhya/how-to-use-docker-compose-a58cd56cec80?source=collection_archive---------9----------------------->

![](img/17535f002dee5785d65cb96662074732.png)

Docker 是一个令人惊叹的集装箱化解决方案，它可以帮助您无缝地运输您的代码，而不管它在什么环境下运行。Docker 提供了在松散隔离的环境(称为容器)中打包和运行应用程序的能力。

交付 Docker 容器的支柱是 **docker-compose** 。作为开发人员，您可以使用 docker-compose 命令来设计和运行多容器 docker 应用程序。

使用 docker-compose 包括 2 个步骤，

1.  创建一个 **docker-compose.yml** 文件，定义构成你的应用的服务
2.  运行 **docker-compose up** 来创建并启动你的应用

**在开始**之前，你必须在你的机器上安装[对接器](https://docs.docker.com/get-docker/)和[对接器组合](https://docs.docker.com/compose/install/)。

**docker-compose** 获取一个. yml 文件并执行其中存储的“配方”。默认情况下，它会在当前目录中查找一个 **docker-compose.yml** 文件。典型的 **docker-compose.yml** 文件如下所示:

```
version: '3'services:
  sql-server:
    image: mcr.microsoft.com/mssql/server
    restart: always
    environment:
      ACCEPT_EULA: y
      SA_PASSWORD: <your_password>
  adminer:
    image: adminer
    restart: always
    ports:
      - 8080:8080
```

这里，我们有 2 个容器运行在我们的应用程序中，即 SQL Server 和 Adminer。

的。上面的 yml 文件创建了一个运行 Microsoft SQL Server 的 Docker 容器。“图像”指定 Docker 需要为容器下载什么图像。您可以在 [Docker Hub](https://hub.docker.com/) 找到所有 Docker 图片的完整列表。

应用程序中的容器可以通过名称相互访问。从上面看。yml 文件，你可以使用 **sql-server:1433** 来访问 SQL Server。

Docker hub 映像附带了某些环境变量，您可以根据自己的需求进行配置，例如上面的。yml 文件，我们已经配置了一个服务帐户密码。浏览 Docker 映像的文档，了解它为您提供了哪些配置选项。

docker 编写文件的主要部分，

docker-compose 文件中最常用部分的描述

容器级选项，

docker-compose 中容器可用的配置选项

一些常用的 docker-compose 命令，

**创建&开始容器**

```
docker-compose up
```

**列出集装箱**

```
docker-compose ps
```

**删除已停止的集装箱**

```
docker-compose rm
```

**指定一个. yml 文件**

```
docker-compose -f docker-compose.yml
```

docker-compose 可用选项的完整列表

```
Define and run multi-container applications with Docker.

Usage:
  docker-compose [-f <arg>...] [options] [COMMAND] [ARGS...]
  docker-compose -h|--help

Options:
  -f, --file FILE             Specify an alternate compose file
                              (default: docker-compose.yml)
  -p, --project-name NAME     Specify an alternate project name
                              (default: directory name)
  --verbose                   Show more output
  --log-level LEVEL           Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  --no-ansi                   Do not print ANSI control characters
  -v, --version               Print version and exit
  -H, --host HOST             Daemon socket to connect to

  --tls                       Use TLS; implied by --tlsverify
  --tlscacert CA_PATH         Trust certs signed only by this CA
  --tlscert CLIENT_CERT_PATH  Path to TLS certificate file
  --tlskey TLS_KEY_PATH       Path to TLS key file
  --tlsverify                 Use TLS and verify the remote
  --skip-hostname-check       Don't check the daemon's hostname against the
                              name specified in the client certificate
  --project-directory PATH    Specify an alternate working directory
                              (default: the path of the Compose file)
  --compatibility             If set, Compose will attempt to convert deploy
                              keys in v3 files to their non-Swarm equivalent

Commands:
  build              Build or rebuild services
  bundle             Generate a Docker bundle from the Compose file
  config             Validate and view the Compose file
  create             Create services
  down               Stop and remove containers, networks, images, and volumes
  events             Receive real time events from containers
  exec               Execute a command in a running container
  help               Get help on a command
  images             List images
  kill               Kill containers
  logs               View output from containers
  pause              Pause services
  port               Print the public port for a port binding
  ps                 List containers
  pull               Pull service images
  push               Push service images
  restart            Restart services
  rm                 Remove stopped containers
  run                Run a one-off command
  scale              Set number of containers for a service
  start              Start services
  stop               Stop services
  top                Display the running processes
  unpause            Unpause services
  up                 Create and start containers
  version            Show the Docker-Compose version information
```

对 docker-compose 的介绍到此结束

你可能也喜欢，

[](/analytics-vidhya/setup-elasticsearch-kibana-via-docker-ce21cf6f5312) [## 通过 Docker 设置 Elasticsearch & Kibana

### 首先，你必须在你的机器上安装 Docker 和 Docker compose。如果你没有它…

medium.com](/analytics-vidhya/setup-elasticsearch-kibana-via-docker-ce21cf6f5312)