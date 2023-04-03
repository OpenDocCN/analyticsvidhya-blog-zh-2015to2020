# 对一个简单的 Python Flask 应用程序进行 Dockerizing

> 原文：<https://medium.com/analytics-vidhya/dockerizing-a-simple-python-flask-application-ddc509da6781?source=collection_archive---------4----------------------->

对一个简单的 python flask 应用程序进行 docker 化是一项简单易行的任务，有助于理解 docker 文件的基础知识，并掌握 python-flask 的概念。对于这个任务，我们将使用一个基本的 **hello world** flask 应用程序

# DOCKER 文件的解释

下面是为应用程序设置 docker 映像的完整 docker 文件。

```
FROM python:alpine3.7
MAINTAINER Syed Saad Ahmed, syedsaadahmed2094.sa@gmail.com
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python ./index.py
```

## `FROM python:alpine3.7`

FROM 命令用于说明您打算将什么操作系统用作基本映像。在这一行中，我们从 alpine 3.7 python image 继承了我们的图像，因为我们将部署 python 应用程序，所以我们使用了 alpine python image。

## `MAINTAINER Syed Saad Ahmed, syedsaadahmed2094.sa@gmail.com`

特定文档维护者的姓名或其他信息。

## `COPY . /app`

要将文件从 Docker 主机复制到 Docker 映像，可以使用 copy 命令。我们正在将目录中的所有文件复制到位于 */app* 的 docker 映像中

## `WORKDIR /app`

WORKDIR 允许您在 Docker 构建映像时更改目录，新目录仍然是其余构建指令的当前目录。

## `RUN pip install -r requirements.txt`

在构建 Docker 映像时，您可能需要运行命令，例如安装应用程序和软件包作为映像的一部分。

## `EXPOSE 5000`

您将使用 EXPOSE 命令来选择哪些端口应该可用于与容器通信。当运行 Docker 容器时，可以传入-p 参数 publish，它类似于 EXPOSE 命令。

## `CMD python ./index.py`

Docker 只能运行一个 CMD 命令。因此，如果您插入两条或多条 CMD 指令，Docker 只会运行最后一条指令，即最近的一条指令。ENTRYPOINT 类似于 CMD，但是，您可以在启动时运行命令，它不会覆盖您在 ENTRYPOINT 定义的指令。

# 执行应用程序

## 首先构建 docker 映像

`docker build -t sample_image .`

这里，sample_image 是 Docker 图像的名称。你可以给它另一个名字。圆点(。)表示您正在处理的文件位于当前目录中。

## 运行 docker 容器

`docker run -itd --name python-app -p 5000:5000 sample_image`

在这里，python-app 是你将要启动的容器的名字，docker run 有-i、-t、-d 等多个标志，你可以在 docker 官方文档[这里](https://docs.docker.com/engine/reference/commandline/run/)探究一下，-p 标志用于向主机发布/公开容器的端口。

## 这是输出

`[https://localhost:5000/](https://localhost:5000/)`

## 检查应用程序的日志

`docker logs <container_id>`

此外，这里是 github 资源库的 URL，用于完整的应用程序和设置。https://github.com/syedsaadahmed/Dockerizing_Python_App
T3