# 码头集装箱

> 原文：<https://medium.com/analytics-vidhya/docker-container-7a3694bc00fc?source=collection_archive---------15----------------------->

![](img/01502593960134524e76f02bd9f06cda.png)

弗兰克·麦肯纳在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

**什么是 Docker:**

您能回忆起有多少次您遇到了代码在开发人员机器上正常工作而在测试环境中不起作用的问题吗？

fundamental 的 Docker 解决了代码在一个平台上工作而在另一个平台上不工作的问题。

Docker 存在于整个工作流程中(即设计、开发、部署和测试/发布)，但它主要用于部署阶段。

Docker 使应用程序部署过程变得非常简单和高效，并解决了许多与部署应用程序相关的问题。

Docker 是全球领先的集装箱平台。让我们明白这一点，

典型的软件包括:

-前端框架

-后端工作人员

-数据库

-Env/Libs 依赖项

这些部分中的每一个都应该在不同的平台上正确工作(即每个软件组件都与每个硬件兼容)。

该容器允许开发人员将应用程序与它需要的所有部分(如库和其他依赖项)打包在一起，并作为一个可能的包运送。

**Docker 如何工作:**

在 Docker 中，workflow developer 将在 Docker 文件中定义所有的应用程序及其依赖关系。

Dockerfile:描述创建 docker 图像的步骤(例如，它就像是包含制作每道菜所需的所有配料和步骤的菜谱)

这个 docker 文件用于创建一个包含所有应用程序及其需求和依赖项的 docker 映像，当您创建 Docker 映像时，您将获得 Docker 容器。

Docker 容器:docker 映像的运行时实例。

Docker Hub: docker 图像也存储在一个名为 Docker Hub 的在线云存储库中。这个存储的图像可以被拉到环境中来创建容器。

Docker 是一个容器平台，让我们了解虚拟化和容器化的区别，

虚拟化:

在虚拟化中，我们有一个名为 hypervisor 的软件，使用它我们可以在主机系统上创建多个操作系统。

主机操作系统->虚拟机管理程序->具有不同来宾操作系统的多个虚拟机，因此主机操作系统上有大量开销。

集装箱化:

在容器化中，我们有一个容器引擎，其中我们没有单独的操作系统，但有不同的容器，其中有应用程序及其所有依赖项，它将使用主机操作系统。在这里，空间、内存和其他资源是不固定的，它将根据应用程序的需求来使用，因为它是轻量级和快速的。

Docker 有客户端-服务器架构，其中命令行界面是客户端，我们有 docker 服务器/守护程序，它将拥有所有容器，服务器接收所有命令或 rest API 调用中的请求，所有 docker 客户端和服务器形成 docker 引擎。

**码头工人的好处:**

-解决了代码只能在一个系统上运行而不能在其他系统上运行的问题。

-只构建一次应用程序:任何容器中的应用程序都可以在任何安装了 docker 的系统上运行，因此无需在不同的平台上多次构建配置应用程序。

-使用 docker，您可以在一个容器中测试应用程序，并将其运输到容器中，这意味着您测试的环境与生产中运行的环境是相同的。

-可移植性:Docker 容器可以在任何平台上运行。

-版本控制:像 Git docker 一样有内置的版本控制。

-隔离:有了 docker，每个应用程序都可以使用自己的容器独立工作。

-生产力:Docker 支持更快、更高效的部署。

**Docker 命令:**

基本命令:

- docker 版本:给出关于 docker 的客户端和服务器(即引擎)的信息。

-docker–v/docker-version:给出 docker 的版本。

- docker info:关于系统上运行的 docker 的信息。

-docker–help:对于获取有关其他 docker 命令的信息非常有用。

- docker 登录:使用 id 和密码登录 docker hub。

图像命令:

- docker images:给出所有图像的列表。

- docker pull imagename:从库中获取指定的图像。

- docker rmi:删除一个或多个 docker 映像。

容器命令:

- docker ps:用于列出容器。

- docker run:在新容器中运行一个命令，如果指定的容器不在本地，它将从库中取出它。

- docker start containerid:启动容器。

- docker stop containerid:停止集装箱。

系统命令:

- docker stats:关于运行容器的详细信息(内存、使用等。)

- docker 系统 df:docker 映像、容器等的磁盘使用情况。

- docker 系统清理:删除未使用的数据。

**Dockerfile:**

带有构建图像说明的文本文件。Docker 图像创建的 docker 文件自动化。

Docker 文件-> docker 构建->Docker 图像

步骤 1)创建一个名为 Dockerfile 的文件

步骤 2)在 Dockerfile 中添加指令

步骤 3)构建 docker 文件以创建图像

步骤 4)运行映像以创建容器

**Docker 撰写:**

docker compose 在您的项目架构类似微服务时非常有用，例如电子商务平台，它可以分为不同的服务，如帐户服务器、产品服务器、购物车服务器等。

-用于定义和运行多容器 docker 应用程序的工具。

-使用 YAML 文件配置应用程序服务。

-可以用一个命令启动所有服务:docker compose up。

-可以用一个命令停止所有服务:docker compose down。

-可以根据需要扩展选定的服务。

步骤 1)安装 docker compose。

docker compose–v:用于获取 docker compose 的版本。

步骤 2)在任何位置创建 docker 合成文件。

docker-撰写。Yaml

步骤 3)通过命令检查文件的有效性，

docker 撰写配置

步骤 4)运行 docker-compose。通过命令的 yaml 文件。

docker-compose up–d(用于分离模式)

步骤 5)通过命令关闭应用程序。

docker-向下撰写。

缩放命令:docker-compose up–d–scale service name = countofcontainers。

**Docker 音量:**

卷是保存 docker 容器生成和使用的数据的首选机制。

这对于从容器中分离存储是有用的。

在不同的容器之间共享卷(存储/数据)。

我们可以将卷附加到容器，在删除容器后，数据不会被删除。

- docker volume create volumename:创建 docker 卷。

- docker 卷 ls:列出卷。

- docker volume inspect volumename:获取指定卷的详细信息。

- docker volume rm volumename:删除指定的卷。

- docker 卷清理:删除所有未使用的卷。

默认情况下，在容器中创建的所有文件都存储在可写的容器层上，当容器不再运行时，数据不会持续存在。

容器的可写层与运行容器的主机紧密相连。你不能轻易地将数据转移到其他地方。Docker 为容器提供了两个选项来将文件存储在主机中，这样，即使在容器停止卷和绑定挂载之后，文件也会持久化。

卷存储在由 Docker 非 Docker 进程管理的主机文件系统的一部分中，不应修改文件系统的这一部分。绑定安装可以存储在主机系统上的任何位置。Docker 主机上的非 Docker 进程或 Docker 容器可以随时修改它们。

在绑定装载中，文件或目录通过其在主机上的完整路径来引用。卷是在 Docker 中保存数据的最佳方式。卷由 Docker 管理，并与主机的核心功能相隔离。给定的卷可以同时装入多个容器中。当没有正在运行的容器正在使用卷时，Docker 仍然可以使用该卷，并且不会自动删除该卷。您可以使用 docker 卷清理删除未使用的卷。当您装入一个卷时，它可能是命名的或匿名的。匿名卷在第一次装入容器时没有明确的名称。卷还支持使用卷驱动程序，这允许您将数据存储在远程主机或云提供商等设备上。

例如，将卷附加到 Jenkins 的两个实例，如下所示，

docker run-name myjenkinstestsystem 1-v myvolume 1:/var/Jenkins _ home-p 8080:8080-p 50000:50000 Jenkins

docker run-name myjenkinstestsystem 2-v myvolume 1:/var/Jenkins _ home-p 9090:8080-p 60000:50000 Jenkins

这两个 Jenkins 系统共享一个名为 myvolume1 的数据卷。