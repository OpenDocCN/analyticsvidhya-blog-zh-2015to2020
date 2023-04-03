# Docker 图像管理的基础知识(2/2)

> 原文：<https://medium.com/analytics-vidhya/the-very-fundamentals-of-docker-images-management-2-2-f2a7ce3ae546?source=collection_archive---------34----------------------->

在本系列的这一部分，我将讨论`Docker Registries`，它是如何工作的，pull 和 push 命令，以及一些关于运行容器的内容。

“T1”是一个无状态的、高度可伸缩的应用程序，它存储并允许你分发“T2”。这个注册中心是开源的，但是您可以找到各种其他实体作为注册中心工作。

这里有一个简单的列表

1.  `Docker Registry`(默认)，本地开源注册表。
2.  `Docker Trusted Registry` (DTR)，Docker 企业版的一部分，高端全功能可信注册中心
3.  `Docker Hub`，一个基于云的注册中心
4.  第三方注册中心 w/ `VMWare`(我认为它集成了`VSphere`和`Google Cloud`中的`Container-Registry`

我确信还有更多，但实际上，大多数人使用 Docker Hub，因为它是不属于 EE 计划的普通开发人员的 Docker 默认设置。

请转到 Docker hub 链接并创建一个帐户。

## 部署 Docker 注册中心

我将尝试解释从运行容器开始到推拉过程的整个过程，所以我们开始吧。

1.  让我们运行一个容器，一个像这样的样本容器

`docker run -d -p 5000:5000 --restart=always --name registry registry:2`

`-d`标志用于将容器作为`daemon`运行，`-p`用于生成`port forwarding`，`--restart`标志(实际上非常具有描述性)，用于命名容器的`--name`和`registry:2`是基本图像及其标签的模拟名称(或可能使用的版本)。

2.如果你愿意，你可以给图像加标签

3.一旦我们证明一切正常，就让我们将我们的映像推送到本地注册中心

`docker push <d_hub_username>/registry:2`

您可能需要对`Docker registry`做一个`CLI login`，然后过程将正常开始。这是在`docker login`的帮助下完成的，设置用户和密码仅此而已。

如果您需要来自`Hub`的推送映像，您将需要使用`pull`命令将其下载到本地注册表。

`docker pull <d_hub_username>/registry:2`

如果您是`Docker Hub`的`DTR registry`(商业付费选项)用户，您可以访问更多功能，如用户私人注册、组织和群组创建、安全扫描等。

完成后，您可以使用`docker logout`注销

小贴士:

1.  `pull`和`push`是否登录
2.  如果你想要更多的隐私，付费使用 DTR 服务。
3.  其他指定的注册服务有自己的方式来拉和推 docker 图像，你需要检查他们的文件。

有一种叫做`Docker Content Trust`的服务，这种类型的服务让你用私钥对图像进行数字签名，用于图像所有者验证，但这在 Docker EE 版本上可用。这是 Docker 公证人的开源版本，请随意查看。

这部分很短，很抱歉让你失望了。但这是 Docker 图像管理的最基本的基础。容器、网络、卷、服务、部署等有一个完整的整体。

希望这对您有所帮助。如果你发现任何错误，请随时改正。这是我的亲身经历。

快乐编码:)