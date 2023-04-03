# 伪文件服务器与谷歌云存储和 GoLang。

> 原文：<https://medium.com/analytics-vidhya/pseudo-file-server-with-google-cloud-bucket-and-golang-7dacce1a3e5a?source=collection_archive---------6----------------------->

![](img/7d4b1a699a233ded1939d6dcc3032067.png)

**“你什么都不知道琼恩·雪诺”——耶哥蕊特**

# 介绍

我们知道文件服务器在 web 应用程序中的重要性。在很多情况下，我们需要制作网络资源(图片、文件等)。)可由客户端访问。

文件服务器通常不代表其客户端工作站执行计算任务或运行程序。
文件服务器常见于办公室，用户使用[局域网](https://en.wikipedia.org/wiki/Local_area_network)连接他们的客户端计算机以访问资源。
它是存放贵公司数据的地方。

![](img/f15ec35fa01a71145b7583b92bea88fd.png)

当你内心发生变化时，你周围的事情也会发生变化。 **―未知**

在技术和公共云的新兴时代，类似的设置可以迁移到公共云。

谷歌云产品(GCP)以谷歌云存储(GCS)的形式提供了这样一种产品。

**谷歌云存储(GCS)** 是一个 [RESTful](https://en.wikipedia.org/wiki/Representational_state_transfer) [在线文件存储](https://en.wikipedia.org/wiki/Online_file_storage) [web 服务](https://en.wikipedia.org/wiki/Web_service)用于存储和访问[谷歌云平台](https://en.wikipedia.org/wiki/Google_Cloud_Platform)基础设施上的数据。该服务将谷歌云的性能和可扩展性与高级安全和共享功能相结合。这是一个*基础设施即服务* ( [IaaS](https://en.wikipedia.org/wiki/Infrastructure_as_a_service#Infrastructure) )，堪比[亚马逊 S3](https://en.wikipedia.org/wiki/Amazon_S3) 在线存储服务。

GoLang 因其易用性和高度并发性而越来越受欢迎。

在本文中，我们将使用 GoLang 构建一个简单的 web 服务客户端，它将使用 Google Cloud Bucket 作为存储。
我们将构建几个端点，用于从 Google bucket 存储和检索文档。

那么，让我们开始吧…

![](img/94bc3d0e8f8e40b9d53e0a887ee36fec.png)

“只要记得打开灯，即使在最黑暗的时候也能找到幸福。”――阿不思·邓布利多

# **设置**

我们将为 Golang 使用 GCS 云库。
您可以通过在 mac 或等效 CLI 上的终端中运行以下命令来下载依赖项。

1.  获取谷歌云图书馆。

```
go get -v cloud.google.com/go/storage
```

2.从你的[谷歌控制台账户](https://cloud.google.com/docs/authentication/getting-started)下载服务账户凭证。

3.设置以下环境变量，以便您的客户端可以访问服务帐户。

```
Set **export****GOOGLE_APPLICATION_CREDENTIALS=[PATH]**as ENV variable, where [PATH] is the location of google credentials on file system.
```

# **功能**

我们将添加以下功能作为云集成的一部分。

*   上传功能，将采取多种形式的数据作为输入，并上传到谷歌存储。
*   下载功能，它将资源链接作为请求的一部分，并通过从 GCS 获取资源来返回相应的资源。
*   此外，我们将创建一个单例存储客户机，作为连接 API 和 GCS 的桥梁。

![](img/bfacb0d413fc4b820ca409c88a58eb3d.png)

世界是一本书，那些不旅行的世卫组织人只读了其中的一页~圣奥古斯丁

> 创建将用于所有云交互的单一存储客户端。

# 文件上传

上传功能实现。这将把多表单数据作为请求的一部分，并将其存储到 GCS。

![](img/60478e9625b46ea47284b0195a20f099.png)

**“我们不是从祖先那里继承了地球，我们是从子孙那里借来的。”**

# 文件下载

这是下载功能，它将资源名作为路径参数。我们将返回一组响应头作为响应的一部分。
假设客户端在大多数情况下是浏览器，我们可以设置一个名为 *Content-Disposition* 的附加头，该头将以`<disposition-type>;filename=<file-name>` 的形式取值，其中`<disposition-type>` 可以取值为`**inline**`以内联方式呈现内容，也可以取值为`**attachment**` 以附件方式下载内容。

# 路线

我们将分别为上传和下载端点配置两条路由。

# 最后

这是最后的主文件。试试吧！！！

![](img/3c521c5f9a56ea5e0a341697533fe609.png)

# 谢谢大家！编码快乐！！！