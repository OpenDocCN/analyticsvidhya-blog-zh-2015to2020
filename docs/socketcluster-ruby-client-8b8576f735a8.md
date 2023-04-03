# socketcluster.io 和 Ruby 客户端简介

> 原文：<https://medium.com/analytics-vidhya/socketcluster-ruby-client-8b8576f735a8?source=collection_archive---------3----------------------->

*全码在*[*【https://github.com/OpenSocket/socketcluster-client-ruby】*](https://github.com/OpenSocket/socketcluster-client-ruby)

我一直对实时框架和基于聊天的应用程序感兴趣。我在寻找可以用来构建基于聊天的应用程序的可用框架。有很多可用的选项，如 Pubnub、Pusher、Firebase、socket.io 和 socketcluster。

Pubnub、Pusher 和 Firebase 等基于云的服务的问题是在后端编写定制逻辑和最终定价方面的限制。socket.io 似乎有更大的开发者社区，在设置环境和灵活的 API 架构方面相当不错。然而，我需要比这更多的东西；一个易于扩展并能处理数百万并发连接的框架。最后，我找到了一个[**socket cluster**](http://socketcluster.io)**框架作为最佳解决方案，它由最快的 WebSocket 引擎 [uwebsockets](https://github.com/uWebSockets/uWebSockets) 提供支持。**

**SocketCluster 是 Node.js 的一个开源实时框架，它既支持直接的客户端-服务器通信，也支持通过 pub/sub 通道的组通信。它被设计成可以轻松扩展到任意数量的进程/主机，是构建聊天系统的理想选择。SocketCluster 旨在简化构建可伸缩实时系统的过程。**

**SocketCluster 是一个快速、高度可伸缩的 HTTP 和 WebSockets 服务器环境，它允许您构建多进程实时系统，该系统可以利用机器/实例上的所有 CPU 内核。它消除了必须将节点服务器作为单个进程运行的限制，并通过自动从工作崩溃中恢复来使后端具有弹性。**

**SocketCluster 的工作方式类似于一个一直延伸到浏览器/物联网设备的发布/订阅系统，它只向真正需要的客户端发送特定事件。SocketCluster 被设计为既可以垂直伸缩(作为一个进程集群)，也可以水平伸缩(多个机器/实例)。**

****技术特征****

*   **随着 CPU 内核和工作人员的增加而线性扩展。**
*   **跨多台机器水平扩展。**
*   **在客户端和后端都具有弹性—流程崩溃、失去连接和其他故障都可以无缝处理。**
*   **支持在传输过程中压缩消息的自定义编解码器。**
*   **支持发布/订阅通道和直接的客户端-服务器通信(对于 RPC)。**
*   **符合 JSON Web 令牌(JWT)的身份验证引擎。**
*   **如果客户端套接字失去连接，它们会自动重新连接(默认情况下)。**

**有什么好处？**

*   **需要呈现实时数据的单页应用程序。**
*   **金融、加密货币和其他区块链应用。**
*   **聊天机器人和其他聊天相关的应用。**
*   **使用 React Native 或 Ionic/Cordova 等网络技术构建的移动应用。**
*   **任何需要扩展到数百万用户的实时应用或服务。**

**我们使用 **Ruby 客户端完成了**实时 SDK**的设计。这是 socketcluster.io 的 ruby 客户端****

**该客户端提供以下功能:**

*   **易于设置和使用**
*   **支持发送和监听远程事件**
*   **自动重新连接**
*   **发布/订阅**
*   **认证(JWT)**
*   **可用于所有服务器端功能的广泛单元测试**
*   **对 ruby >= 2.2.0 的支持**

**这个网站也是开源的，欢迎在 GitHub 上投稿。**

**如果你不熟悉 Websockets、socket.io 和 socketcluster.io，我建议你阅读以下链接:**

**[https://www . Li node . com/docs/development/introduction-to-web sockets/](https://www.linode.com/docs/development/introduction-to-websockets/)**

**[https://socketcluster.io/](https://socketcluster.io/)**

**就是这样。希望这有所帮助。**

**如果你喜欢这篇文章，我会非常感激你把它发给朋友，或者在推特或脸书上分享。谢谢大家！**