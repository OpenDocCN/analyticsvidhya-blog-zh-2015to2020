# 用 Python 中的 gRPC 为计算机视觉应用编写简单的一元 RPC

> 原文：<https://medium.com/analytics-vidhya/writing-simple-unary-rpc-for-machine-learning-applications-with-grpc-in-python-8c26e2065f7b?source=collection_archive---------9----------------------->

编写可伸缩的应用程序需要更好的设计原则。假设我们想设计一个电子商务应用程序，它可能有一组不同的客户端(浏览器、移动设备等。)，也是在开发阶段的中期，需要编写一个基于深度学习的推荐系统来处理图像以提取产品信息，以改善用户体验，但主要应用程序是用 PHP 编写的。为了使设计尽可能模块化，最好选择基于微服务的设计策略…