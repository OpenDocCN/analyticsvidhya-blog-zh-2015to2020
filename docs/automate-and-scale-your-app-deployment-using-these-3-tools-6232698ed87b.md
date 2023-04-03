# 使用这 3 种工具自动化和扩展您的应用部署

> 原文：<https://medium.com/analytics-vidhya/automate-and-scale-your-app-deployment-using-these-3-tools-6232698ed87b?source=collection_archive---------26----------------------->

![](img/8f6f5dd986cd5fb273c4c4515d37fef2.png)

SpaceX 在 [Unsplash](https://unsplash.com/s/photos/rocket?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的

作为一名软件工程师，我喜欢构建新的应用程序，为每个需要实现的功能考虑最佳的解决方案，并以优化和可维护的方式设计它们的架构。

从零开始构建一些东西，并看到它如何在我们完成的产品中转化，这真是太棒了。事实证明，当你有一个伟大的团队可以支持你，并帮助你获得新的观点和新的想法时，那就更好了。

在这个新软件构建完成后，是时候将它部署到我们的客户那里了。虽然这是一项非常重要的任务，但我想在这里花尽可能少的时间，然后回去做我最擅长的事情:设计和编写新的软件产品。

在 [YooniK](https://yoonik.me) 我们的大多数软件解决方案都以 REST APIs 的形式提供，其中很大一部分依赖于深度学习模型来处理数据和交付结果。因此，使它们在生产环境中可伸缩是极其重要的。

这是我们用来部署和扩展 API 的 3 个工具:

## 1.Docker 容器

通过使用 Docker，我们可以将一个应用程序及其所有依赖项打包到一个虚拟容器中。这提供了在多种计算环境中运行软件的灵活性，例如内部、公共云或私有云中。Docker 容器映像是一个独立的包，包含运行应用程序所需的一切。

## 2.库伯内特星团

在打包我们的应用程序之后，我们将它部署在 Kubernetes 集群中。Kubernetes 是一个开源平台，用于自动化部署、扩展和管理容器化的应用程序。在生产环境中，我们需要管理运行应用程序的容器，并确保没有停机。所有这些都可以由 Kubernetes 来处理。

在其他重要特性中，Kubernetes 还提供了服务发现、负载平衡以及自动化的部署和回滚。

## **3。亚马逊 EKS**

最后，在用 Docker 封装了我们的软件产品并为 Kubernetes 设置了配置文件之后，我们就为部署做好了准备。

Amazon Elastic Kubernetes Service(EKS)提供完全托管的 Kubernetes 基础架构，安全、可靠且可扩展。这是让您的应用投入生产的绝佳方式，无需担心安全性、可用性以及扩展更多计算能力的需求，因为我们通过 EKS 获得了所有这些。

感谢阅读，享受你一天中的每一刻！

*最初发表于*[*【www.vitorpedro.com】*](https://vitorpedro.com/automate-and-scale-your-app-deployment-using-these-3-tools/)*。*