# 用 Python + Flask 开发强大 API 的技巧和秘密

> 原文：<https://medium.com/analytics-vidhya/tips-and-secrets-for-developing-a-powerful-api-with-python-flask-8c3c3ea0103?source=collection_archive---------11----------------------->

## 我喜欢烧瓶和 Python。这就是我对#MLOps 之后构建高效 API 的需求的热爱。发现一些让我的 API 更强大的技巧和秘密。

![](img/dcc3f2b932bdb9f73facfc376bd847e9.png)

知道如何开发更好的 API

Python 是一种很棒的编程语言，它对于人工智能和数据科学的世界来说非常强大。

自从我在这个世界开始了我的职业生涯，我就试图用 Python 为我的服务、微服务和项目使用最好的工具。

考虑到这一点，API 是为我的客户交付数据科学项目的最佳方式。此外，这也是在企业内部构建微服务环境，并使用多个以集成方式工作的服务来创建新产品的最佳方式。

但是构建一个 API 可能很困难。下面是一些技巧和秘密，它们使我的 API 更加可用，并改善了客户体验。

看看吧！

我们走吧！

# 您应该使用环境变量。但是用的更好！

环境变量并不是新的，它是根据您的工作环境配置项目的好方法。但是构建 API 的一个很好的技巧是创建一个方法或类来调用所有这些变量。

因此:

这样，您可以简单地在代码中将 env 变量作为一个类来使用。它使你更容易改变任何东西，并保持你的工作在多个操作系统和计算机上。

# 发现烧瓶监控仪表板

构建 API 可能简单而快速，至少在您知道如何做的时候是这样——但是当您开始使用一些云和 DevOps 概念时，跟踪和监控您的应用程序可能会有些棘手。我真的很喜欢 Docker 和 Kubernetes，但是使用所有的 *kubectl* 命令来获得日志或 Python 输出是很复杂的。

这就是我在 API 中使用 Flask Monitoring Dashboard 的原因。

[](https://github.com/flask-dashboard/Flask-MonitoringDashboard) [## 烧瓶仪表板/烧瓶监控仪表板

### 一个自动监控 Flask web 服务的仪表板。主要功能*如何使用*现场演示*反馈*…

github.com](https://github.com/flask-dashboard/Flask-MonitoringDashboard) 

这个 Python 库直接与您的 Flask 项目一起工作，如下所示:

这个库可以让你访问一系列关于你的应用的信息，比如 profiler，使用和性能——它甚至可以通过你的 Flask API 的路径来分离。

# 验证您的 Json 模式

**如今，安全是一项规则。句号。**

我在这里强调一下:

> 我并不建议或推荐它是您的应用程序的唯一安全形式。这只是另一个好的补充，好吗？

在我的项目中，我使用了许多不同形式的认证和最佳实践来保证我的 API 的安全。但是第一个简单的安全实现可能是您自己的 Json 模式。

您可以实现一个函数来验证关于以前模式的 json，如下所示:

这样，您就可以实现一个良好的第一层，将那些不了解您的请求或与您的应用程序交流的模式的人拒之门外。

# 使用 APM 洞察。而且，更好地使用它！

我的一些应用程序是使用云服务部署的，比如 Azure。我工作的地方使用 Azure 提供一些项目服务，包括 AKS-Azure Kubernetes 服务。另一个非常棒的工具是 APM、应用性能管理或应用洞察。

[](https://docs.microsoft.com/pt-br/azure/azure-monitor/app/app-insights-overview) [## o queéo Application Insights do Azure？- Azure 监视器

### O Application Insights，um recurso Application Insights，Azure Monitor 的一个特性，éum servio de APM…

docs.microsoft.com](https://docs.microsoft.com/pt-br/azure/azure-monitor/app/app-insights-overview) 

有了它，我们可以跟踪任何日志或错误，我们的 API 的性能，以及任何类型的坏请求。

APM 并不新鲜，我做了哪些不同或者更好的地方？

我用 Python 日志创建了自己的库，并导入到我所有的 Flask 项目中。它对我跟踪我的应用程序帮助很大，即使是在实验室环境中。我的库为 Application Insights SDK 获得了大量导入，还配置了遥测、处理程序和密钥管理。

这可能是集成和统一所有环境中的开发和监控工作的更好方法。

# 如果你有微服务，就把它 dockerize！

Docker，docker，docker…每个人都在谈论它，有一段时间，每个人都在评论这项技术的力量。它很棒，我经常用它。问题是:

> 如何使用 Docker 和 Kubernetes 创建更好的 Python Flask API？

我最近的所有项目都是用 Docker + Kubernetes 交付的。一些提示是:

*   为您的项目使用基础图像来跟踪您的库。对于使用 Python 和 Flask 的 API，我有一些库一直被用作 Flask、Flask Monitoring、Application Insights SDK、Json Schema…；
*   在 docker 文件中正确声明环境变量；
*   不要忘记在 CMD 之前暴露你的端口；
*   了解 kubectl 的基本命令，如 use-context、apply、get pod、scale……当你必须管理你的 API 时，这将会有所不同。

下面是一个简单 Dockerfile 文件的很好的例子:

# 就这样，伙计们！

使用 Python 和 Flask 构建 API 可能是向客户交付项目或服务的一种简单方式，但一些技巧和秘密对于实现更好的解决方案至关重要。

这 5 个是我在开发中经常使用的，并且在最多样化的项目中帮助了我很多。

我知道可能有其他甚至更好的选择来部署 API，但这是对我有用的。

你知道更多的选择或其他方法来获得更好的结果吗？

**请随时评论并告诉我更多信息或保持联系** [**LinkedIn**](https://www.linkedin.com/in/gilvandroneto1991/) **或**[**Github**](https://github.com/gilvandroneto)**！**

评论:)

## 如果这些建议对你有任何帮助，请分享！