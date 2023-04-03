# 实践中的集装箱化

> 原文：<https://medium.com/analytics-vidhya/containerization-in-practice-47516a0f283a?source=collection_archive---------20----------------------->

## 第二部分:数据科学项目如何受益于 Docker？

![](img/61a8672fce91593f398e43ad7710a010.png)

照片来自[https://www.pexels.com/@cottonbro](https://www.pexels.com/@cottonbro)

这篇文章建议您如何在数据科学项目的不同阶段受益于 Docker，从研究到部署，再到生产中的数据和应用程序监控。特别是，我讨论了将 Docker 技术应用于

*   检查现有解决方案
*   准备开发环境
*   创建测试环境(包括测试数据库)
*   监控您的数据和应用
*   部署您的项目
*   共享您的项目

# 检查现有解决方案

一个新的数据科学项目通常从考察领域和测试现有解决方案对所述问题的可行性开始。有了 Docker，研究成果可以很容易地共享，并应用于其他问题和数据。

机器学习工具和算法有大量现成的 docker 图像。为了验证这一说法，我查阅了 [Mybridge](/@Mybridge) 在这篇[文章](https://medium.mybridge.co/amazing-machine-learning-open-source-tools-projects-of-the-year-v-2019-95d772e4e985)中公布的 2018 年最受欢迎的开源 ML 项目的文档。事实上，32 个项目中有 13 个(40%)有 docker 文件或 docker 图像可用。注意，我从这两个类别中排除了作为包可用的项目。

也就是说， [Mask R-CNN](https://github.com/facebookresearch/Detectron?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more) 、 [Deepvariant](https://github.com/google/deepvariant?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more) 、 [DensePose](https://github.com/facebookresearch/Densepose?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more) 等算法，只差几个命令。这些工具的应用就像(1)在你的机器上安装 [Docker 桌面](https://www.docker.com/products/docker-desktop),( 2)构建或提取 Docker 映像，以及(3)运行容器一样简单。确切的命令及其参数由工具作者提供。此外，您可以使用 docker engine API 的库将它集成到您的项目中，这些库可用于许多编程语言(例如用于 [Python](https://docker-py.readthedocs.io/en/stable/) 、 [Java](https://github.com/spotify/docker-client) 或[)。网](https://www.nuget.org/packages/Docker.DotNet/))。

# 准备开发环境

通常，您可以使用依赖管理工具(如 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) )为您的项目创建甚至访问一个环境。在这种情况下，仔细跟踪依赖关系就足够了。当您的项目需要多个运行时的组合(比如同时使用 Python 和 Java)或者当您的管道的不同阶段依赖于不兼容的依赖项时，使用 Docker 设置一个开发环境可能会有所帮助。

但是，即使在这种情况下，也要考虑使用例如虚拟环境独立地开发组件，并利用您最喜欢的 IDE 和它的所有优势。并且仅出于测试和部署的目的，考虑使用 Docker。一般来说，使用 Docker 来建立一个开发环境会降低速度，使调试(和一般的开发)更加复杂，需要额外的管理，等等。

![](img/a75e0bbbdb416ebf5b88be156fa568a0.png)

图片来自[https://www.pexels.com/@cottonbro](https://www.pexels.com/@cottonbro)

# 创建测试环境

考虑使用 Docker 为您的项目创建一个测试环境和/或测试数据库。

## **测试数据库**

通常，解决方案的输入数据存储在其他地方的数据库中。使用 Docker，您可以轻松地模拟生产数据库，用测试数据填充它，并将其作为独立的服务器运行。这将允许您测试完整的管道，包括读取/写入数据库。Docker hub 上公开了 PostgreSQL、MySQL、MongoDB、Redis 和许多其他数据库的官方图片。

需要考虑两个方面:

*   根据最佳实践的建议，您可以在每次测试时填充数据库，或者在创建时填充一次，例如从之前准备的 CSV 中填充。无论如何，请记住，当数据库容器停止时，数据库以及您的数据将会消失(在这里阅读更多关于数据持久性的内容)。
*   请记住，目标是创建一个尽可能类似于生产的环境。因此，使用与生产环境中数据库迁移相同的工具来创建测试数据库模式。

## **测试环境**

在 docker 容器中执行测试有几个好处:

*   它使您的测试管道可移植且易于应用。例如，您可以在您的本地机器上创建、修改和运行测试，并且仅仅使用修改过的命令，作为 CI/CD 管道的一部分。
*   它可以让你模仿一个几乎和作品一样的环境。
*   此外，它允许你在多种不同的环境下进行测试(例如 Ubuntu 和 Windows)。

一个 docker 映像(或多个映像)将会像您定义开发/生产环境一样被定义(即使用相同的配置、相同的需求文件等)。).当执行时，容器将运行一个脚本来启动测试。

有几个方面需要考虑:

*   在 run 命令中，也就是在启动容器时，对项目和测试项目的源代码进行批量处理会更方便。这样，您就不必在每个项目或测试源更新之后构建新的映像。
*   确保您的管道区分测试/开发/生产，并处理适当的数据源(例如之前创建的数据库容器)和其他特定于环境的配置。

# 监控您的数据和应用

Docker hub 包含各种软件的官方图片，可能对您的 ML 项目有所帮助。例如， [Grafana](https://grafana.com/) 和 [Kibana](https://www.elastic.co/kibana) 等工具的图像可用于检查和可视化数据或/和应用程序的状态。

有了 Docker，这些应用程序可以在几秒钟内在你的本地机器或其他地方启动。使用这些工具，您可以更好地了解数据和日志。您可以毫不费力地创建可视化，列出异常值，跟踪时间和异常，访问系统或数据状态等等。对于数据科学家来说，这可能是一个巨大的好处，因为他们的首要任务是确保系统的输入和输出得到很好的理解和控制。

# 部署您的项目

Docker 极大地简化了应用程序的部署。即使您的开发环境在容器之外(正如我们前面讨论的那样，这种情况很常见)，将您的应用程序容器化以进行部署也是值得的。

开发机器上的容器在测试、试运行、生产或任何其他环境中的工作方式完全相同。在任何服务器上启动应用程序时，不需要进一步安装(docker 引擎除外)。此外，许多云提供商实施了多种选项来极其轻松地部署和管理容器化的应用程序。

# 共享您的项目

与您在项目开始时处理现有解决方案的方式一样，您可以与社区分享您的发现。将应用程序打包到容器中后，您可以将 docker 映像发布到公共存储库中(如 Docker hub)，或者提供一个原始的 Docker 文件，允许用户自己构建一个包含源代码的映像。这样，任何安装了 Docker 的人都可以在几分钟内测试甚至集成你的工具。

![](img/e66177621699b4fc0a6429148d604550.png)

照片来自[https://www.pexels.com/@cottonbro](https://www.pexels.com/@cottonbro)

总之，Docker 是一个非常强大的工具，数据科学项目在许多情况下和各个阶段都可以从中受益。如果应用得当，它不仅能提高开发人员的效率，还能带来更值得信赖的数据科学方法。

在下一篇文章中，让我们看看如何将这样一个项目部署到 AWS，同时选择最合适的服务提供者来支持容器化应用程序。

几个有用的链接:

*   通过 Docker 的全面实用[介绍(从设置您的计算机到将应用程序部署到云)让您的双手变脏。](https://docker-curriculum.com/)
*   如果你想玩 Docker，但还不准备在你的机器上安装它，试试官方的[基于浏览器的 Docker 游乐场](https://labs.play-with-docker.com/)或[kata coda 的免费 Docker 迷你课程](https://www.katacoda.com/courses/docker)。
*   在 Grafana 游乐场玩 Grafana 功能，并检查在 Docker 容器中运行它[有多容易。](https://grafana.com/docs/grafana/latest/installation/docker/)