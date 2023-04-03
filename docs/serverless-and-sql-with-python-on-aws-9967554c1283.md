# AWS 上使用 Python 的无服务器和 SQL

> 原文：<https://medium.com/analytics-vidhya/serverless-and-sql-with-python-on-aws-9967554c1283?source=collection_archive---------8----------------------->

![](img/9937dddeefb4b33cde837a99c11e62b3.png)

卡斯帕·卡米尔·鲁宾在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

在我们最近的一个项目中，由于数据及其关系的性质，我们需要一个经典的 SQL 数据库。只要有可能，我们就为我们的云项目使用无服务器方法。因此，我们决定在 AWS 环境中构建支持 SQL 的 lambdas。让我告诉你一些我们是如何做到这一点的。

# 设置

我们从以下组件开始:

*   [无服务器](https://www.serverless.com/)带 AWS Lambda 和 API 网关
*   AWS RDS 上的 MySQL 社区数据库

为了扩大和缩小规模，你应该考虑使用 [AWS Aurora 无服务器](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/aurora-serverless.html)。查看这篇关于[为无服务器应用](https://www.serverless.com/blog/choosing-a-database-with-serverless/)选择数据库的好文章。

在我们的无服务器项目中，我们试图尽可能清晰地分离不同的模块，以便每个模块都有明确的职责。对于这个展示，我们将 *dogs* 存储在数据库中，并通过 lambda 函数访问它们。

```
project
  common
    - db.py
  data
    model
      - dog_data_object.py
    - dog_data_access_service.py
  function
    common
      - abstract.py
    - dog.py
  - serverless.yml
```

# 无服务器配置

我们使用 serverless.yml 文件将数据库连接变量设置为环境变量。这有助于我们稍后运行不同的环境概要文件(dev、int、prod……)，并允许我们使本地测试更加灵活。在这个例子中，我们仅仅创建了一个 *dog* GET 函数来访问数据库。

注意，我们使用了[server less-python-requirements](https://www.npmjs.com/package/serverless-python-requirements)，这使得管理需求更加容易。

# 与数据库的连接

为了连接到我们的 MySQL 数据库，我们使用强大的 python 库 [SQLAlchemy](https://www.sqlalchemy.org/) 。Alchemy 是管理数据库会话的一个非常好的工具，可以帮助您进行对象关系映射。

为此，我们创建了一个单独的 python 模块来管理数据库引擎，提供[会话](https://docs.sqlalchemy.org/en/13/orm/session_basics.html#what-does-the-session-do)和 Alchemy JSON 编码器。编码器可以得到增强，以解决递归和循环对象结构(比较这个伟大的[职位](https://stackoverflow.com/questions/5022066/how-to-serialize-sqlalchemy-result-to-json/41204271))。

# 数据访问模块

为了将功能和数据清楚地分开，我们引入了一个提供数据库对象的数据访问模块。为此，您需要定义映射到某个表的数据库对象。

另一个数据访问模块提供了对数据库对象的访问。注意，我们将 SQLAlchemy 会话作为参数传递，因为该会话是在函数级别上创建和管理的，并且还可以用于多个数据库访问调用。

# 该功能

现在我们想从函数中访问数据库。为此，我们创建了一个小的抽象处理程序，它包装我们的函数调用，并在函数成功运行后获取数据库会话并提交会话。

这个抽象处理程序提供了一个很好的方法来抽象像会话处理、用户管理或其他适用于您的函数集的一般事情。注意，我们也可以在这里使用装饰方法。

我们函数的实际逻辑非常简单，只专注于获取数据并返回数据的实际业务逻辑。

# 摘要

用无服务器方法构建基于 SQL 的应用程序在业界肯定还不是默认的，但我相信它在未来会越来越强大。从实际的“服务器”中抽象出无服务器的美妙之处，使得 DevOps 世界中的软件工程师变得更加容易，以至于更多的经典应用程序(通常依赖于 SQL 数据库)最终也会考虑迁移到无服务器。

除了无服务器的巨大优势，在构建基于 SQL 的 lambda 函数时，还需要考虑两个主要的架构点:可伸缩性和关注点分离

无服务器方法是使应用程序易于伸缩的好方法，但是如果数据库不能伸缩，这就帮不了你。你需要从一开始就考虑这一点，并找到合适的方法。显然，本文中的方法不太具有可伸缩性，但是有许多方法可以引导您。这里有一篇关于这个话题的不错的[文章](https://www.red-gate.com/simple-talk/cloud/cloud-data/designing-highly-scalable-database-architectures/)。

无服务器可以很好地将您的服务完全分离，并迁移到微服务架构，但请记住，您的数据组织需要代表您的微服务架构。因此，明智地决定哪一个无服务器功能负责哪一部分数据。让数百个功能分离您的服务是没有意义的，但是一个巨大的数据库由所有功能访问，没有任何分离。相应地分离您的数据，并为您的功能承担干净的数据责任。

我同意 Paul Johnston 的[陈述](/@PaulDJohnston/serverless-and-microservices-a-match-made-in-heaven-9964f329a3bc)中的观点，即无服务器方法需要改变思维方式，需要分享更多的例子和经验，这样才能获得更大的动力。因此，分享你的观点和你对使用无服务器方法构建基于 SQL 的应用程序的看法。