# 如何对烧瓶中的限制路线进行评级

> 原文：<https://medium.com/analytics-vidhya/how-to-rate-limit-routes-in-flask-61c6c791961b?source=collection_archive---------1----------------------->

![](img/e36fa46980f96e498392f32715dfd26e.png)

马库斯·斯皮斯克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

[Flask](https://pymbook.readthedocs.io/en/latest/flask.html) 是一个用 python 写的 web 框架。它为您提供了工具、库和技术，允许您构建 web 应用程序。这些 web 应用程序可以是一些网页、博客、wiki，或者大到基于 web 日历应用程序或商业网站。

**什么是限速？** 速率限制允许你控制在给定的时间段内某人访问你的 API 的频率。烧瓶中有一个名为**烧瓶限制器的扩展。**此扩展允许您限制特定路线上的请求数量。每个路线可以有不同的请求限制，也可以不设限制。它通过跟踪用户的 IP 地址来限制用户。

**你为什么要使用它？** 一个 API 端点不应该被无限制的访问。我们需要确保我们的应用程序路由尽可能高效地运行。关注应用程序的性能非常重要，因为我们不希望用户有不好的体验。它还有助于通过限制请求来降低成本，并提高我们抵御 DoS 攻击的安全性。

**如何实施？**
我们正在使用`flask_limiter`库在我们的 flask 应用中提供速率限制特性。在完成文件后，你将需要一些进口货。确保您的环境已经安装了`flask`和`flask-limiter`。

```
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
```

在这些导入之后，您将需要添加一个默认限制器。使用下面的代码片段。

```
limiter = Limiter(
    application,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

对于`Limiter`，第一个参数应该是主 flask 文件应用程序的名称。那么`key_func` 应该等于`get_remote_address`。您可以将`default_limits`设置为您想要的任何值。

在每条路线下，您可以单独定制请求限制，它将绕过默认限制。

```
# example 5 per minute route 
@application.route("/slow")
@limiter.limit("5 per minute")
def slow():
    return "5 per minute!"
```

我们还可以将路由从默认限制中免除，用户将可以无限制地访问该路由。

```
# example no limit route
@application.route("/exempt")
@limiter.exempt
def exempt():
    return "No limits!"
```

**示例烧瓶 App:** *将下面的代码块复制粘贴到****application . py****文件中并运行*

```
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_addressapplication = Flask(__name__)limiter = Limiter(
    application,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)@application.route("/")
def default():
    return "default limit!"@application.route("/five")
@limiter.limit("5 per minute")
def five():
    return "5 per minute!"

@application.route("/exempt")
@limiter.exempt
def exempt():
    return "No limits!"

if __name__ == "__main__":
    application.run(port=5000, debug=True)
```

测试每条路线，并设置任何你想要的限制。一条路线也可能有多个限制。每当请求超过速率限制时，将不会调用该函数，而是会引发 429 http 错误。

**速率限制字符串符号**
举例:

*   每分钟 10 次
*   每小时 10 英镑
*   10/小时
*   10/小时；100/天；每年 2000 英镑
*   100/天，500/7 天
*   100/天，500/7 天

限制请求将由您将运行的应用程序决定。所以你会知道每条路线的最佳价格。:)

这篇文章最终将成为我和我的团队在从事一个名为 Cryptolytic 的项目时所学到的经验的系列文章的一部分(文章将很快写出来并链接到这里)。这是我们在进行项目时希望拥有的指南。

这篇文章最终将成为我和我的团队在从事一个名为 Cryptolytic 的项目时所学到的经验的系列文章的一部分(文章将很快写出来并链接到这里)。这是我们在进行项目时希望拥有的指南。

![](img/6213600eacd8f04bb18d2fba1e0e8599.png)

**包含本文中使用的所有代码的笔记本可以在这里找到**[](https://github.com/Cryptolytic-app/cryptolytic/tree/master/medium_articles)****，这是我们的密码破解项目的 repo 内部——所以如果你很好奇，可以去看看！****

****在 Twitter [@malexmad](https://twitter.com/malexmad) ， [Github](https://github.com/malexmad) 上找我，或者在 [LinkedIn](https://www.linkedin.com/in/marvin-davila/) 上联系我！****

****[](https://flask-limiter.readthedocs.io/en/stable/) [## 烧瓶限制器-烧瓶限制器 1.0.1+0.gb390e64 .脏文档

### 上述 Flask 应用程序将具有以下速率限制特征:通过远程地址的速率限制…

烧瓶限制器. readthedocs.io](https://flask-limiter.readthedocs.io/en/stable/)  [## 快速启动烧瓶文档(1.1.x)

### 一个最小的 Flask 应用程序看起来像这样:那么代码做了什么？首先，我们导入了这个类。安…

flask.palletsprojects.com](http://flask.palletsprojects.com/en/1.1.x/quickstart/#) [](https://github.com/Cryptolytic-app/cryptolytic) [## 密码破解-app/密码破解

### 你可以在 Cryptolytic 找到这个项目。Trello 板产品 Canvas Cryptolytic 是一个为初学者修修补补的平台…

github.com](https://github.com/Cryptolytic-app/cryptolytic)****