# Heroku 和部署

> 原文：<https://medium.com/analytics-vidhya/heroku-and-deployments-c3fa56d11904?source=collection_archive---------9----------------------->

![](img/f83c700dc18325a34a2ad383c7628635.png)

照片由[萨法·萨法罗夫](https://unsplash.com/@codestorm?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

每当我们谈论用任何语言或框架开发一些 web 应用程序或任何项目时，一个问题总会浮现在我们的脑海中，那就是**“在哪里部署这个应用程序”**，如何才能看到我的应用程序或项目的运行版本。 ***Heroku*** 是一个平台即服务(PaaS)解决方案(基于容器)，使其用户和其他开发人员能够完全在云中构建、运行和操作应用程序。最初，它只支持完全用 Ruby 编程语言开发的应用程序，但现在它也支持 Java、Python、Node.js、PHP 和 GO。

# **Heroku 上的基本 Python 应用部署**

Heroku 实际上非常简单、优雅、灵活且易于使用。实际上，Heroku 官方网站上有一个存储库设置，允许用户在 Heroku cloud 上测试基本的 python 应用程序部署。这里是提供的 URL

[https://github.com/heroku/python-getting-started](https://github.com/heroku/python-getting-started)

这是一个简单的 python 应用程序，包含一个指定 Python 版本为 3.7.3 的`runtime.txt`和一个用于安装依赖项的`requirements.txt`文件。

# **部署:**

首先，你必须熟悉赫罗库的环境。对于基本的应用程序部署，我们有 Heroku-CLI，通过它我们可以轻松地使用命令。

首先，我们将在 Heroku 上初始化一个应用程序，让 Heroku 准备好设置您的源代码:

```
heroku createCreating app... done, ⬢ secret-sea-89849
[https://secret-sea-89849.herokuapp.com/](https://secret-sea-89746.herokuapp.com/) | [https://git.heroku.com/secret-sea-89849.git](https://git.heroku.com/secret-sea-89746.git)
```

它将自动创建一个随机命名的应用程序，如**“secret-sea-89849”**，一个 git remote(称为`heroku`)也是用您的本地 git 库创建的。

现在我们将把我们的代码推送给 heroku git created

```
git push heroku masterCounting objects: 407, done.
Delta compression using up to 8 threads.
Compressing objects: 100% (182/182), done.
Writing objects: 100% (407/407), 68.65 KiB | 68.65 MiB/s, done.
Total 407 (delta 199), reused 407 (delta 199)
remote: Compressing source files... done.
remote: Building source:
remote:
remote: -----> Python app detected
remote:        Using supported version of Python 3.7 (python-3.7.3)
remote: -----> Installing python-3.7.3
.
.
.
```

它会将 requirements.txt 文件中的所有依赖项安装到 git 存储库中。完成推送过程后，您的应用程序现在部署到 Heroku cloud。之后，您可以扩展应用程序当前运行的实例数量。你可以通过简单的命令来完成它。

```
heroku ps:scale web=1
```

此命令不会部署应用程序。它会在您部署后启动它。当您部署应用程序时，heroku 会创建一个“slug”。应用程序的可运行压缩版本，然后存储。然后你可以启动“dynos ”,它会获取你当前的 slug 并在 heroku 的一个服务器上启动它。

运行`heroku ps:scale web=1`会将你的应用扩展到一个运行 dyno 的服务器，基本上意味着你现在有一个运行你的应用的服务器。

如果您再次部署您的应用程序，将会生成并存储一个新的 slug，并且您当前运行的 dynos 将会被销毁，并被新版本的代码所取代。

现在，您可以访问由其应用程序名称生成的 URL。使用 heroku-CLI，您可以使用一个简单的命令:

```
heroku open
```

它会显示您的应用程序当前运行的 URL，您可以在浏览器中访问该 URL，并且您会看到首页。会有些像这样；

`[https://{your-project-name}.herokuapp.com/](https://{your-project-name}.herokuapp.com/)`

# 日志记录:

用户可以使用 CLI 中的一个简单命令轻松检查正在运行的应用程序的日志。

```
heroku logs --tail
```

基本上，gunicorn 服务器运行在那里，所以它会显示应用程序的详细日志，这将有助于用户在调试和其他过程中。

# **简单地说:**

这是一个简单的教程，解释了如何在 Heroku cloud 上部署一个简单的 Python 应用程序。在下一篇博客中，我们可以使用我们在之前的博客中创建的示例 python-flask 应用程序，并将它部署在 Heroku cloud 上。关于 Heroku cloud 还有很多需要学习和实践的地方。

开发一个软件是不够的，还必须学习大规模部署和扩展应用程序的基础知识。