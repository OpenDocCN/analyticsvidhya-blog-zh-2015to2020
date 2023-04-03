# 如何在 Heroku 上免费部署您的 Streamlit 应用程序！

> 原文：<https://medium.com/analytics-vidhya/how-to-deploy-your-streamlit-app-on-heroku-for-free-284c96c2a06d?source=collection_archive---------1----------------------->

*本文假设你已经用 Streamlit 开发了一个 app。如果你还没有，在这里查看文档*[](https://docs.streamlit.io/en/stable/)**。这是一个奇妙而简单的工具！**

*![](img/ca7e8e7654c0e5439accc742170466a7.png)*

*艾蒂安·布朗热在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片*

*所以你用 Streamlit 开发了一个 web 应用程序，你想与全世界分享它，但是你不知道如何或者不想为托管服务付费。让我把你介绍给 Heroku！Heroku 有一个非常好的免费层服务，让你一次托管多达 5 个未经验证的应用程序(未经验证意味着你没有给他们信用卡)，每个应用程序都有一个自定义域。好吧，让我们开始吧！*

## *报名参加 Heroku*

*首先，你需要[创建一个 Heroku 账号](https://signup.heroku.com)。然后你要下载 [Heroku 命令行界面](https://devcenter.heroku.com/articles/heroku-cli)。注册并安装 CLI 后，您可以打开终端并运行:*

```
*heroku login*
```

*这将提示您在网络浏览器中登录您的新 Heroku 帐户。完成后，您的终端应显示“以<email>身份登录”。</email>*

## *启动您的项目回购*

*接下来，您将使用 Git 启动项目回购。只需导航到您想要的目录并运行下面的代码。请注意，这个项目名称将是您的域名，形式为 your-app-name.heroku.com。所以你给它起的名字越通用，它被使用的几率就越高。*

```
*heroku create your-app-name --buildpack heroku/python*
```

*实际上你不需要指定 python buildpack，但是有时 Heroku 不能识别 python 文件，所以最好确认一下。您的终端应该告诉您应用程序已经创建，并提供远程回购 url。复制 repo url 并运行下面代码，同样是在您的项目所在的目录中。*

```
*git clone remote-repo-url*
```

*这将创建一个 repo 的本地副本，您可以在其中工作，然后将更改推回到远程 repo。(它会说你克隆了一个空的资源库，没问题)。*

## *项目目录中需要的文件*

*现在您的应用程序已经创建，并且您已经有了 repo 的本地副本，您可以将 Streamlit 应用程序的 py 脚本转移到项目目录中。*

*但是，在部署之前，您需要一些特殊的文件。*

## *需求文件*

*首先你需要一个需求文件，告诉 Heroku 你的应用需要哪些 Python 库，以及哪些版本。生成这个文件最简单的方法是使用 pipreqs 库。你可以用 [pip](https://pypi.org/project/pipreqs/) 或者 [anaconda](https://anaconda.org/conda-forge/pipreqs) 安装 pipreqs。然后在您的项目目录中运行下面的。(或者，您可以在 pipreqs 之后添加目录路径，以便从其他地方运行它)。*

```
*pipreqs*
```

*就是这样。非常容易。该库读取您的 py 脚本，并自动确定您的应用程序将需要哪些库，然后将 requirements.txt 文件保存在指定的目录中。*

## *设置 Bash 脚本*

*接下来，您将需要一个 bash 脚本来配置 Streamlit，以便在 Heroku 上使用。该文件应该被命名为“setup.sh ”,并且应该具有下面的确切代码。您可以在任何文本编辑器中编辑该文件。同样，这需要在项目的主目录中。*

```
*mkdir -p ~/.streamlitecho "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml*
```

## *Procfile*

*最后，您需要一个称为 Procfile 的奇怪文件，它指定了应用程序在启动时执行的命令。它应该拼写为“Procfile”，没有任何文件扩展名，并且应该位于项目的主目录中。你可以用任何文本编辑器编辑它。用您的 Streamlit python 脚本替换下面代码的最后一部分。*

```
*web: sh setup.sh && streamlit run your-script.py*
```

## *推送远程回购和部署！*

*现在，所有必需的文件都应该在您的本地 repo 中:*

*   *您的 python 脚本(用于 Streamlit)*
*   *Procfile*
*   *setup.sh*
*   *requirements.txt*

*读入 python 脚本的任何文件也需要在该目录中。你现在要做的就是把文件推送到远程仓库！*

```
*git add .
git commit -m "message"
git push origin master*
```

*似乎默认的远程名称是 origin，默认的分支名称是 master。您可以通过在项目存储库中显示隐藏文件，打开。git，然后是配置文件。它应该告诉您远程和分支名称。*

*这将加载一切到远程回购和你的应用程序将很快上线！去 https://your-app-name.herokuapp.com 看现场直播。*