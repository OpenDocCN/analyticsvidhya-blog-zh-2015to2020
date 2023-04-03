# 如何在 Google App Engine Flex 上运行 Pyppeteer

> 原文：<https://medium.com/analytics-vidhya/how-to-run-pyppeteer-on-google-app-engine-flex-or-how-i-lost-it-trying-701b09fadf99?source=collection_archive---------5----------------------->

![](img/4b1889e817fc07d5dd0729a212912f96.png)

演职员表:[https://hacking ands lacking . com/deploy-app-containers-with-GCP-app-engine-1681 BD 120357](https://hackingandslacking.com/deploy-app-containers-with-gcp-app-engine-1681bd120357)

**TL；DR:** 我浪费了一天时间研究如何让 Pyppeteer 在 GCP 的 App Engine Flex 环境下运行(特别是)。在网上找不到解决方案，只是运用常识，我正在写一篇我希望已经找到的解决方案的文章。跳到下面的解决方案

# 一些背景

[木偶师](https://github.com/puppeteer/puppeteer)是一个为 NodeJS 构建的无头 Chromium API。 [Pyppeteer](https://github.com/miyakogi/pyppeteer) 是一个*或多或少*奇偶 Python 3 端口的木偶师。

它的主要用途是前端单元测试，但也可以作为一个自动化的虚拟浏览器，从中提取有价值的数据。

[我是 Adtech 的一名数据工程师](http://www.paulpierre.com)(意思是:喜欢大数据的后端工程师),我目前从事的项目的一部分涉及到从专有平台提取数据，通常没有 API ***或类似的东西。***

大多数时候，对前端进行逆向工程、提取 devtools 中的 XHR 并将它们放入 Python 的请求库中是相当容易的。对于通过请求和 cookies 的简单身份验证，获取所需的数据是轻而易举的。

我很少会遇到障碍，但如果我遇到了，通常要么是一个非常安全的前端，带有烦人的计时令牌，要么是一个设计复杂的遗留系统，不值得为其构建模块。

在所有情况下，Pyppeteer 都是非常可靠的*，直到您决定将其集成到 App Engine Flex* 代码库并进行部署。

***标准的 GCR Python docker 镜像缺少 Chromium 运行*** 的合适库和驱动。当您部署它时，构建会很好地完成**但是**当您实例化“launch”方法时，您会得到:

> **加载共享库时出错:libX11.so.6:无法打开共享对象文件:没有这样的文件或目录**

# 解决方案

不幸的是，Pyppeteer 的任何文档中都没有这个解决方案，而且一开始也不明显，但是以下是必需的:

不要在你的应用程序的 YAML 文件中使用默认的“python”值作为**运行时**，而是将其设置为“**自定义**”。这让 App Engine 知道您将使用 Dockerfile 文件。

1.  在应用程序的主目录中创建一个 docker 文件
2.  使用以下内容作为模板:

```
FROM gcr.io/google_appengine/pythonRUN apt-get update && apt-get -y install  gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils wget

RUN virtualenv /env
ENV *VIRTUAL_ENV* /env
ENV *PATH* /env/bin:$*PATH* RUN mkdir /your/app/path
ADD . /your/app/path
WORKDIR /your/app/path
ADD requirements.txt /your/app/path/requirements.txt
RUN pip3 install -r /your/app/path/requirements.txt
EXPOSE 8080
ENTRYPOINT ["gunicorn", "-b", ":8080", "main:app"]
```

因此，让我们一部分一部分地看一下。

```
FROM gcr.io/google_appengine/python
```

首先，我们想使用 Google Cloud Repository 的容器映像。这相当于将您的运行时设置为“python”..所以什么都不会改变，但是我们可以向我们的环境中添加包。

```
RUN apt-get update && apt-get -y install ...etc
```

接下来我们要运行 apt-get 并安装所有的 Chromium 依赖项。

后续步骤是可选的，但为您的应用程序创建一个虚拟环境是一种很好的做法，这样就不会与容器映像发生冲突。

另外你喜欢整洁，对吗？

然后在 requirements.txt 中安装软件包，公开端口 8080 (App Engine Flex 的默认端口)，并设置入口点——在我的例子中是 gunicorn。

3.在使用 Pyppeteer 的 Python 文件中，确保将标记“-no-sandbox”添加到参数中，如下所示:

```
args = ['--no-sandbox']
browser = await launch(args=args)
page = await browser.newPage()
```

这个标志很重要，因为默认情况下你的容器将以 root 用户身份运行，而 Chromium 不允许你以 root 用户身份在沙盒模式下运行。

# 结论

我希望你会发现这很有用，它可以节省一些开发人员的时间，这些时间本可以花在[/r/PraiseTheCameraMan](https://www.reddit.com/r/PraiseTheCameraMan)*(稍后感谢我)*上。如果是的话，请在下面留下您的评论和反馈。

如果你好奇，我创建了“[告密者——电报大规模监视](https://github.com/paulpierre/informer)”这是一段代码，演示了如何大规模部署无限数量的电报机器人，这些机器人与“真实”账户没有区别，它向你展示了如何以自动化的方式完成这一任务。

原谅该项目的标题，FWIW 它是[特色的黑客新闻](https://news.ycombinator.com/item?id=21750353)，随时[发送一些明星](https://github.com/paulpierre/informer)研究员开发。

祝你愉快。继续。