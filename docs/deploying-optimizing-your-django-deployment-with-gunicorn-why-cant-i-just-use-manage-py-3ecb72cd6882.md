# 使用 gunicorn 部署和优化您的 django 部署(为什么我不能只使用 manage.py runserver？)

> 原文：<https://medium.com/analytics-vidhya/deploying-optimizing-your-django-deployment-with-gunicorn-why-cant-i-just-use-manage-py-3ecb72cd6882?source=collection_archive---------6----------------------->

你的应用越来越受欢迎了。你扩展得很快。现在是时候开始考虑如何优化您的部署了。

![](img/3349dd56da9a2ce6beaa7391ce066add.png)

在本文中，我们将讨论:

1.  什么是 gunicorn(为什么我不能直接用 manage.py runserver？)
2.  您的 gunicorn 部署的最佳设置(我需要多少工作人员)
3.  如果我已经在使用 gunicorn，为什么还需要 Nginx 或 Apache？
4.  每个工作线程多个线程
5.  绿色线程

## 什么是 Gunicorn

> Gunicorn 'Green Unicorn '是一个用于 UNIX 的 Python WSGI HTTP 服务器。这是一个前叉工人模型。Gunicorn 服务器广泛兼容各种 web 框架，实现简单，占用服务器资源少，速度相当快。

Gunicorn 是一个 HTTP 服务器，它加载您的 WSGI 应用程序([WSGI 是什么？](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface))，‘分叉’它的一个拷贝到多个‘工作’进程，并通过它提供流量。因此，如果你在 gunicorn 中创建四个“工人”,你的应用程序可以同时处理四个 web 请求！

Gunicorn 依靠操作系统的内核来为它进行负载平衡，并在它的工作进程之间分配工作，所以你不必这样做

这种方法有几个优点:即使你的一个工作线程被终止了([可能是因为一个分段错误](https://discuss.codechef.com/t/why-do-i-get-a-sigsegv/1773))，你的 web 服务器也不会宕机。流量被无缝地重定向到其他健康的工作线程。

此外，如果您的一个或多个工作人员陷入了类似无限循环的情况，其他健康的工作人员仍然可以为请求提供服务。同步工作者之间没有内存共享([什么意思？点击这里阅读更多关于同步和异步工人](https://docs.gunicorn.org/en/stable/design.html)的信息，这样无论一个工人进程遇到什么样的麻烦，它都不会给你的其他工人带来麻烦

Gunicorn 还将恢复任何被操作系统杀死的工人，它还可以定期杀死和替换工人(例如，如果您的应用程序有内存泄漏，这将有助于遏制它)。了解更多:[https://docs . guni corn . org/en/18.0/settings . html # max-requests](https://docs.gunicorn.org/en/18.0/settings.html#max-requests)

最后也是最重要的:它加快了你的应用程序！使用 manage.py，您一次只能接受一个请求。当您的 Django 应用程序等待外部 API 或数据库调用返回一些信息时，您的计算机处于空闲状态。更不用说 manage.py 只使用一个核心，因此如果您的计算机有多个可用的核心，其余的核心就不会被使用。Gunicorn 工作人员在多个内核之间分配工作。

## Gunicorn 部署的最佳设置

好吧。我应该去管理 Gunicorn。伟大的阿尤斯。但是我应该启动多少工人呢？

我应该只用一个工人吗？四个？八个？八千？工人越多就越好吗？

```
The recommended formula for the number of Synchronous workers is (2 * number_of_cpus_you_want_to_use) + 1
```

这是你应该经常使用的吗？不。不过这是个好的起点。更多的工人意味着更多的进程(这意味着更多的内存使用，因为每个工人都在单独加载你的整个应用程序)

如果您的应用程序需要等待很长时间(如大型数据库查询、API 调用等)，更多的并发将会有所帮助。这被称为 I/O 绑定应用程序。我们将在本文后面讨论另一种方法来代替使用 workers:线程化&绿色线程

如果您的应用程序进行大量处理(假设您有一个微服务，其中您接受一些输入，并使用重型机器学习模型对该输入进行预测)，您可能只需要让它保持在 2*cpu + 1。您甚至可以通过稍微减少这个数字来减少 RAM 的使用。这是因为如果你的 CPU 资源已经完全饱和，那么这几乎是你能从 gunicorn 中得到的所有优化。一旦您最大化了 CPU，增加工作人员或使用 AsyncIO 不会带来显著的不同。

## 如果我已经在使用 gunicorn，为什么还需要 Nginx 或 Apache？

你会在互联网上看到同样的事情:你不应该使用 gunicorn 作为前端应用程序。您应该始终使用 Nginx 或 Apache 来反向代理 Gunicorn。Gunicorn 的官方网站也是这么说的:[https://docs.gunicorn.org/en/18.0/deploy.html](https://docs.gunicorn.org/en/18.0/deploy.html)

> 我们强烈建议在代理服务器后面使用 Gunicorn。

但是为什么呢？Gunicorn 已经在处理所有的负载了。通过第三方转发请求有什么意义？

有几个原因:

1.  丢弃的请求:Gunicorn 没有一个内置的队列，当请求进来时，它将请求排队，并在请求可用时将它们分发给其中一个工作线程。Gunicorn 完全依靠操作系统内核来完成这项工作，内核会将一个作业分配给任何一个可用的工作者。
    但是如果找不到工人怎么办？假设你有 9 个工人，他们都很忙，又有一个请求进来？第 10 个请求会因 IO 错误而出错
2.  缓冲:Gunicorn 真的很容易 DoS ( [什么是 DoS 攻击？)](https://en.wikipedia.org/wiki/Denial-of-service_attack#:~:text=In%20computing%2C%20a%20denial%2Dof,host%20connected%20to%20the%20Internet.)，甚至无心。Gunicorn 将立即读取发送给它的数据，一旦它完成读取数据，它将开始处理它。这意味着在接收数据时，gunicorn 的 worker 是空着的:它除了等待数据流结束之外，没有做任何有价值的工作。恶意攻击者也可以利用这一点，根本不会完成数据流。它会启动一个连接，发送数据，但永远不会结束。如果您有 9 个工作人员，那么只需要 9 个这样的请求就可以让您的服务器脱机。
    即使没有恶意，gunicorn 的工作人员仍然在浪费时间等待数据的到来。
3.  静态文件:Gunicorn 不提供静态文件。它评估代码，仅此而已。请求一个静态文件，它会出错。
4.  安全性:如何保护 gunicorn 应用程序？Gunicorn 的建造从来没有把安全性放在首位。像 Nginx 和 Apache 这样的大型 web 服务器有很多集成和插件来帮助抵御常见的威胁(参见[https://www . Nginx . com/blog/remeding-DDOS-attacks-with-Nginx-and-Nginx-plus/](https://www.nginx.com/blog/mitigating-ddos-attacks-with-nginx-and-nginx-plus/))
5.  预处理:如果您想在数据到达应用服务器之前对其进行修改，该怎么办？一个很好的例子是，当您使用 cloudflare 时，cloudflare 的 IP 将显示为客户端 IP，而不是实际用户的 IP。您可以使用 Nginx 来编辑请求并设置正确的 X-FORWARDED-FOR 头，这样您的应用程序就可以知道用户的真实 IP，而无需特殊代码来处理 CloudFlare 的头([https://support . cloud flare . com/HC/en-us/articles/20017 07 86-Restoring-original-visitor-IPs-Logging-visitor-IP-addresses-with-mod-cloud flare-](https://support.cloudflare.com/hc/en-us/articles/200170786-Restoring-original-visitor-IPs-Logging-visitor-IP-addresses-with-mod-cloudflare-))

为了解决所有这些问题，我们使用类似 Nginx 或 Apache 的反向代理。它将提供静态文件，如果没有工作人员响应请求，将请求排队，动态编辑请求并缓冲它们，这样您的 gunicorns 就不会浪费时间。Nginx 和 Apache 还可以更好地与各种安全系统集成，以过滤请求和/或记录请求，发出警报等。

## 线

从 Gunicorn 19 开始，您可以为每个工作进程创建多个线程。这很好，因为您现在可以在不消耗资源的情况下同时处理更多的请求。

我应该有几根线？其实也差不多。

```
workers * threads_per_worker = (2*cpu)+1
```

那么使用线程到底有什么意义呢？只产生 9 个工作进程不是比 3 个工作进程各有 3 个线程更好吗？区别在于线程共享内存。3 个工作线程的 3 个线程组合将比 9 个工作线程的配置使用更少的内存。其次，多线程允许进程向主 gunicorn 进程发出信号，表明一个工人仍然活着，即使它正忙于一个高 CPU 任务(否则 gunicorn 会认为它处于无限循环中并杀死它)

## 绿色线程

好了，现在我们有工人了。每个工人占用相当大的内存，所以我们不能有太多的工人。如果我们的需求是 CPU 受限的，这是没问题的，因为我们基本上最大化了机器，以获得最大的收益。

但是如果我们是 I/O 绑定的呢？如果我们的机器只是浪费大量时间等待事情发生呢？我们不能只产生几十或几百个工人，因为他们会用掉太多的资源。如果每个工作人员对一个慢速 API(假设一个 API 需要一分钟以上的时间来响应)进行 API 调用，我们的服务器就会因为只有几十个并发用户而过载。

看起来单靠线程并不能帮助我们解决这个问题。但是 Gunicorn 还有另一个妙招:绿色线程(异步工作者)。

Python 的整个协程支持相当惊人。有效的协同例程允许我们在一个操作系统线程中运行多个独立的代码。令人惊奇的是，由于绿色线程只是一个虚拟的概念，我们可以在一个 worker 中创建数百个甚至数千个绿色线程！所以一个工人可以同时处理成百上千的请求。

那会有什么不同吗？对于 CPU 受限的应用程序来说，不会。但是如果您的工作人员花费大部分时间等待事情发生，协程允许工作人员简单地放弃控制，让其他线程在等待时做一些事情。

您可以创建 9 个 workers，并设置它们每个都使用 1000 个 eventlets。理论上允许 gunicorn 同时接受 9000 个请求。当然，如果您受到 CPU 的限制，这不会有什么不同，大多数请求会处理得很慢，以至于超时。但是，如果你是 I/O 绑定的，这可能会产生巨大的差异，因为当一个绿色线程等待一个缓慢的 API 时，其他绿色线程正在“共享”同一个工作线程来完成工作

## 结束了

因此，在本文中，我们涵盖了:为什么我们使用 gunicorn，为什么我们使用反向代理连接到 gunicorn，优化 gunicorn 的一些基本信息，线程和绿色线程如何工作，以及它们何时有用。

如果你想了解更多，这里有一些推荐读物:

关于协程和绿色线程:

1.  [维基百科关于协同程序的文章](https://en.wikipedia.org/wiki/Coroutine#:~:text=Coroutines%20are%20computer%20program%20components,iterators%2C%20infinite%20lists%20and%20pipes.)
2.  迷人的 Python:用 Python 生成器实现“无重量线程”，作者 David Mertz(2002 年 6 月 1 日)
3.  [使用 Eventlets 在 Python 中打猴子补丁](https://eventlet.net/doc/patching.html)

关于 gunicorn 工人:

1.  [Stackoverflow 关于管理工人、线程的回答&绿色线程](https://stackoverflow.com/a/41696500/1639052)
2.  [Stackoverflow 解释 Gunicorn 使用的 Fork 前工人模型](https://stackoverflow.com/questions/25834333/what-exactly-is-a-pre-fork-web-server-model)

与我联系:

[https://twitter.com/AgentAayush](https://twitter.com/AgentAayush)

[](https://www.linkedin.com/in/aayushagrawal101/) [## Aayush Agrawal - Python 开发者-proton shub Technologies | LinkedIn

### 查看 Aayush Agrawal 在全球最大的职业社区 LinkedIn 上的个人资料。Aayush 有 2 份工作列在…

www.linkedin.com](https://www.linkedin.com/in/aayushagrawal101/) 

请继续关注更多关于使用 Docker & Kubernetes 将 gunicorn 配置带到云中以及利用 Kubernetes 的水平 Pod 自动伸缩和集群自动伸缩的文章