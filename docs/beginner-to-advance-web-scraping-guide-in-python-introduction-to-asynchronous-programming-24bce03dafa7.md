# 初级到高级 Python 中的 Web 抓取指南:异步编程简介

> 原文：<https://medium.com/analytics-vidhya/beginner-to-advance-web-scraping-guide-in-python-introduction-to-asynchronous-programming-24bce03dafa7?source=collection_archive---------12----------------------->

![](img/9bc557c4f5ea853f4d4328eaa9af418d.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Hitesh Choudhary](https://unsplash.com/@hiteshchoudhary?utm_source=medium&utm_medium=referral) 拍摄的照片

本教程将概述异步编程，包括其概念元素、Python 异步 API 的基础知识，以及异步 web scraper 的示例实现。同步程序很简单:启动一个任务，等待它完成，然后重复直到所有的任务都被执行。然而，等待会浪费宝贵的 CPU 周期。异步程序通过使用上下文切换来交错任务，以改善程序响应，并最小化处理器空闲时间和总程序执行时间，从而解决了这个问题。
在同步编程中，如果我们将一个请求的超时时间设置为 10 秒，那么我们的系统在理想状态下会等待 10 秒，这段时间可以用来处理之前的请求和发送新的请求。

# Python 中的异步编程

在[之前的教程](/@kaus.pathak_30409/beginner-to-advance-web-scraping-guide-in-python-ba24dca5dce0)中，我们实现了 Requests 异步编程，允许您编写在单线程中运行的并发代码。与多线程相比，第一个优势是您可以决定调度程序从一个任务切换到另一个任务的位置，这意味着在任务之间共享数据更加安全和容易。实现 Asyncio Web Scrapping 的主要库是`aiohttp`。`aiohttp`的示例代码附后:

# aiohttp 库的复杂性

*   虽然创建 aiohttp 库是为了处理多个请求，但它不是为了从单独的代理和头发送每个请求。
*   aiohttp 的概念在 python 中相对较新，python 对 Asyncio 编程的支持有限。
*   针对 python 中复杂并发函数的文章相对较少，但针对串行编程的文章较多。

因此，我们将实现我们的 Asyncio 编程，不使用 aiohttp，使用本教程中介绍的进程。但这是我们将要遵循的示例代码。

[ThreadPoolExecutor](https://docs.python.org/3.4/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor) 是一个 [Executor](https://docs.python.org/3.4/library/concurrent.futures.html#concurrent.futures.Executor) 子类，它使用线程池来异步执行调用。它使用最多 *max_workers* 线程池来执行调用。ThreadPoolExecutor 提供了一个简单的抽象，围绕着旋转多个线程并使用这些线程以并发方式执行任务。在适当的环境下，为应用程序添加线程有助于显著提高应用程序的速度。通过使用多线程，我们可以加速面临输入/输出瓶颈的应用程序。

# 更深入

在之前的教程中，我们创建了一个类来删除其他站点的用户代理和代理。但是我们搜集的大部分代理都不工作，或者服务器上的端口被屏蔽了。因此，为了从我们收集的代理中过滤工作代理，我们必须通过继承代理类并扩展该类来发送请求和分离工作代理。

这里我们使用 max_workers 为 50 来限制线程数量为 50，但是您也可以为具有更高处理能力和更高带宽的系统增加这个值。这段代码将被追加到 bin/proxy . py 文件中。

> **你已经在 Python 中成功实现了异步函数。做得好…！！！**

# 从这里继续前进。

## 教程 1 —简介

在[之前的教程](/@kaus.pathak_30409/beginner-to-advance-web-scraping-guide-in-python-799ffd367067)中，我们了解了网页抓取的基本概念，并创建了简单的函数来使用请求和 BeautifulSoup 从页面提取歌词。

## 教程 2 —使用 Python 中的头池和代理循环请求。

为了创建一个更大的项目，可以从互联网上删除成千上万的页面，你需要一个更清晰的工作环境，使用面向对象和继承的概念。你还需要有更详细的关于头文件池和代理池的知识来保持对服务器的匿名，我们已经在本教程的第二部分中讨论过了。

## 教程 3 —工作环境和异步 I/O 编程

我们将进行异步 I/O 编程来提高你的报废速度，这将在本教程的第三部分[中介绍。](/@kaus.pathak_30409/beginner-to-advance-web-scraping-guide-in-python-introduction-to-asynchronous-programming-24bce03dafa7?postPublishedType=initial)

## 教程 4 —自动化站点抓取

有了之前教程中学习的所有概念，我们将在本教程的第四部分中创建实际的自动抓取器来从网页下载并保存歌词。

## 教程 5 — API 访问

为了方便地从网上访问歌词，我们将创建 Flask API 和前端来访问我们在本教程第五部分[中废弃的歌词。](/analytics-vidhya/beginner-to-advance-web-scraping-guide-in-python-build-and-deploy-a-python-web-app-using-flask-202ffdf8fd40)

## 教程 6 —在 Heroku 上托管我们的 Flask 服务器

为了提供容易的歌词访问，我们将在本教程的第六部分[的 Heroku 上托管我们的 Flask 服务器。](/@kaus.pathak_30409/beginner-to-advance-web-scraping-guide-in-python-deploy-a-python-web-app-using-flask-and-a0e3cc8ce9f6)

![](img/cda93aa49539c4fe06f0885477abdc08.png)

# 最后的话

感谢你阅读这篇文章，我们希望听到你的反馈。请随意评论任何问题。如果你喜欢，请为我们鼓掌:)。关注我们，获取我们的最新文章。