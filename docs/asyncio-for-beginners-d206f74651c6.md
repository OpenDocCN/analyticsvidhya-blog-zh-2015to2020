# 使用 AsyncIO 在 python 中进行异步编程(适用于初学者)

> 原文：<https://medium.com/analytics-vidhya/asyncio-for-beginners-d206f74651c6?source=collection_archive---------15----------------------->

![](img/5c63acf00277c662fb96b2ff7fafea0e.png)

在本文中，我将通过 AsyncIO 库解释 Python 中的异步编程。

在继续之前，让我们了解一些术语

> ***事件循环*** *:* *用于并发运行异步任务(可以是协程、未来或任何可唤醒的对象)，可以注册将要执行的任务，执行它们，延迟或取消它们*
> 
> *假设我们有 2 个异步任务(task1 和 task2 ),现在我在事件循环上调度这两个任务。现在假设事件循环开始执行 task1 并遇到 IO 操作，然后它将从 task1 收回控制权，并将其交给 task2 执行。当 task1 完成 IO 时，当控制权返回 task 1 时，它将从停止状态恢复，因此两个或更多任务可以同时运行*
> 
> **:如果一个对象可以用在* `*await*` *或* `*yield from*` *表达式中，我们说它是一个可唤醒的对象。**
> 
> *python 中有三种主要类型的可用对象:协程、任务和未来。*

# ***异步输入输出***

*AsyncIO 是一个使用单线程或事件循环帮助并发运行代码的库，它基本上使用 async/await API 进行异步编程。*

*在 python 3.3 中发布 AsyncIO 之前，我们使用线程、greenlet 和多处理库来实现 python 中的异步编程*

## *异步编程为什么需要 AsyncIO？*

1.  *因为线程可用于同时运行多个任务，但 python 线程由操作系统管理，与绿色线程相比，操作系统必须进行更多的上下文切换 b/w 线程*
2.  *Greenlet(绿色线程)从操作系统中取走调度程序，并自己运行调度程序，但是 CPython 默认情况下不使用绿色线程(这就是 asyncio、gevent、PyPy 等的用途)*
3.  *通过使用多处理库，我们可以创建多个进程，我们可以让程序充分利用计算机的所有内核，但是产生进程的成本很高，因此对于 I/O 操作，线程在很大程度上被选择*

*总的来说，asyncIO 是一种可读性更强、更简洁的异步编程方法。*

## *如何使用 asyncIO？*

*当 asyncIO 发布时，它正在使用带有基于生成器的协程的`[@asyncio.coroutine](https://docs.python.org/3/library/asyncio-task.html#asyncio.coroutine)`装饰器来实现异步编程*

*Asyncio 生成器协同程序使用`yield from`语法来挂起协同程序。*

*在下面的例子中,`some_async_task()`是一个基于生成器的协程，要执行这个协程，首先我们需要获得事件循环(在第 11 行),然后使用(`loop.run_until_complete)`)调度这个任务在一个事件循环中运行(注意:直接调用`some_async_task()`不会调度这个任务执行，它只会返回生成器对象)*

> **此处第 8 行* `*asyncio.sleep()*` *是一个协程(我们可以在此使用任何协程或任务/未来)当语句* `*yield from*` *执行它时，它将放弃控制权返回事件循环让其他协程执行，当协程* `*asyncio.sleep()*` *完成且事件循环将控制权返回* `*some_async_task()*` *协程时，它将运行进一步的指令(如第 9 行)**

*在 Python 3.5 中，该语言引入了对协程的本地支持。现在我们可以使用 async/await 语法来定义本机协程*

> *以`async def`为前缀的方法自动成为本机协程。*
> 
> *`await`可用于获得可应用对象的结果(可以是协程、任务或未来)*

## *`How to run the event loop?`*

*在 python 3.7 之前，我们手动创建/获取事件循环，然后调度我们的任务，如下所示:*

```
***loop = asyncio.get_event_loop()** #if there is no event loop then it will create new one. **loop.run_until_complete(coroutine())** #run until coroutine is completed.*
```

*在 python 3.7 及更高版本中，以下是运行事件循环的首选方式*

```
***asyncio.run(coroutine())**# This function runs the passed coroutine, taking care of managing the asyncio event loop and *finalizing asynchronous generators*.*
```

*AsyncIO 提供高级 API 和低级 API，一般来说，应用程序开发人员使用高级 API，库或框架开发人员使用低级 API*

## *`**Futures**` **在 Asyncio***

*它是一个低级的可感知的对象，应该在未来有一个结果。*

*当一个未来对象被等待时，这意味着协程将一直等待，直到该未来对象在某个其它地方被解析。*

*这个 API 允许基于回调的代码与 async/await 一起使用*

*通常在应用程序级代码中，我们不处理未来对象，它通常由 asyncio API 或库公开。*

## *AsyncIO 中的任务*

*Task 是 futures 的子类，用于在一个事件循环中同时运行协同程序。*

*创建任务有多种方式:*

*   *`loop.create_task()` →通过低级 API，它只接受协程。*
*   *`asyncio.ensure_future()` →通过低级 API，它可以接受任何可用的对象，这将在所有 python 版本上工作，但可读性较差。*
*   *`asyncio.create_task()` →通过高级 API，它在 Python 3.7+中工作，接受协程并将它们包装成任务*

## *asyncio.create_task()*

*当一个协程被打包到一个具有类似于`asyncio.create_task()`的功能的任务中时，这个协程会被自动调度为很快运行*

*在下面的例子中，我使用`aiohttp`库从黑客新闻公共 API 获取新闻文章，我创建了两个任务(任务 1 和任务 2)来同时获取两条不同的新闻，并显示两条新闻文章的标题。*

***asyncio . assure _ future**()类似于`asyncio.create_task()`，但它也可以接受未来，如下例所示*

## *asyncio . gather(* a waitiable _ objects，return_exceptions)*

*它负责收集所有的结果，它将等待所有的 awaitables 对象完成，并按照给定的 await ables 对象的顺序返回结果*

*如果任何一个可用对象出现异常，它不会取消其他可用对象*

*在下面的例子中，我们同时运行两个任务，我们可以看到，如果`some_async_task2`出现异常，它不会取消`some_async_task()`协程*

*如果 return_exceptions 为假，并且在任何一个合适的对象中出现任何异常，那么`await asyncio.gather()`立即返回并在屏幕上显示一个错误。因此，出于演示目的，在第 17 行，我们正在等待另一个协程(它将在 6 秒后解析),以确保程序不会在 4 秒后退出*

*我们可以在输出中看到第 6 行的执行*

*如果我们想在一个数组中收集所有的结果(连同异常),那么我们可以使用 return_exceptions=True，它将异常作为一个结果，并且它将被聚集在结果列表中。*

*现在就这样，将来，我会写关于我们如何利用 Django 的 AsyncIO 库和 ASGI*