# Python 中的异步、线程和多重处理

> 原文：<https://medium.com/analytics-vidhya/asyncio-threading-and-multiprocessing-in-python-4f5ff6ca75e8?source=collection_archive---------0----------------------->

![](img/8e6058acdbb081175293789ae82e2cb6.png)

图片取自另一篇关于 java 线程的中型文章

AsyncIO 是在 python 中实现并发性的一个相对较新的框架。在本文中，我将把它与多线程和多重处理等传统方法进行比较。在开始举例之前，我将补充一些关于 python 中并发性的内容。

*   CPython 强制执行 GIL(全局解释器锁),这阻止了充分利用多线程。在运行任何字节码之前，每个线程都需要获得这个互斥锁
*   多线程通常是网络 I/O 或磁盘 I/O 的首选，因为线程之间不需要激烈竞争来获取 GIL。
*   多处理通常是 CPU 密集型任务的首选。多重处理不需要 GIL，因为每个进程都有自己的状态，然而，创建和销毁进程并不简单。
*   带线程模块的多线程是抢占式的，它需要自愿和非自愿的线程交换。
*   AsyncIO 是一个单线程单进程协作多任务处理。asyncio 任务独占 CPU，直到它希望将 CPU 让给协调器或事件循环。(稍后将涉及术语)

# 实践示例

## 延迟消息

程序延迟打印消息。当主线程休眠时，CPU 是空闲的，这是对资源的一种不良使用。

```
12:39:00:MainThread:Main started
12:39:00:MainThread:TWO received
12:39:02:MainThread:Printing TWO
12:39:02:MainThread:THREE received
12:39:05:MainThread:Printing THREE
12:39:05:MainThread:Main Ended
```

## 线程并发

使用 python 的线程模块在独立的非守护线程*上多次调用 *delay_message* 。*不出所料，程序执行速度比上面的同步版本快了两秒。当线程空闲(休眠)时，操作系统交换线程。您可以将睡眠与进行系统调用以与外部环境通信联系起来。

```
12:39:05:MainThread:Main started
12:39:05:Thread-4:TWO received
12:39:05:Thread-5:THREE received
12:39:07:Thread-4:Printing TWO
12:39:08:Thread-5:Printing THREE
12:39:08:MainThread:Main Ended
```

## 线程池

尽管线程是轻量级的，但是创建和销毁大量线程的代价是昂贵的。`concurrent.futures`建立在线程模块之上，提供了一个简洁的接口来创建线程池和进程池。它没有为函数调用创建新的线程，而是重用了池中现有的线程。

```
10:42:36:ThreadPoolExecutor-0_0:TWO received
10:42:36:ThreadPoolExecutor-0_1:THREE received
10:42:38:ThreadPoolExecutor-0_0:Printing TWO
10:42:38:MainThread:TWO Done
10:42:39:ThreadPoolExecutor-0_1:Printing THREE
10:42:39:MainThread:THREE Done
```

## 使用 AsyncIO 的并发性

1.  **协程:**与传统的单点退出函数不同，协程可以暂停和继续执行。创建协程就像在声明函数之前使用关键字`async`一样简单。
2.  **事件循环或协调器:**管理其他协程的协程。你可以把它想象成一个调度器或主控器。
3.  **可应用对象**协程、任务和未来都是可应用对象。协程可以等待合适的对象。当一个协程在等待一个可请求的对象时，它的执行被暂时挂起，并在 Future 完成后恢复。

```
07:35:32:MainThread:Main started
07:35:32:MainThread:Current registered tasks: 1
07:35:32:MainThread:Creating tasks
07:35:32:MainThread:Current registered tasks: 3
07:35:32:MainThread:TWO received
07:35:32:MainThread:THREE received
07:35:34:MainThread:Printing TWO
07:35:35:MainThread:Printing THREE
07:35:35:MainThread:Main Ended
```

尽管程序运行在单线程上，但它可以通过协作式多任务处理实现与多线程代码相同的性能水平。

## 创建异步任务的更好方法

使用 asyncio.gather 一次性创建多个任务。

```
08:09:20:MainThread:Main started
08:09:20:MainThread:ONE received
08:09:20:MainThread:TWO received
08:09:20:MainThread:THREE received
08:09:20:MainThread:FOUR received
08:09:20:MainThread:FIVE received
08:09:21:MainThread:Printing ONE
08:09:22:MainThread:Printing TWO
08:09:23:MainThread:Printing THREE
08:09:24:MainThread:Printing FOUR
08:09:25:MainThread:Printing FIVE
08:09:25:MainThread:Main Ended
```

## 关于在异步任务中阻止调用的警告

正如我前面所说的，asyncio 任务拥有使用 CPU 的独占权利，直到它自愿放弃。如果一个阻塞调用错误地溜进了你的任务，它将会延缓程序的进程。

```
11:07:31:MainThread:Main started
11:07:31:MainThread:Creating multiple tasks with asyncio.gather
11:07:31:MainThread:ONE received
11:07:31:MainThread:TWO received
11:07:31:MainThread:THREE received
11:07:34:MainThread:Printing THREE
11:07:34:MainThread:FOUR received
11:07:34:MainThread:FIVE received
11:07:34:MainThread:Printing ONE
11:07:34:MainThread:Printing TWO
11:07:38:MainThread:Printing FOUR
11:07:39:MainThread:Printing FIVE
11:07:39:MainThread:Main Ended
```

当 delay_message 接收到消息三时，它进行阻塞调用，直到完成任务才放弃对事件循环的控制，从而延缓了执行进度。因此，它比前一次运行多花了*三秒*。虽然这个例子看起来是特制的，但是如果你不小心的话，它也可能发生。另一方面，线程是抢占式的，如果操作系统在等待阻塞调用，它会抢先切换线程。

## 竞赛条件

如果不考虑竞争条件，多线程代码会很快崩溃。当使用外部库时，这变得尤其棘手，因为我们需要验证它们是否支持多线程代码。例如，流行请求模块的`session` 对象不是线程安全的。因此，试图使用一个`session`对象来并行化网络请求可能会产生意想不到的结果。

```
20:28:15:ThreadPoolExecutor-0_0:Update Started
20:28:15:ThreadPoolExecutor-0_0:Sleeping
20:28:15:ThreadPoolExecutor-0_1:Update Started
20:28:15:ThreadPoolExecutor-0_1:Sleeping
20:28:17:ThreadPoolExecutor-0_0:Reading Value From Db
20:28:17:ThreadPoolExecutor-0_1:Reading Value From Db
20:28:17:ThreadPoolExecutor-0_0:Updating Value
20:28:17:ThreadPoolExecutor-0_1:Updating Value
20:28:17:ThreadPoolExecutor-0_1:Update Finished
20:28:17:ThreadPoolExecutor-0_0:Update Finished
20:28:17:MainThread:Final value is 1
```

理想情况下，最终值应该是 2。然而，由于线程的抢先交换，`thread-0`在更新值之前被交换，因此`updates`错误地产生最终值为 1。我们必须使用锁来防止这种情况发生。

```
21:02:45:ThreadPoolExecutor-0_0:Update Started
21:02:45:ThreadPoolExecutor-0_0:Sleeping
21:02:45:ThreadPoolExecutor-0_1:Update Started
21:02:45:ThreadPoolExecutor-0_1:Sleeping
21:02:47:ThreadPoolExecutor-0_0:Reading Value From Db
21:02:47:ThreadPoolExecutor-0_0:Updating Value
21:02:47:ThreadPoolExecutor-0_0:Update Finished
21:02:47:ThreadPoolExecutor-0_1:Reading Value From Db
21:02:47:ThreadPoolExecutor-0_1:Updating Value
21:02:47:ThreadPoolExecutor-0_1:Update Finished
21:02:47:MainThread:Final value is 2
```

## AsyncIO 很少出现竞争情况

由于任务可以完全控制何时暂停执行，asyncio 很少出现竞争情况。

```
20:35:49:MainThread:Update Started
20:35:49:MainThread:Sleeping
20:35:49:MainThread:Update Started
20:35:49:MainThread:Sleeping
20:35:50:MainThread:Reading Value From Db
20:35:50:MainThread:Updating Value
20:35:50:MainThread:Update Finished
20:35:50:MainThread:Reading Value From Db
20:35:50:MainThread:Updating Value
20:35:50:MainThread:Update Finished
20:35:50:MainThread:Final value is 2
```

如您所见，一旦任务在`sleeping`之后被恢复，它就不会放弃控制权，直到完成协程的执行。有了线程，线程交换不是很明显，但是有了 asyncio，我们可以控制协程执行应该被暂停的确切时间。尽管如此，当两个协程进入死锁时，它可能会出错。

## 多重处理

如前所述，当实现 CPU 密集型程序时，多处理非常方便。下面的代码对带有`30000`元素的`1000`列表执行合并排序。如果下面的合并排序实现有点笨拙，请原谅。

## 同步版本

```
21:24:07:MainThread:Starting Sorting
21:26:10:MainThread:Sorting Completed
```

## 异步版本

```
21:29:33:MainThread:Starting Sorting
21:30:03:MainThread:Sorting Completed
```

默认情况下，进程的数量等于机器上处理器的数量。您可以观察到两个版本之间的执行时间有了相当大的改进。

我将写这篇文章的`PART 2`关于异步网络请求和文件读取。如果您发现代码中有任何错误，请随时发表评论或在 [LinkedIn](https://www.linkedin.com/in/ajitsamudrala/) 上联系我。