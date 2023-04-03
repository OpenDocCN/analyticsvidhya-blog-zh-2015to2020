# Kotlin Coroutines，RxJava 的衰落

> 原文：<https://medium.com/analytics-vidhya/kotlin-coroutines-the-fall-of-rxjava-1888c34803e2?source=collection_archive---------18----------------------->

RxJava 长期以来一直是救命恩人。凭借其提供的所有功能，Android 编程思维模式已经大大转变为一种更具反应性的方式。

现在我们有了协程(Coroutines)，它在很多会谈和会议中得到了赞扬和推荐，所以我开始学习它。

协程是编写可读性和可维护性非常好的异步代码的好方法。

在这篇文章中，我将尝试用简单的语言解释协程的优势，以及它是如何超越竞争对手的。为了长话短说，我不会深入研究基于协程的高级结构。重点是给出一个概述，并分享我对协程的心理模型。

# 什么是 Kotlin 协同程序？

![](img/8238bdc84dcaf840c5e2875aa3d4c9b9.png)

> C 例程是计算机程序的组成部分，它通过允许暂停和恢复执行来概括非抢占式多任务的子程序

Kotlin 团队将协程定义为“轻量级线程”。它们是*实际*线程可以执行的任务。

它们是一种并发设计模式，您可以在 Android 上使用它来简化异步执行的代码。[协同程序](https://kotlinlang.org/docs/reference/coroutines/coroutines-guide.html)在版本 1.3 中被添加到 Kotlin 中，并且基于其他语言中已建立的概念。

# 它解决什么问题？

在 Android 上，协程有助于解决两个主要问题:

1.  管理长时间运行的任务，否则可能会阻塞主线程并导致应用程序冻结。

在 Android 上，每个应用都有一个处理用户界面和管理用户交互的主线程。如果你的应用程序给主线程分配了太多的工作，那么这个应用程序看起来可能会冻结或者变慢。

Kotlin 使用一个*堆栈框架*来管理哪个函数与任何局部变量一起运行。当挂起协程时，当前堆栈帧被复制并保存以备后用。当恢复时，堆栈帧从它被保存的地方被复制回来，函数再次开始运行。即使代码看起来像普通的顺序阻塞请求，协程也能确保网络请求避免阻塞主线程。

2.提供*主安全*，或者从主线程安全调用网络或磁盘操作。

Kotlin 协程使用*调度程序*来决定哪些线程用于协程执行。要在主线程之外运行代码，可以告诉 Kotlin 协同程序在*默认*或 *IO* 调度程序上执行工作。在 Kotlin 中，所有协程都必须运行在调度程序中，即使它们运行在主线程上。协程可以挂起自己，调度程序负责恢复它们。

# 【RxJava 呢？

在 kotlin 的协同程序出现之前，异步任务的现有解决方案是 Rxjava。

RxJava 是一个库，目标是在 Java 编程语言中使用反应式编程。Rx 代表 Reactive Extensions(react vex)，这是一个为不同的编程语言提供反应式编程实现的项目。

RxJava 是一个通用的、文档完善的 API，它允许将任何操作的功能性声明性表示为异步数据流，可以在任何线程中创建，并由来自不同线程的多个对象使用。

当您在 Java 8 上使用 RxJava 时，它的全部功能是显而易见的，最好是使用像 retrieval 这样的库。它允许您将操作简单地链接在一起，并完全控制错误处理。例如，考虑下面给出的代码`id`:指定订单的 int 和 apiClient:订单管理微服务的改进客户端:

```
apiClient
.getOrder(id)
.subscribeOn(Schedulers.io())
.flatMapIterable(Order::getLineItems)
.flatMap(lineItem ->
    apiClient.getProduct(lineItem.getProductId())
             .subscribeOn(Schedulers.io())
             .map(product -> product.getCurrentPrice() * lineItem.getCount()),
    5)
.reduce((a,b)->a+b)
.retryWhen((e, count) -> count<2 && (e instanceof RetrofitError))
.onErrorReturn(e -> -1)
.subscribe(System.out::println);
```

这将使用以下属性异步计算订单的总价:

*   在任何时候，最多有 5 个针对正在运行的 API 的请求(您可以调整 IO 调度器，为所有请求设置一个硬上限，而不仅仅是针对单个可观察链)
*   网络出错时最多重试 2 次
*   -1 以防失败(反模式 TBH，但这是另一个讨论)

此外，在 IMO 中，每个网络调用之后的`.subscribeOn(Schedulers.io())`应该是隐式的——您可以通过修改创建改进客户端的方式来做到这一点。对于 11+2 行代码来说还不错，即使它比 Android 更像后端。

# Kotlin 协同程序的优势

与 RxJava 不同，协程侧重于提供一种机制来编写异步代码，这种机制也可以部分顺序运行，允许省略代码中的回调，从而转化为更紧凑的代码，易于生成和重构。

协程通过对更多并发代码使用挂起函数，使回收的回调成为更有序的代码，从而避免了回收回调的使用，这些代码仍然可以轻松地在线程间切换。协程的好处是可以直接集成到 Kotlin 语言中，虽然它们是最近才出现的，但是已经有多个应用程序甚至其他库支持它们。

## Kotlin 和协程是编写更简洁代码的一种方式

由于协程是用标准的 Kotlin 编写的，这个库适合新的 Android 项目，因为 Google 宣布 Kotlin 是 Android 的新语言。Google 已经在其 Android API 中为 Kotlin 添加了强大的文档支持，所以深入研究一下可能是个更好的主意。

## 为管理线程提供了一种更简单的方法

就协程而言，它们提供了一种管理线程的简单方法，非常适合在需要时在后台线程上执行进程。这种范式还提供了并发和结构化并发的简单实现。它易于扩展和维护，使得程序员编写代码、解决 bug 或组织和清理代码的时间的生产周期和效率更好。

## RxJava 非常……复杂

它可以优化代码以提高应用程序的响应能力，而且非常容易扩展，但由于过度使用，维护起来非常困难。它的复杂性是具有挑战性的，到处使用它的趋势使得代码非常复杂，更难调试，因此，使程序员在它上面浪费更多的时间来解决错误，围绕它开发新的代码，甚至维护或重写代码，因为它不返回错误，并且调试的唯一方法非常原始。

## 协程比 RxJava 更高效

就性能而言，协程比 RxJava 更高效，因为它使用更少的资源来执行相同的任务，同时速度也更快。RxJava 使用更多的内存并需要更多的 CPU 时间，这意味着更高的电池消耗和可能的用户界面中断。

# 总之…

在这篇文章的最后，我要说的是，在处理异步数据和函数时，Kotlin 协同程序是一个更容易学习和使用的更快的选择

如果您目前对使用 RxJava 感到满意，我不会告诉您停止，我只是要求您看一看协程。

# 那么，协程还是 RxJava？

这是一个困难的选择，也很难对它们进行比较，因为 Kotlin 协程是一个精简的语言特性，而 RxJava 是一个相当庞大的库，有大量现成的操作符。

答案取决于您的用例场景。谢谢你把这个看完。

在 GitHub 上关注我:[https://github.com/GeraudLuku](https://github.com/GeraudLuku)

在 Linkedin 上关注我:【www.linkedin.com/in/luku-geraud-3530b2177 

## 参考

[](https://exaud.com/rxjava-vs-coroutines-part-ii/) [## RxJava VS Coroutines:你该选哪个？(第二部分)

### 我们回到 RxJava & Coroutines 101 的第二部分——加入我们，继续探索这个理论分析…

exaud.com](https://exaud.com/rxjava-vs-coroutines-part-ii/) [](https://developer.android.com/kotlin/coroutines) [## 借助 Kotlin 协同程序提高应用性能| Android 开发人员

### 协程是一种并发设计模式，您可以在 Android 上使用它来简化异步执行的代码…

developer.android.com](https://developer.android.com/kotlin/coroutines) [](https://kotlinlang.org/docs/reference/coroutines/coroutines-guide.html) [## 协同程序指南- Kotlin 编程语言

### 作为一种语言，Kotlin 在其标准库中只提供了最少的低级 API 来支持各种其他库…

kotlinlang.org](https://kotlinlang.org/docs/reference/coroutines/coroutines-guide.html)