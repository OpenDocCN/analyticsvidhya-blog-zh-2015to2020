# Python 中如何在线程和进程之间显式共享数据？

> 原文：<https://medium.com/analytics-vidhya/python-tips-multithreading-vs-multiprocessing-data-sharing-tutorial-52743ed48825?source=collection_archive---------0----------------------->

## Python 技巧教程

## 指南:抛弃复杂细节，快速开始使用数据共享。

![](img/ebcb40a25c5ac7560597a214c819a205.png)

[https://unsplash.com/photos/CyFBmFEsytU](https://unsplash.com/photos/CyFBmFEsytU)

作为多线程和多处理的新手，并行工作的动机是推动代码执行。然而，一旦你让任务变得更复杂，你就会陷入困境。

# 从哪里开始？

> 这篇文章的重点是“快速入门”，而不是完整的解释，但我保证这将使你在深入真正扎实的知识之前更容易理解。

最常见的挑战是**多线程和多处理**之间的数据共享，与此主题相关的大量资源已经存在。看看人们是如何用不同的方式解释同一个问题的。这些是我学习时读过的最好的文章。

[](/@bfortuner/python-multithreading-vs-multiprocessing-73072ce5600b) [## Python 中的线程和进程简介

### 并行编程初学者指南

medium.com](/@bfortuner/python-multithreading-vs-multiprocessing-73072ce5600b) [](/analytics-vidhya/multiprocessing-for-data-scientists-in-python-427b2ff93af1) [## Python 中数据科学家的多重处理

### 如果不能全部使用，为什么还要花钱买一个强大的 CPU 呢？

medium.com](/analytics-vidhya/multiprocessing-for-data-scientists-in-python-427b2ff93af1) [](/better-programming/python-memory-and-multiprocessing-ed517b8db8fd) [## Python 内存和多重处理

### 以下是如何提高 CPU 密集型程序的速度

medium.com](/better-programming/python-memory-and-multiprocessing-ed517b8db8fd) [](https://www.pythonforthelab.com/blog/handling-and-sharing-data-between-threads/) [## 在线程之间处理和共享数据

### 在 Python 中使用线程时，您会发现能够在不同的任务之间共享数据非常有用。其中一个…

www.pythonforthelab.com](https://www.pythonforthelab.com/blog/handling-and-sharing-data-between-threads/) 

但为了简单起见，我将重点放在如何让你开始，所以如果你想快速获得它，记住这两种方法以获得完整的想法:

1.  线程间共享数据:可变对象。
2.  进程间共享数据:进程间通信。

![](img/e5030fd723161b80421c9f20ab423f01.png)

# 可变对象

当您在全局范围内创建一个可变对象(如列表或字典)时，只要您将它用作线程参数，它就会共享相同的内存地址，这在 C 或 C++等低级语言中称为“指针”。因此，无论何时调用变量，我们都可以很容易地更改数据。

粘贴并执行我！！！

输出应该是两个相同的 ID 号，全局列表也被更改。这个很简单，你也可以把列表形式转换成字典，你可能会得到同样的结果。

事实上，python 中没有指针，使用可变对象是模拟概念最简单的方法。您也可以使用:

*   自定义类:使用`@property`decorator 来扩展 dictionary 的思想。
*   `ctypes`模块:一个内置的标准库，用来创建真正的 C 风格指针。

关于指针以及如何使用自定义类和`ctypes`模块的更多细节。点击下面的文字。

[](https://realpython.com/pointers-in-python/#simulating-pointers-in-python) [## Python 中的指针:有什么意义？-真正的蟒蛇

### 如果你曾经使用过像 C 或 C++这样的低级语言，那么你可能听说过指针。指针允许…

realpython.com](https://realpython.com/pointers-in-python/#simulating-pointers-in-python) 

# 进程间通信

> 为什么不像线程那样使用指针来共享数据呢？

问题是进程间禁止“指针”。否则，任何人都可以在其他正在运行的应用程序中修改数据。

尽管如此，在两个进程之间使用“共享内存”是可能的，但它将由不同的指针表示，这被称为线程间通信。

最简单的方法是在`multiprocess`模块中使用`Array`创建共享对象，共享对象可以被子进程继承。

粘贴并执行我！！！

遗憾的是，`concurrent`模块不支持进程间共享内存，`multiprocessing`模块更好学，更好实现。

这里有一些你可以使用的高级模块。[点击我](https://docs.python.org/3/library/ipc.html)。

这些是分享数据的最基本的用法，还有很多你最好熟悉的关键词，可以在网上搜索:

1.  多线程队列
2.  多线程锁
3.  多重处理队列
4.  多重处理管道
5.  蟒蛇 GIL

举几个例子，但是简单使用就足够了。我希望你能得到一些关于如何开始类似话题的想法。