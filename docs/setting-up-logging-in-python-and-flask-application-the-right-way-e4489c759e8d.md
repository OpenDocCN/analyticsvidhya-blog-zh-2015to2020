# 正确设置 Python 和 Flask 应用程序中的日志记录

> 原文：<https://medium.com/analytics-vidhya/setting-up-logging-in-python-and-flask-application-the-right-way-e4489c759e8d?source=collection_archive---------2----------------------->

![](img/f613db4505b42d65f2a750dab2adebf2.png)

> 调试比一开始写代码要难两倍。因此，如果你尽可能聪明地编写代码，从定义上来说，你没有足够的聪明去调试它。— Brian Kernighan，C 语言的共同创造者

但是一个好的日志文件可以使调试更加容易和愉快。😄

但是建立一个好的日志和警报管道是一门艺术，因为它需要随着应用程序的增长而发展，并且应该是可伸缩的。

默认的`Python`日志记录也没有帮助。它给出的信息要么太多，要么太少。文档到处都是。除此之外，还有多种方法可以用来设置它。

如果你想在应用程序中使用其他模块，那么如果你想打印这些模块的日志，那将是一场噩梦。

## **Python**提供的不同类型的 `**handlers**`

*   **StreamHandler**
*   **文件处理器**
*   **NullHandler**
*   **WatchedFileHandler**
*   **BaseRotatingHandler**
*   **旋转文件处理器**
*   **TimedRotatingFileHandler**
*   **SocketHandler**
*   **DatagramHandler**
*   **SysLogHandler**
*   **NTEventLogHandler**
*   **SMTPHandler**
*   **内存处理器**
*   **HTTPHandler**
*   **队列处理程序**
*   **队列监听器**

# ****Python 日志记录级别****

****调试****

> **详细信息，通常仅在诊断问题时感兴趣。**

****信息****

> **确认事情按预期运行。**

****警告****

> **表示发生了意想不到的事情，或者表示近期会出现一些问题(例如“磁盘空间不足”)。该软件仍按预期工作。**

****错误****

> **由于更严重的问题，软件无法执行某些功能。**

****危急****

> **严重错误，表明程序本身可能无法继续运行。**

# **当前实施**

1.  **实时错误通知应该发送到 Slack 频道或电子邮件(在调试错误之前，首先需要知道有错误)。**
2.  **完整的`stacktrace`应该保存在主机上的错误文件中，这些文件可以异步发送到一些集中式日志记录(Elasticsearch)。**
3.  **`access log`应保存在日志文件中以供审计。**
4.  **当日志变大时，必须相应地轮换和删除文件。**
5.  **在`debug`模式下，一切都必须在控制台中打印。**
6.  **在`production`中，只有绝对必要的日志才会被打印到控制台。**
7.  **不同的用例应该使用不同的`formatters`。**
8.  **当然，它应该很容易实现。**

**正确设置 Python 和 Flask 应用程序中的日志记录**

**在上面的实现中，您可以更改`handlers`以在不同通道之间切换。**

****参考文献****

**[https://gist . github . com/Deepak Sood 619/99e 790959 F5 EBA 6ba 0815 e 056 a 8067d 7](https://gist.github.com/deepaksood619/99e790959f5eba6ba0815e056a8067d7)**

**[](https://stackoverflow.com/a/61258002/5424888) [## 使用 Python 的 dictConfig 在 Flask 应用程序中为 Gunicorn 设置日志记录

### 我想通过电子邮件或 slack webhook 发送我的错误日志，并将访问日志保存在一个文件中。当我试着和…

stackoverflow.com](https://stackoverflow.com/a/61258002/5424888) [](https://docs.python.org/3/howto/logging.html) [## 日志 HOWTO - Python 3.8.2 文档

### 日志记录是跟踪某些软件运行时发生的事件的一种方式。该软件的开发人员添加了日志调用…

docs.python.org](https://docs.python.org/3/howto/logging.html)  [## 日志记录指南- Python 3.8.2 文档

### 这个页面包含了许多与日志记录相关的方法，这些方法在过去很有用。多次呼叫…

docs.python.org](https://docs.python.org/3/howto/logging-cookbook.html)**