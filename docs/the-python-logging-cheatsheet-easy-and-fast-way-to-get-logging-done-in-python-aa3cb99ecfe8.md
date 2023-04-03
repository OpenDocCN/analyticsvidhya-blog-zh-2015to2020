# Python 日志记录清单。用 Python 实现日志记录的简单快捷的方法。

> 原文：<https://medium.com/analytics-vidhya/the-python-logging-cheatsheet-easy-and-fast-way-to-get-logging-done-in-python-aa3cb99ecfe8?source=collection_archive---------5----------------------->

拥有良好的日志对于监控应用程序、了解生产行为和捕获错误至关重要。python 中的日志制作得很好，也有很好的文档记录。然而，有时你只是需要快速完成工作，而你只是没有时间去阅读[优秀但相当长的官方文档](https://docs.python.org/3/howto/logging.html)。

![](img/a5153bb899b16438fc8c4464b4b3037a.png)

日志记录对于您了解应用程序的行为非常重要，并且有助于 IT 运营拥有一个专用的控制面板来快速监控和分析应用程序问题。

在本文中，我们将尝试找出快速的方法来提高日志记录的效率。在我们开始之前，有一些关于 python 日志的基本术语:

*   **Loggers** :提供几种方法允许应用程序运行时记录日志的主要参与者。
*   **处理器**:它们接收你的日志信息，并将它们发送到特定的位置，比如文件或控制台。
*   格式化程序:它们让你的日志以你想要的方式出现，以你可以定义的特定格式出现

事不宜迟，让我们直接进入主题，获取一些快速方法，让您的 python 应用程序立即登录。

# 场景 1:我有一个简单的应用程序，我需要设置一些基本的日志文件

为简单的应用程序完成日志记录的最快方法是使用 [logging.basicConfig](https://docs.python.org/3/library/logging.html#logging.basicConfig) ，它将创建一个 StreamHandler(和一个 FileHandler，如果我们指定了文件名)并将其添加到根日志记录器。
配置如下:

```
*import* logging
logging.basicConfig(filename='my_log_file.log',level=logging.INFO)
```

然后按如下方式使用它:

```
*import* logging
*from* time *import* sleep

*def* setup_logger():
    logging.basicConfig(filename='my_log_file.log', level=logging.INFO)

*def* my_app_logic():
    logging.info("Just entered the function")
    sleep(0.1)
    logging.info("Just after the sleep")
    *try*:
        res = 1 / 0
    *except ZeroDivisionError*:
        logging.exception("Attempted division by zero")

*if* __name__ == '__main__':
    setup_logger()
    my_app_logic()
```

这将在我们的 *my_log_file.log* 文件中产生以下输出:

```
INFO:root:Just entered the function
INFO:root:Just after the sleep
ERROR:root:Attempted division by zero
Traceback (most recent call last):
  File "/xxxx/simple_application.py", line 14, in my_app_logic
    res = 1 / 0
ZeroDivisionError: division by zero
```

如果您还想显示时间，只需添加如下格式:

```
logging.basicConfig(filename='my_log_file.log', format='%(asctime)s - %(message)s', level=logging.INFO)
```

这会将日志文件中的输出消息更改为:

```
2020-07-20 17:05:29,686 - Just entered the function
2020-07-20 17:05:29,790 - Just after the sleep
2020-07-20 17:05:29,790 - Attempted division by zero
...
```

**何时使用**:简单的应用程序，你只需要在没有太多模糊的情况下完成一些日志记录。最少使用外部库，并且不需要对如何执行日志记录进行太多控制。

**何时不使用**:当您需要对如何执行日志记录或任何多进程应用程序进行细粒度控制时。

# 场景 2:我有一个单线程/多线程应用程序，它正在使用几个库，我想完成一些日志记录

如果您的应用程序开始变得稍微复杂一些，并且您需要对日志进行更多的控制，我们应该避免使用根日志记录器，相反，让我们创建自己的日志记录器，如下所示:

```
logger = logging.getLogger('non_simple_example')
```

让我们添加处理程序来将日志发送到控制台和/或文件，格式化程序来指定我们喜欢的日志记录格式:

```
*# create console handler and set level to info* stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

*# create file handler and set level to info* file_handler = logging.FileHandler(filename='my_log_name.log')
file_handler.setLevel(logging.DEBUG)

*# create formatter* formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

*# add formatters to our handlers* stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

*# add Handlers to our logger* logger.addHandler(stream_handler)
logger.addHandler(file_handler)
```

然后在整个应用程序中使用 logger，例如:

```
logger.info("Info message")
logger.debug("Debug message")
```

我们将根据日志级别和处理程序配置将事情记录到控制台/文件中。在前面的例子中，文件处理程序有一个调试级别，而控制台处理程序有一个信息级别。这意味着具有调试级别的每个日志不会显示在控制台中，但会显示在日志文件中。

这是控制台输出:

```
2020–07–20 17:13:34,439 — non_simple_example — INFO — Info message
```

以及日志文件中的以下输出:

```
2020-07-20 17:13:34,439 - non_simple_example - INFO - Info message
2020-07-20 17:13:34,439 - non_simple_example - DEBUG - Debug message
```

在此找到完整代码[。
这种方法在多线程应用中也能很好地工作，事实上日志模块是**线程安全的**。然而，对于需要更多努力的多进程日志来说，这是行不通的。](https://gist.github.com/arocketman/5f9e2043425c57d17676af1407bfe1e1)

**何时使用**:当您需要对日志进行更多控制，并且希望避免使用/污染 root logger 时。

**何时不用**:如果你有多个进程。

# 场景 3:我的日志越来越大，怎么办？

非常简单，从简单的文件处理器切换到更复杂的[旋转文件处理器](https://docs.python.org/3/library/logging.handlers.html#logging.handlers.RotatingFileHandler)，当日志文件达到特定大小时，它将切换日志文件:

```
handler = RotatingFileHandler('my_log.log', maxBytes=1000, backupCount=10)
```

如果您想在每天/每周/每月结束时切换日志，您可以使用[TimedRotatingFileHandler](https://docs.python.org/3/library/logging.handlers.html#logging.handlers.TimedRotatingFileHandler):

```
handler = TimedRotatingFileHandler(filename="my_log.log", when="midnight")
```

# 场景 4:我有一个多进程应用程序，我需要记录日志。

从不同的进程中记录日志并不简单，这需要一些额外的工作。我建议不要采用快速而肮脏的解决方案，而应该通读官方文档[中的内容，该文档详细解释了正在发生的事情以及应对挑战的策略。](https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes)

解决这个问题的一个有趣的方法是使用一个 [WatchedFileHandler](https://docs.python.org/3/library/logging.handlers.html#watchedfilehandler) 并使用一个外部实用程序如 [logrotate](https://linux.die.net/man/8/logrotate) 来根据配置实际旋转日志。

# 结论

这绝不是一个详尽的日志指南，而是一个快速而简单的完成日志记录的方法。
要了解更多信息和详细指南，请参考[官方文档](https://docs.python.org/3/howto/logging-cookbook.html)，其中也提供了几个例子来让您体验一下。