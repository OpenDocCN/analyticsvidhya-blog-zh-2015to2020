# 如何在 Raspberry Pi 中自动运行 python 脚本

> 原文：<https://medium.com/analytics-vidhya/how-to-automate-run-your-python-script-in-your-raspberry-pi-b6fe652443db?source=collection_archive---------0----------------------->

![](img/2096b8b36fa3460f9203c420a54f34fc.png)

大家好，今天我们将讨论自动化任务。你们中的许多人可能拥有 python 脚本来帮助你做一些事情。我有一个。我的 python 脚本是一个日常新冠肺炎机器人。它只是将日常案例的通知消息推送到 LINE 应用程序和我的 Twitter 帐户。只是更好地了解情况，不是吗？

嗯，我不想每天手动运行它，因为如果我这样做，它可能会远离“机器人”这个词。我决定把它保存在我的单板电脑里，我有一台叫做树莓派的电脑。它也 24/7 运行，所以，这是一个好地方。

我假设你们所有人的单板计算机上都安装了 Linux。在 Linux 中有一个用于任务调度的工具，叫做**【Crontab】**。今天，我们将使用 crontab 运行我们的 python 脚本。

## 介绍

> 最好了解一下 crontab

首先，我们可以通过打开终端并传递以下命令来访问 crontab:

```
crontab -e
```

*(如果你第一次运行，它可能会要求你选择要编辑的内容。我会推荐 Nano 或者 VIM。它很容易使用。)*

打开编辑器后，它将向您展示该命令是如何工作的。可以总结如下:

```
* * * * * command to be executed
 - - - - -
 | | | | |
 | | | | ----- Day of week (0 - 7) (Sunday=0 or 7)
 | | | ------- Month (1 - 12)
 | | --------- Day of month (1 - 31)
 | ----------- Hour (0 - 23)
 ------------- Minute (0 - 59)
```

*   第一个`*`是 0–59 之间的分钟范围
*   第二个`*`小时范围在 0–23 之间
*   第三`*`日的范围是在 1-31 之间吗
*   4 号`*`月份范围在 1-12 之间
*   5 日`*`是工作日的范围在 0–6 之间(0 是星期天，6 是星期六)
*   **第 6 个**T5**是运行脚本的命令**

例如，如果你想在每天 10:45 做一些事情，你可以把命令写成:

```
45 10 * * * yourcommand
```

对于那些有困难并且不确定命令语法的人，你可以使用下面的网站来了解公会的想法。

 [## crontab . guru——cron 调度表达式编辑器

### 我们创建了 Cronitor，因为 cron 本身无法在您的作业失败或从未启动时提醒您。Cronitor 易于集成…

crontab.guru](https://crontab.guru/) 

或者，如果您想通过选择日期和时间来执行相反的操作，并且需要生成命令，您可以访问下面的网站:

 [## Crontab 生成器-生成 crontab 语法

### 如果您想要定期执行任务(例如，发送电子邮件、备份数据库、进行定期维护等)。)…

crontab-generator.org](https://crontab-generator.org/) 

## Python 脚本任务计划

现在，我们都知道 crontab 是如何工作的。让我们将它与我们的 python 脚本集成在一起。首先，我将创建一个简单的**‘hello world’**python 脚本，并将其保存在桌面中。转到终端和`cd Desktop`。然后，我用 Nano 创建 python 脚本为`nano pytest.py`。在 nano 编辑器中，我编写了经典的代码。

```
print('hello world')
```

按下“Cmd+x ”,或者如果你是 Windows 的人，使用“Ctrl+x”。然后，您可以按“y”保存并退出编辑器。让我们试着运行代码，打开终端并键入`python3 pytest.py`。终端应该显示如下内容:

```
hello world
```

工作正常！现在，我们可以将 python 脚本添加到 crontab 中。然而，在此之前，我们需要知道我们的 python 执行器在哪里。在 Linux 中，我们可以使用命令`which python3` *(我用的是 python3)* 找到执行程序的路径。通常，它应该位于以下路径中:

```
/usr/bin/python3
```

我们已经准备好自动化 python 脚本。关于 crontab，您可能需要了解的最后一步。crontab **在终端不显示任何输出**。要检查您的任务计划是否仍在运行，您可以创建日志文件来保存结果。这可以通过简单地对日志文件使用`>>`语法来完成，如下例所示:

```
>> /home/pi/Desktop/log.txt
```

上面的命令将在您的桌面上创建名为“log”的文本文件。它可以包含 python 脚本的任何输出。在这种情况下，每当脚本运行时，它都会追加“hello world”。您可以参考下面的完整命令:

```
* * * * * /usr/bin/python3 /home/pi/Desktop/pytest.py >> /home/pi/Desktop/log.txt
```

当您可以确保您的 crontab 正在工作时，您可能不需要创建`log.txt`。您只能为您的任务计划程序保留`* * * * * /usr/bin/python3 /home/pi/Desktop/pytest.py`。

仅此而已！简单有用就好。下次见。

你可以在 LinkedIn 上找到我，链接如下:

 [## sasiwut chaiyade cha-EY 高级顾问| LinkedIn

### Sasiwut 是泰国 EY 公司从事绩效改进(PI)的高级顾问。目前，主要重点是…

www.linkedin.com](https://www.linkedin.com/in/sasiwut-chaiyadecha/)