# 自动清除日志文件

> 原文：<https://medium.com/analytics-vidhya/automate-log-file-clearing-9908bb55e70f?source=collection_archive---------12----------------------->

![](img/8f82ca8676c4379aeaaa8b4b0fffc0a7.png)

迈克尔·贾斯蒙德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

重复的日常活动会变得繁琐和累人。最近，我发现自己试图浏览个人电脑中的所有文件夹，搜索日志文件并一个接一个地删除它们。这个过程很累。出于这个原因，我写了一个 python 脚本来清除我的所有日志。

在我们继续之前，如果你正在阅读这篇文章，并且一直想知道什么是网络抓取，并且想学习如何抓取一个网站，[我的这篇关于简单的网络抓取的 5 个步骤的文章](/analytics-vidhya/5-steps-to-easy-web-scraping-17d063824c41)是给你的。看看吧！

本文中实现的解决方案可以在这里找到:

[](https://github.com/josephdickson11/python_utilities) [## josephdickson 11/python _ utilities

### 更轻松、更快速地完成工作在运行脚本之前，请始终运行“pip install requirements.txt ”,以避免错误和…

github.com](https://github.com/josephdickson11/python_utilities) 

**使用的工具**

1.  Python 编程语言

**使用的包**

1.Os.path:访问目录和文件

2.Datetime:创建数据实例和文件上次修改的时间

3.Beepy:用户通知警报

**求解过程**

为了有效地解决问题，要执行的每个步骤都被功能化。我将在本节中解释每个功能。如果你想跳过解释并回顾实现的解决方案，你可以在这里找到 github 链接。

1.  success_alert:提醒用户任务是否成功完成

2.error_alert:在执行过程中出现错误时提醒用户

3.check_path:检查个人计算机上是否存在用户指定的路径

4.confirm_directory:检查现有路径是否为目录。这是因为脚本需要一个目录，它在这个目录中循环选择特定的文件

5.create_file_path:检查子文件夹并遍历它们以创建 file_path 并删除今天创建的所有日志文件。

**用法**:要使用该解决方案，只需从这个[链接](https://www.python.org/downloads/)下载并安装 python，运行 pip install requirements.txt 并运行 github repo 提供的 python 脚本[。](https://github.com/josephdickson11/python_utilities)

**此外:请随意将您自己的 python 实用程序脚本贡献给这个**[](https://github.com/josephdickson11/python_utilities)****，让我们一起构建一个社区。期待你的拉请求！。****