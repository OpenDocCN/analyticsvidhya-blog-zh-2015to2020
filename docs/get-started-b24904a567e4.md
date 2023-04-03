# Python 训练营:入门

> 原文：<https://medium.com/analytics-vidhya/get-started-b24904a567e4?source=collection_archive---------29----------------------->

# 开始

欢迎光临！这是你学习 Python 编程语言的第一步。无论您是编程新手还是有经验的开发人员，学习和使用 Python 都很容易，因为与其他编程语言相比，它的语法相对简单。

Python 是一种解释型高级通用编程语言。1991 年，吉多·范·罗苏姆开发了 Python 编程语言。它用于 web 开发、机器学习、统计处理等等。

对于外面所有的新手来说，学习 Python 是如此的有趣，而且你已经选择了最令人惊奇和简单的教程来学习，所以开始吧！

# 装置

让我们来看看如何在 Windows 上安装 Python 3:

1.前往[https://www.python.org/ftp/python/3.8.3/python-3.8.3.exe](https://www.python.org/ftp/python/3.8.3/python-3.8.3.exe)

2.根据您的 PC，下载 32 位或 64 位 Python 3.x 的最新版本。

3.一旦您选择并下载了安装程序，只需双击下载的文件即可运行它。选中将 Python 3.x 添加到路径复选框。然后单击立即安装。

4.检查完窗口中选定的选项后，单击下一步。完成这些步骤后，安装将开始。

5.您可以通过在 cmd 中键入 python -v 来检查安装是否成功。如果它返回您已经安装的 Python 版本，那么恭喜您一切就绪。

如果您使用的是 Linux 操作系统，那么:

1.打开终端，使用以下命令开始安装:

```
sudo apt-get install python
```

2.允许该过程完成，并使用以下命令验证 Python 版本是否安装成功:

```
python --version
```

# 第一个节目！

让我们从创建第一个 Python 程序 helloworld 开始，它显示“Hello World！”。它可以在任何文本编辑器中完成。

在 Python 中，我们使用 print 语句输出文本:

```
print(“Hello World!”)
```

用保存您的文件。py 扩展名。

在命令行上运行 python 文件的方法是:

```
python <filename>.py
```

现在，打开命令行，导航到保存文件的目录，然后运行:

```
python helloworld.py
```

就这么简单。输出应为:

```
Hello World!
```

恭喜你，你已经编写并执行了你的第一个 Python 程序！

# 评论

在一行代码的开头添加#将使整行代码成为注释。运行代码时不会执行注释，所以可以用注释来留言。

示例:

```
#this is a comment.print (“Learning Python is fun!”)
```