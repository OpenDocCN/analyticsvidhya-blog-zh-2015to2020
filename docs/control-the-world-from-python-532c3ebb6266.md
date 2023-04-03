# 从 Python 控制世界

> 原文：<https://medium.com/analytics-vidhya/control-the-world-from-python-532c3ebb6266?source=collection_archive---------7----------------------->

## 使用 wexpect 从 Python 与 Windows 应用程序交互

![](img/62dea3962d76a0f6f5af829d34032c91.png)

照片由[卡斯帕·卡米尔·鲁宾](https://unsplash.com/@casparrubin?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

几个月前，一个阳光明媚的下午，我想从 Python 中控制一个 Windows 应用程序。这对我来说似乎是一件容易的任务，但是我还没有找到任何令人满意的解决方法。最后，我成为了名为[we expect](https://github.com/raczben/wexpect)的 [PyPI 项目](https://pypi.org/project/wexpect/)的维护者…

如果你想启动一个应用程序并得到它的输出，你的朋友是`[subprocess.communicate()](https://docs.python.org/3.7/library/subprocess.html#subprocess.Popen.communicate)`方法。但是，这不是真正的互动。如果您的子流程需要输入(与其输出同步)，可以使用`[subprocess.Popen.stdout.readline](https://stackoverflow.com/a/4417735/2506522)` 。(Pexpect 的`[popen_spawn](https://github.com/pexpect/pexpect/blob/master/pexpect/popen_spawn.py)`类使用了这种技术。)但遗憾的是，有些应用需要终端，至少是伪终端，这些应用不会使用`subprocess.Popen`进行交互。

这是我发现 wexpect 的地方，当时它在网上有几个变种，没有文档，没有测试，缺乏集成等等。现在[we expect](https://github.com/raczben/wexpect)运行在 Python 3.x 下，并上传到 [PyPi](https://pypi.org/project/wexpect/) ，所以可以用 pip 安装。

# 我们期待

[we expect](https://github.com/raczben/wexpect)在 Python for Windows 中有一个伪终端实现。它使用 [pywin32](https://github.com/mhammond/pywin32) 包，该包提供了对 [Win32 API](https://docs.microsoft.com/en-us/windows/win32/apiindex/windows-api-list) 的大量访问。Wexpect 在这个伪终端中运行子应用程序。这就是为什么所有的 Windows 应用程序都可以使用 Wexpect 进行交互。

## 要求

要使用 wexpect，您需要以下材料:

*   Windows PC。(在 Linux 上，您可以使用[PE expect](https://github.com/pexpect/pexpect)
*   Python 3.x，带 pip。

不幸的是，ide(py charm、Spyder 等…)不被[支持](https://github.com/raczben/wexpect/issues/1)，所以使用 cmd 命令提示符。

## 安装

要安装 wexpect，请运行:

`pip install wexpect`

它将安装 wexpect 本身，以及唯一需要的模块 pywin32。(开发使用运行`pip install wexpect[test]`

## 你好世界！

![](img/2cf3149005ff22feb27b6c8bda8f8327.png)

由[2 个摄影罐](https://unsplash.com/@2photopots?utm_source=medium&utm_medium=referral)在[的 Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

让我们使用 wexpect 编写第一个脚本:

```
'''
hello_wexpect.pyThis script lists the current directory and exits.
'''import wexpect# Start dir as child process
child = wexpect.spawn('dir')# Waiting for dir termination.
child.expect(wexpect.EOF)# Prints the directory content
print(child.before)
```

该脚本将打印目录内容的输出:

```
D:\medium>python hello_wexpect.py
hello_wexpect.pyD:\medium>
```

现在我们已经知道，子应用程序可以用`spawn()`方法启动。`expect()`方法用于捕捉子输出中的给定模式，这是现在输出的结尾。(在下一个例子中，我们将看到一个更复杂的`expect()`的例子)`before`字段保存期望模式之前的子输出。

## FTP 包装

下面的例子将产生 FTP 程序，所有输出将被打印到控制台，所有命令将被发送到子 FTP 应用程序:

```
# ftp_wrapperimport sys
import os
import re
import wexpect# Path of ftp executable:
ftp_exe = 'ftp'
# The prompt should be more sophisticated than just a '>'.
ftpPrompt = ['ftp> ', ': ', 'To ']# Start the child process
p = wexpect.spawn(ftp_exe, timeout=5)# Wait for prompt
p.expect(ftpPrompt, timeout = 5)# print the texts
print(p.before, end='')
print('wexpect-' + p.after, end='')while True:
# Wait and run a command.
    command = input()
    p.sendline(command)

    try:
        # Wait for prompt
        p.expect(ftpPrompt)

        # print the texts
        print(p.before, end='')
        print(p.after, end='')

    except wexpect.EOF:
        # The program has exited
        print('The program has exied... BY!')
        break
```

如果您运行了这段代码，您会得到一个`wexpect-ftp>`提示。[所有 FTP 命令](https://www.serv-u.com/features/file-transfer-protocol-server-linux/commands)都被接受，因为所有命令都被发送到子 FTP 应用程序。试试`help`或者`open`。用`bye`退出。

注意，`expect()` 的参数现在是一个列表，因为 FTP 在不同的命令中使用不同的提示。一个新的基本方法出现了，即`sendline()`。这会向孩子发送命令。最后一个新东西是`after`字段，它保存提示本身。

## 下一步是什么？

我们已经学习了 wexpect 模块的基本知识，但是，在大多数情况下这已经足够了。在[示例](https://github.com/raczben/wexpect/tree/master/examples)和[测试](https://github.com/raczben/wexpect/tree/master/tests)目录下可以找到更多的用法示例。

所以，现在可以用 Python 来控制窗口了。