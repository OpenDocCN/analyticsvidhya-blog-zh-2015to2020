# Windows 终端:Windows 程序员必备的生产力工具

> 原文：<https://medium.com/analytics-vidhya/windows-terminal-a-must-have-productivity-tool-for-windows-programmers-9270d9dc4bfb?source=collection_archive---------8----------------------->

## 使用新工具 windows 终端管理编程工作的提示和技巧

![](img/210da8ac60fa4876e7fb008a2debb486.png)

作为一名程序员，我们经常需要使用不同的命令行工具，如命令提示符(CMD)、PowerShell、Anaconda、google cloud SDK、Azure SDK 等..所有这些工具都是相互独立的。你可能会觉得这些都可以自己搞定。然而，我确信集成工具 Windows 终端将会改变你的编程效率。让我们看看 Windows 终端。

**Windows 终端的特性**

Windows 终端是一个面向命令行工具和 Shell(如命令提示符、PowerShell 和 Windows Subsystem for UNIX (WSL ))用户的现代终端应用程序。微软在 2019 年 5 月做了初始发布，2020 年 7 月做了预览版，2020 年 8 月做了第一个稳定版。

Windows 终端具有以下重要功能:包括集成任何命令行工具、多个选项卡、Unicode 和 UTF-8 字符支持、GPU 加速文本呈现引擎，以及创建自定义主题、文本、颜色和背景的能力。

让我们先安装它，然后探索几个关键特性。

**安装 Windows 终端**

Windows 终端的安装超级简单快捷。可以通过微软商店一键安装 Windows 终端。

![](img/454865d39ec4ae845a0f33f54096a1f9.png)

**如何集成所有命令行工具**

Windows 终端默认初始安装包括命令提示符、Windows PowerShell、Azure 云壳、Ubuntu。我们可以添加其他常用的工具。下面是我添加到我的 Windows 终端的一些工具的截图:

![](img/1fbfa8aab806caa5536a21d0fdeee26b.png)

从截图可以看出，我已经添加了 Anaconda、gcloud SDK、Git Bash 和 PuTTY。我将使用 Anaconda 作为例子来演示如何向 Windows 终端添加新的应用程序。

通过单击菜单上的“设置”,您的计算机将使用 microsoft visual studio 打开配置文件(settings.json ),您将在该文件中找到列出了您当前包含在 Windows 终端中的所有的配置文件设置。您可以通过向列表中插入项目来添加任何其他工具。以下是我为安康达添加的项目:

```
{
        // Make changes here to the cmd.exe profile.
        "guid": "{799fa555-861d-4a1e-ac05-f44dfe39bdb6}",
        "name": "Anaconda",
        "commandline": "cmd.exe /K C:\\ProgramData\\Anaconda3\\Scripts\\activate.bat",
        "hidden": false,
        "icon": "C:\\ProgramData\\Anaconda3\\Menu\\anaconda-navigator.ico",
        "startingDirectory": "%USERPROFILE%"
            }
```

在“guid”上:您可以通过以下代码在 PowerShell 中生成新的 guid:

```
[guid]::NewGuid()
```

您可以将生成的 guid 复制并粘贴到这个新项目中。

你可以随意取名字。我将这个链接命名为“Anaconda”。

下一个关键是“命令行”信息。要找到这些信息，我们首先进入系统菜单，右键单击“Anaconda 提示符(Anaconda3)”进入“打开文件位置”，如下所示:

![](img/cc52e7666e809c9d3f346ffa848d0e08.png)

Anaconda 提示链接的属性

在文件位置文件夹中，我们会看到 Anaconda 下的所有快捷方式。继续右键单击“Anaconda Prompt (Anaconda3)”，我们可以获得以下屏幕截图:

![](img/da3936b16c045b7767c092f1d52318cf.png)

Anaconda 提示符的命令行信息

命令行信息应该来自“目标”框，“起始目录”来自“起始位置”框。

图标等其他信息是可选的。你可以随意定制图标。

我们可以用同样的方法添加“谷歌云 SDK 外壳”的快捷方式。你可以得到如下信息:

```
{
    "guid": "{8dbe5f27-48f1-4ce3-b695-ec3dde7dcd6d}",
    "hidden": false,
    "name": "gcloud SDK",
    "commandline": "C:\\windows\\system32\\cmd.exe /k \"C:\\Users\\ppeng\\AppData\\Local\\Google\\Cloud SDK\\cloud_env.bat\""
}
```

git bash 信息可能如下所示:

```
{
    "guid": "{05e3ab66-d88a-42b5-9431-ead17851486d}",
    "hidden": false,
    "name": "Git Bash",
    "commandline": "C:\\Program Files\\Git\\git-bash.exe --cd-to-home"
}
```

PuTTY 信息可能看起来像

```
{
    "guid": "{66444279-3522-4e8d-a0f7-d288e6139a39}",
    "hidden": false,
    "name": "PuTTY",
    "commandline": "C:\\Program Files\\PuTTY\\putty.exe"
}
```

基于所有应用程序的顺序，Windows 终端自动为每个应用程序创建快捷方式:ctrl + shift + number 1，2，…n

**功能:搜索命令历史**

在工作会话期间，您可能已经键入了许多命令行，并且经常想要重用其中的一些命令。通过滚动历史记录来查找正确的历史记录可能会花费您大量的时间。使用 Windows 终端，您可以通过按键“Ctrl+Shift+F”打开历史命令搜索框来快速定位命令行。

**功能:多个标签**

我们经常需要同时打开许多应用程序。在这么多打开的应用程序窗口之间导航可能会浪费一些时间。Windows 终端多选项卡功能可以帮助更好地组织和管理这些应用程序。通过预选的图标，我们可以轻松识别我们想要选择的应用程序。我们还可以重命名选项卡或对选项卡进行颜色编码，以帮助我们区分选项卡。使用同一个选项卡，我们还可以在同一个选项卡中使用多个窗格。更多详情可以参考[https://docs . Microsoft . com/en-us/windows/terminal/tips-and-tricks](https://docs.microsoft.com/en-us/windows/terminal/tips-and-tricks)。

**功能:Windows 终端窗格**

有时您可能想要在同一个标签中打开几个窗格。您可以通过 Windows 终端窗格来实现。窗格使您能够在同一个选项卡中并排运行多个命令行应用程序。

![](img/eb1e2c68c1a5c16bcd89793e65548a6c.png)

图片来源:[https://docs . Microsoft . com/en-us/windows/terminal/images/open-panes . gif](https://docs.microsoft.com/en-us/windows/terminal/images/open-panes.gif)

alt + shift + +':在当前窗格中创建一个新的垂直窗格。

alt + shift + -':在当前窗格中创建新的水平窗格。

' ctrl + shift + w ':关闭当前窗格。

所有这些设置都在 keybindings 部分配置。您可以参考[文档页面](https://docs.microsoft.com/en-us/windows/terminal/panes)了解更多详情。

通过探究这些奇妙的特性，我们可以发现所有这些特性在设置文件中都有明确的定义。我们甚至可以尝试按照相同的方法定义一些定制的特性。

有了 Windows 终端，我已经爱上了命令行编码。