# 学习编码:途中安装的东西

> 原文：<https://medium.com/analytics-vidhya/learning-to-code-things-installed-on-the-way-9f14a15efd28?source=collection_archive---------25----------------------->

学习编程意味着安装工具和设置东西。通常安装软件和设置是一件无聊的任务，我很可能会忘记——除非我写下来或者不得不重新做一遍(真痛苦！)在不同的笔记本电脑上。

关于如何开始编码、比较资源等等，有相当多的内容可以阅读。我读过的关于你可能会中途安装的所有东西的文章很少。所以我回忆了一下我的下载文件夹。这是一次指导之旅！

这是一个试图包装各种东西和我已经安装的东西。

![](img/cf27d1aa56b95b44dca9d77678c612d3.png)

【像叠木头一样叠软件:资源。马库斯·斯皮斯克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

## 1.RStudio:控制降价

学术界迫使你在书目格式化和发送不同种类的文件上浪费时间。有时你的乳胶纸不被接受。你需要一个. doc 版本。

这太好了，有 Pandoc 在那里为我们做了相当多的转换。如果你需要一本书，这里有 [**Rbookdown**](https://bookdown.org/yihui/bookdown/) 。如果你需要一个博客，这里有 [**Rblogdown**](https://bookdown.org/yihui/blogdown/) 。这些工具背后的思想让你真的想去学习 r。

反正 RStudio 很快就成了我的*扩展笔记本*。此外，在 markdown 中编写解释和文字时，有机会运行不同编程语言的代码，这使得它成为编写教程或通过编写教程来巩固知识的绝佳场所。[好的，我已经写了关于 RMarkdown 的文章，并在这里开始写。

## 2.Python(实际上是通过 Anaconda)

Python 是我编程的切入点。实际上，比起在解释器或 IDE 中输入命令，我读到的更多(感谢 humblebundle.com)。我首先从 python.org 下载了 Python，但是后来我选择了 Anaconda 的捆绑版本。

这里有更多的细节。]

除了 Python 之外，还有各种你可能要安装的插件(我指的是不同的程序，不是你 pip/conda 安装的包)。

## 3.金德莱根

如果你尝试用 Python 创作电子书，你可能会偶然发现亚马逊的 KindleGen，也就是亚马逊的创作工具。mobi 电子书。

下载它，放在你的图书转换程序的文件夹中，然后命令行操作它。

![](img/699afd00c22d0fd409a765d5f81a3c88.png)

【我希望我的桌面也像这张图一样干净。在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上 [chuttersnap](https://unsplash.com/@chuttersnap?utm_source=medium&utm_medium=referral) 拍摄的照片。]

## 4.chrome 驱动器

如果你计划做一些浏览器自动化，你可能最终需要 [ChromeDriver](https://chromedriver.chromium.org/downloads) 。ChromeDriver 将为你提供一个自动化测试和浏览器自动化的工具。

## 5.吉特和 MinGW

如果没有一些命令行工具，你就无法进行开发，如果不在 GitHub 上查找或自己使用它，你可能会遇到更大的困难。

可以一石二鸟，学习 git。要在 windows 上使用 git，你需要一个命令行工具。当你在这里下载 git 的时候，你可能会用 MinGW 带来一些 Unix。MinGW 将为您提供在 git 上操作的主要终端(以及更多的)。

## 6.Java 和 Gephi

用 Python 探索图形让我接触到了 Gephi，一个绘制图形的工具。它需要安装 Java 运行时。另外，你读的关于面向对象编程的书越多，提到 Java 的就越多，然后有人告诉你关于 Head First 的书…

所以是时候安装 Java 运行时环境(JRE)来取悦 Gephi 和 Java 开发者工具包(JDK)来有机会用 Java 写 hello world 了。[以下是两个](https://docs.oracle.com/javase/10/install/installation-jdk-and-jre-microsoft-windows-platforms.htm#JSJIG-GUID-A7E27B90-A28D-4237-9383-A58B416071CA)的 Windows 说明。

你可能会研究 Eclipse，即 Java 的 IDE。(尽管如此，Gephi 仍然不能正常工作)。

## 7.Visual Studio 代码

好了，我们进入另一种编程语言(Java)。另外，我们知道 JavaScript 就在附近。这部分是由于一些命名混乱，部分是因为我们正在摆弄一个有 HTML 和 CSS 文件的网站，并且渴望添加一些 JavaScript。

存储是有限的，这是你开始感觉需要一把瑞士军刀来控制不止一种编程语言的地方。好吧，RStudio 允许我们运行不同的代码，但 Spyder 或 Jupyter 笔记本上没有多少好东西。

所以，稍微探索一下，你就能接触到 Visual Studio 代码(VS 代码)。您可以[在这里](https://code.visualstudio.com/Download)下载它，您将获得跨不同编程语言的一致布局。

它与命令行工具很好地集成在一起(键入“code”在你当前的目录下运行 VS 代码)，一个漂亮的深色主题，等等。它似乎比 Spyder 更轻，但这只是一个未经测试的印象。

您可以下载扩展来支持不同的编程语言。尝试用 C 或 Java 编译 hello world(并为类和 main 的问题做好准备)。

## 8.适配(使用 Node.js 和 MongoDB)

接下来是 JavaScript。再说一次，在我进入输入一些 JS 的思维模式之前，我已经阅读了更多关于它的内容。

我发现 Node.js 捆绑了一个包管理器(叫做 **npm** )，让你觉得‘嘿，pip，conda 欢迎回来！’与基于 JavaScript 的标准版本略有不同。

一个共享的每周项目包括试图让[调整电子学习框架](https://www.adaptlearning.org/)并运行。先决条件说你需要 git (check)以及 Node.js、grunt 和 MongoDB。

这就是我前往[下载节点](https://nodejs.org/it/download/)和[下载 MongoDB](https://www.mongodb.com/download-center/compass) 的路线。

那是相当多的东西，不是吗？无论如何，当我想开始输入一些 JavaScript 时，我已经有 Node 和 VS 代码在那里等着我了。(所以，到最后，似乎事情就水到渠成了)。

## 9.。NET Core SDK for C

最后，出于偶然的原因，我最终需要执行一些 C#。VS 代码已经在运行 C 了，但是——不——那不行。所以我意识到我需要安装[。NET Core SDK](https://dotnet.microsoft.com/download/dotnet-core/3.1) 启动并运行 C#。

![](img/c7819a31c82bb6e45f4b003aa7d190eb.png)

[安装你需要的工具是一个很小很小的步骤，仍然…照片由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [Hayley Catherine](https://unsplash.com/@hayleyycatherine?utm_source=medium&utm_medium=referral) 拍摄。]

## 已下载(但未安装)

哦，还有更多我已经下载但尚未安装的内容:

*   [android-studio](https://developer.android.com/studio) :使用你的 Java 开发移动应用。请注意，不久你可能需要学习科特林；
*   [postman](https://www.postman.com/downloads/) :测试 API，提出请求；
*   [Godot](https://godotengine.org/download/windows) :支持多种编程语言的游戏开发引擎。

## 概述

好了，这些是我安装的东西。列表是一件令人印象深刻的事情:

*   RStudio
*   python(via python . org)；
*   Python(通过 Anaconda)；
*   亚马逊 KindleGen
*   ChromeDriver
*   Git
*   MinGW；
*   VS 代码；
*   JRE
*   JDK；
*   Node.js
*   MongoDB
*   。NET Core SDK。