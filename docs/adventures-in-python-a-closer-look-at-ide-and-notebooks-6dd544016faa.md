# Python 中的冒险:IDE 和笔记本的近距离观察

> 原文：<https://medium.com/analytics-vidhya/adventures-in-python-a-closer-look-at-ide-and-notebooks-6dd544016faa?source=collection_archive---------15----------------------->

开始这个系列来记录我学习语言的冒险经历。

作为一门优雅的语言，ython 本身就很棒。这是一种被广泛应用于不同领域的解释性语言。它的语法足够高级，所以我们可以更专注于解决问题，而不是被这么多规则所淹没。

作为我探索 Python 作为数据科学工具的冒险的一部分，创建直接应用一些概念的简单项目是一个很好的练习。还有什么比创造游戏更有趣的呢！

我第一次接触 Python 是在一年多以前，当时我使用基于 Python 的游戏引擎创建了一个简短的视觉小说游戏。这与我现在所做的完全不同，因为一开始，它仍然使用 Python 2；第二，游戏引擎有一个令人印象深刻的函数库和其他对象，这使得编码变得更加容易。我迷上了故事情节和游戏逻辑。这就是 Python 的美妙之处。它让你专注于你真正想做的事情。

与其他数据科学工具一样，我需要学习 Python 3，因为它就像我在分析丛林中的瑞士军刀一样。我的第一个任务是创建一个显示玩家姓名和分数的测验。

我目前正在使用 Linux Mint 桌面，同时[正在学习一大堆伟大的数据科学工具](/analytics-vidhya/learning-data-science-tools-on-linux-41fca1723b5a)。屏幕截图、导航和其他上下文可能与使用其他操作系统的用户不同。

## IDE 一览

对于做过编程的人来说，IDE(也称为集成开发环境)是一个熟悉的工具。几乎每种语言都有专有的和开源的 IDE 选项。IDE 的例子有代码、Eclipse、NetBeans。Python 实际上有自己的 IDE，叫做 IDLE。

IDE 不仅仅是一个文本编辑器(遗憾的是，Notepad++不是 IDE)。文本编辑器只是用来编写或编辑代码行的，但是 IDE 允许你编译、解释、运行、调试等等，而不仅仅是一个纯文本编辑器。

## Spyder IDE

Spyder 是一个广泛使用的科学编程 IDE，专门用于 Python。它很容易集成更科学的 Python 库，如 *pandas* 、 *Matplotlib* 、 *NumPy* 和 *SciPy* 。毕竟，Spyder 是科学 Python 开发环境的缩写。

我目前使用的是 Spyder 3，通过 apt 安装。然而，Anaconda Navigator 拥有更新的 Spyder 4。我打开了我的 *quiz.py* 文件，探索它的特性。

使用它就像使用其他 IDE 一样。我觉得即使是新手用起来也挺方便的。使用起来并不令人畏惧，你可以专注于需要做的事情。

![](img/42e723bc50ae0cda88ff5c728a7c4dc5.png)

Spyder 界面简单，易于导航。右下部分的控制台运行左侧面板中显示的代码。

## 笔记本到底是什么？

直到最近，我才接触到笔记本的概念。朱庇特一直被人津津乐道；Google 发起了 Colaboratory(或 Colab)的开发；甚至 Spyder 还有一个插件叫 Spyder Notebook。

对于像我这样的门外汉来说，揭开笔记本的神秘面纱是一个挑战。然而，我心中的探索者并没有放弃。我用笔记本，因为我看到我的导师和其他人在网上分享他们的代码。

在笔记本中，你可以自由地混合使用代码、文本、情节和其他视觉效果，并且它们中的每一个都可以放在一个**单元格**中。笔记本其实就是细胞的集合。可以通过使用 Shift+Enter 或点击每个单元的*播放*按钮来运行每个单元。

笔记本不仅仅是一个编写和运行代码的地方。这种体验与使用 IDE 截然不同。我看到有人甚至用笔记本来教学和讲故事。

## Jupyter 笔记本

不像 Spyder 可以直接从我的主菜单打开， [Jupyter](https://jupyter.org/) 需要更多的步骤来启动。

![](img/2b15118a2eaf75dba288ed6d1ddeebcf.png)

从我的主目录，我需要导航到 Jupyter 笔记本文件夹。

从我的主目录，我需要导航到 Jupyter 笔记本文件夹。然后我从那里打开我的终端，使用命令 *jupyter notebook* 来打开这个工具。

![](img/91dcbbfecbe05f42509df3e2db5e0e35.png)

这就是朱庇特笔记本如何启动的。然后在浏览器上打开。

我打开了之前用 Spyder 打开的同一个 quiz.py 文件。这是代码在 Jupyter 中的显示方式。当我悬停在语言菜单上方时，它显示了 Jupyter 可以支持的其他语言的列表。这恰恰说明了 Jupyter 是多么的有用和强大！

![](img/fa27788f0d01df95f573b6d5490f80ca.png)

像 Jupyter 这样的笔记本可以阅读有文化的语言，这促进了机器和人类语言之间的可读性和理解。

Jupyter 可以做的另一件事是转换一个* *。py* (Python)文件转换成* *。ipynb* (IPython 笔记本)文件。Spyder 无法读取笔记本文件，除非使用笔记本插件。我在 Colab 上打开了新的 *quiz.ipynb* 文件，如下图所示。

## 协作室—基于云的笔记本电脑

由于 Jupyter 是一个开源项目，其他开发人员可以从中创建自己的笔记本。这就是谷歌对 Colab 所做的事情。除此之外，他们把它放在谷歌服务器上，这样任何人都可以立即访问它，而不需要安装或设置任何东西。作为奖励，文件可以上传到 Google Drive 或 GitHub。

![](img/932b9d5a37d0f52df9929f86f9d0a715.png)

看到右上角的内存和磁盘指示器了吗？这就是我的测验笔记本在那一刻消耗的资源。

就像在 Jupyter 中一样，Colab 使用单元格，它们可以是文本或代码形式。我的代码可以通过在单元格被选中时单击 play 按钮或按 Shift-Enter 来运行。

![](img/8a88e4b3ed52901e8e0c0295279af086.png)

代码下面是程序实际运行的地方。输入函数要求用户输入。

![](img/bcdfb9ac96a4f68d97b3b732c2dec260.png)

我自己测验得了满分！是啊，我很厉害。

探索 Spyder、Jupyter 和 Colab 很有趣。每一种都有自己的优点，我可以根据自己的需要使用其中的任何一种。数据科学可能是一个要求很高的领域，精通这些工具肯定会为任何想进入该行业的人工作。

查看我的[测验文件](https://github.com/rochderilo/FTW-Py-Quiz)。收听我接下来的 Python 冒险！

*(这是我系列的第一篇文章。下面是第* [*第二*](/@roch.derilo/adventures-in-python-creating-a-quiz-game-with-fancy-features-16837259ad1a) *。)*