# 认真设置 python

> 原文：<https://medium.com/analytics-vidhya/setting-up-python-conscientiously-b27dc0a690ff?source=collection_archive---------8----------------------->

![](img/f8418c5a88966e71c73383bca4f93dd4.png)

你可能是一个精力充沛的开发人员，准备编码下一个大东西，甚至是一个科学家，需要验证一个有前途的和新颖的想法！你喜欢人工智能，网络，密码或任何其他热门和趋势的概念。您可能认为 python 是一个很好的入门工具。而且你知道，它实际上是！Python 可能是最伟大、最成熟的社区之一。您有太多的实现示例，多个库提供了开箱即用的功能。各种各样的博客帖子已经被制作出来，以帮助你在这个漫长的旅程中，探索你自己的 pythonic 世界。

好吧，我们走吧，让我们开始写一些代码。第 0 步是获取某个`stack overflow` ed 例程，然后
执行它来断言它的输出。接下来你应该准备好发射了。

my_python_scirpt.py:

```
import pandas as pd
 import numpy as np if __name__=="__main__":
  # 
  # TODO: write some extra magnificent piece of code here
  # 
  print("something is happening here")> python my_python_script.py
```

不幸的是，您会得到以下错误:

```
ModuleNotFoundError            Traceback (most recent call last)
 <ipython-input-1-7dd3504c366f> in <module>
 ----> 1 import pandas as pd ModuleNotFoundError: No module named 'pandas'
```

在这一点上，你开始谷歌错误，以找到大量的解决方案，最终可能正确地工作在你的情况下。然而，具体的文章不应该被视为另一个修复手稿的问题。这是关于解释 python 的设置过程。我们不提供纯粹的步骤，而是要调查幕后发生了什么，以便能够减轻将来与 python 设置相关的任何问题。

声明:我基本上是一个苹果电脑迷。我确实在 macOS 上开发，在 Linux 上部署，而且通常是以文档化的方式。这就是为什么这篇文章主要是面向 macOS 的。然而，同样的原理也可以应用于 windows 系统。

# 预装 python

首先，任何 macOS 或 Linux 系统都是 python 就绪的。这意味着 python 已经安装，并且在大多数情况下已经包含在相应的`$PATH`环境变量中。Python 对于增强最终用户体验的多个脚本或软件组件来说是必不可少的。这就是 Python 成为任何 macOS (/Linux)发行版的一部分的原因。也就是说，在 shell 中键入`python`应该会触发交互式 python 解释器，提示您输入有效的 Python 表达式。但是 python 的安装并不是专门用于开发的(至少在 macOS 上是这样)。这是由于(a)安全问题以及(b)版本不兼容。关于最后一个，考虑 macOS 具有`python 2.7.x`(在特定帖子被编辑时)，它与苹果提供的任何基于 python 的功能兼容，但不与最新的`Keras`、`Pandas`或`Numpy`实现兼容，例如。或者想想还没有移植到最新 python 版本的研究工具，很可能它们还没有。因此，我们大部分时间都需要安装至少一个 python 版本才能安全轻松地工作。

# 点

另一个让我们生活变得更简单的东西是`pip`，python 包安装程序。`Pip`可以很容易地用来安装我们需要在其上构建的任何`site-packages`(即第三方组件)。例如，要安装广为人知的包`pandas`，我们可以在 shell 中键入以下命令:

```
> pip install pandas
```

在成功安装 pandas 之后，我们可以继续我们的 python 代码片段，导入并使用`pandas`。顺便说一下，对于那些没有意识到的人来说，`pandas`是一个很棒的数据预处理和轻量级分析工具。

当然`pip`不会像 python 一样从盒子里出来。我们需要手动安装它。安装 pip 更简单的方法是使用相应的软件包安装程序(即`brew` if `Mac` else `apt`)。通过键入以下命令:

```
> brew install python<PVersion>-pip #macOS , <PVersion> --> python version 2 or 3> apt install python<PVersion>-pip #Ubuntu, <PVersion> --> python version 2 or 3
```

我们应该安装`pip`，然后我们应该能够安装我们需要的 python 包。对吗？有一点...

`Pip`在`<some_path>/python<Version>/site_packages`目录下安装包，这是 python 解释器在遇到`import`代码行时查找的地方之一。这样，已安装的`pandas`包就可用于上面的代码片段。

python-pip 安装的两个路径变量(即`<some_path>`和`<Version>`)`<PVersion>`以及`$PATH`环境变量是本文讨论的大多数 python 安装难点的原因。

# 路径，环境变量

环境变量注册了所有可执行文件所在的目录。尽管像大多数环境变量一样是一个字符串，它的行为却像一个有序列表。每当请求一个可执行文件时，操作系统都会扫描`$PATH`来定位第一个匹配的文件。

```
> echo $PATH
  /Users/someone/.pyenv/shims:/usr/local/sbin:/usr/local/Manual_installs:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:
```

那是典型的`$PATH`布局。假设您已经安装了以下两个可执行文件:

*   `/usr/local/sbin/test`
*   `/sbin/test`
    击键测试将导致执行第一个，实际上是基于`$PATH`指示的顺序的第一个匹配。

# Python 多版本化

假设您的笔记本电脑提供了预装的`pythonX`，并且您已经安装了相应的`pythonX-pip`。然而，您最终需要导入一个与特定 python 版本`X`不兼容但与`Y`版本兼容的模块。显而易见的解决方案是安装所需的`pythonY`以及相关的，并祈祷其他模块都与`Y`版本兼容。

```
> brew install python3
> brew install python3-pip
```

这对你很有帮助。因此，每当您需要显式定义一个特定的 python 版本时，只需键入:

例如

```
> pip3 install <python_3_package>
> python3 my_python_3_code.py
```

但这是额外的工作，不那么优雅。是吗？

# Pyenv

您可以使用`pyenv`包，而不是手动指示 python 版本。`Pyenv`负责无缝安装和管理多个 python 版本。用户可以通过以下命令来指定`global`或`local` python 版本:

```
> pyenv shell 3.8.1  # sets python 3.8.1 to be the default python for # the current tty session
```

并且在下一时刻，键入`python`或`pip`应该被绑定到设置的版本。

这是通过一个名为垫片的概念实现的。Shims 是轻量级的可执行文件，它只是将您的命令传递给 pyenv。我认为 shims 更像是一个可执行的适配器模式。Pyenv 在`$PATH`前面注入一个垫片目录。这意味着通过键入`python`shims 的目录是第一个被访问的，猜猜看，python 可执行文件似乎就在那里！根据 PyEnv 原始文档，当您击键`pip`操作系统时:

*   在路径中搜索一个名为 pip 的可执行文件
*   在路径的开头找到名为 pip 的 pyenv shim
*   运行名为 pip 的 shim，它依次将命令传递给 pyenv

通过设置您喜欢的版本，命令`python`和`pip`被绑定到该版本。而这让 python 版本管理真的是小菜一碟！

请记住，pyenv 不是由 Python 引导的。它不依赖于任何 python 安装。因此，它的设置在概念上是非常清楚的。然而，缺点是它需要对已安装的版本以及它们所依赖的“东西”有一个很好的概述。考虑运行在特定 python 版本上的 python linter 的情况。全局 python 版本管理的灵活性可能会影响 linters 的行为。

# 虚拟

好的，然后你设置好`pyenv`，你就可以开始了。但是迟早你会遇到混合需求的问题。想象一下，在你的两个不同的项目中工作，项目 A 和项目 B，每个项目都需要不同的第三方包列表。在每个项目都是在不同的 python 版本下开发的情况下，混合需求问题乍一看可能并不明显。但这种情况并不常见。通常，你会在同一个 python 版本中处理多个项目。在全球范围内使用`pip`不应该被认为是明智的选择！除了一个高度损坏的`pythonX/site_packages/`目录之外，你将无法区分所有那些你为项目 A 安装的包和那些与项目 b 相关的包，这就是`virtualenv`被创建的原因；提供需求隔离级别。这对你很有帮助。

要安装，您可以:

```
> pip install virtualenv
```

接下来，您可以创建一个新的项目目录并创建相关的虚拟环境:

```
> mkdir my_project_dir && cd project_dir  # create the project dir and move there
> virtualenv the_venv  # in case of Python 2
> python3 -m venv the_venv  # in case of Python 3
```

激活虚拟环境，你就准备好了！

```
> source the_venv/bin/activate
```

您下载的任何包都将被存储到那个`the_venv/.../site-packages`目录中。

# Virtualenvwrapper

有几个高级工具可以以简洁的方式管理 python 虚拟环境。据我所知，`Pipenv`和`virtualenvwrapper`是最突出的两个。在这篇文章中，我们将使用`virtualenvwrapper`这个，因为这是我熟悉的一个。它是一个 python 包，将所有目录设置细节从 python 源代码目录中抽象出来，成为一个专用于存储虚拟环境的目录。在这种情况下，我们需要创建专用目录:

```
> mkdir ~/.virtualenvs  # virtual environments are going to be 
# installed into the home folder into a hiffen file
```

作为一个 python 包，它可以通过以下方式轻松安装:

```
> pip install virtualenvwrapper
```

或者

```
> sudo pip install virtualenvwrapper #avoid such sudo ops if possible!
```

接下来，我们需要让终端可以访问这个包的功能。换句话说，每当我们在终端中键入任何包 virtualenvwrapper 子命令时，我们都期望相应的功能发生。这些可以通过在我们的`~/bashrc`或`~zshrc`文件中添加以下行来实现。

```
> export WORKON_HOME=$HOME/.virtualenvs  # define the virtual envs # direcotry
> source /usr/local/bin/virtualenvwrapper.sh  # load virtualenvwrapper # functionality to terminal
```

您可能已经准备好创建您的第一个环境:

```
> makevirtualenv my_env  # that's going to crate a virtual env named 
# my_end --> located at $WORKON
> workon my_env  # to start working into a my_env
> deactivate  # is going to deactivate the particular enabled 
# virtualenv
```

通过运行以下命令

```
> lsvirtualenvs
```

您可以列出可用的虚拟环境。通过检查`~/.virtualenvs/`目录，您可以确保所有列出的 virtualenvs 都在那里。

额外提示:请记住，virtualenvwrapper 能够定义您需要使用的 python 版本。通过键入`mkvirtualenv test -p python2`，只要您的`$PATH`上有 python2，test 将成为 python2 virtualenv。

# 混淆了，常见的陷阱

尽管您可以将所有这三种工具结合到一个高度自动化和高效的环境中，但有些情况下事情可能会真的出错。另外，您将要构建的许多自动化可能在不久的将来会变得过时。整个设置过程中最棘手的一个概念是，有一些工具可以管理用 python 创建的 python 环境！例如，你可以在 python2 中安装并运行 virtualenv 和 virtualenvwrapper，但是要处理 python3 虚拟环境。这很难理解，在我看来，这也是在“完美设置”过程中可能面临的所有这些棘手问题的原因。环境变量内容的任何重新排序或令人讨厌的软件都可能破坏这种美丽的自动化。

# 用集装箱装

这是一个面向生产的解决方案，没有上面提到的所有复杂性。通过为一个特定的项目创建一个专门的 docker 容器，您可以将您需要的任何东西全局地安装到该容器中。不要忘记，尽管这种特定解决方案有三个主要缺点，但总体性能会略有下降，可视化在交互工作方面具有挑战性，docker 开发模式与生产模式不同。

# 最后的想法

四种解决方案我都用过。我认为 Virtualenvweapper 是最适合我的。我确实使用多个 python 版本。为了减轻这种情况，我使用 Virtualevnwrapper 的`-p`选项将每个虚拟环境与特定的 python 版本绑定在一起。我的安装作战计划如下:

*   安装我想保留为主的 python(对我来说这是 python3.5)
*   为特定的 python 版本安装`pip`(在我的例子中，python3-pip)
*   在 python 安装程序上安装`virtualenv`
*   在 python 安装程序上安装`virtualenvwrapper`
*   额外提示:在该设置上安装任何林挺工具，而不是将它们安装在不同的虚拟环境中
*   安装我需要的任何 python 版本，并使用 virtualenvwrapper 的`-p`选项将该特定版本绑定到相关环境。对我来说，这是最简单明了的管道。我不喜欢使用外来的工具来管理 python 版本和虚拟环境，除非我对正在发生的事情非常有信心。