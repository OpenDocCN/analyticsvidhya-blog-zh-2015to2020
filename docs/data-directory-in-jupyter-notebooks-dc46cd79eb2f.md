# Jupyter 笔记本中的数据目录

> 原文：<https://medium.com/analytics-vidhya/data-directory-in-jupyter-notebooks-dc46cd79eb2f?source=collection_archive---------3----------------------->

在交互式计算中管理对数据文件的访问

![](img/7971687c3974c346b0dd0543e45d449d.png)

由[哈德逊·辛慈](https://unsplash.com/@hudsonhintze?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 摘要

几乎每个笔记本都包含一个`pd.read_csv(file_path)`或类似的命令来加载数据。然而，在笔记本中处理文件路径有点麻烦:移动笔记本成为一个问题，笔记本现在必须知道项目的位置。在这里，我们讨论几种处理这个问题的方法。

# 介绍

启动一个笔记本总是很容易的，你只需要启动几个通常只包含一个`df.head()`的单元格。然而，随着项目的增长(在工业中他们总是这样)，你将需要组织你的文件夹。你需要一个文件夹存放数据，另一个文件夹存放笔记本。随着 EDA 的进展，您将需要更多的文件夹来表示主分析的不同子部分。最重要的是，你的项目应该是[可复制的](https://www.kdnuggets.com/2019/11/reproducibility-replicability-data-science.html)，这样你的同事就可以下载代码，运行脚本，它就会像预期的那样工作，希望产生和你一样的结果:)

所以，如果你的笔记本上有一个`read_csv(relative_path_to_data)`，把它从一个文件夹移到另一个文件夹需要修改代码。这是不可取的，我们希望它的工作无论其位置。您可以通过使用`read_csv(absolute_path_to_data)`来解决这个问题，但是这更糟糕:您将处理比它们需要的更长的路径，并且如果您试图在另一台机器上运行它，您的代码可能会中断。

假设您在`/system_name/project`上有您的工作目录，从那里您运行`jupyter lab`或`jupyter notebook`。数据目录位于`/system_name/project/data`，你们的笔记本在`system_name/project/notebooks`

我们提出两种方法来解决这个问题:

1.  在笔记本中使用环境变量
2.  使用数据模块

# 环境变量

使用这种方法，我们通过使用环境变量来通知系统数据目录的位置。在启动 jupyter 服务器之前，我们将通过执行以下操作来设置变量

`export DATA_DIR=system_name/project/data`

如果您在`/system_name/project`文件夹中，您可以:

`export DATA_DIR=$(pwd)/data`

达到同样的效果。现在，从 bash 终端启动的所有子进程都可以访问这个变量。在您的笔记本中，您现在可以:

现在你的笔记本唯一需要知道的就是数据集的`file_name`。听起来很公平，对吧？

您可以尝试做的另一件事是通过执行以下操作来更改笔记本本身的工作目录:

这是可行的，但我更喜欢前者，因为后者使笔记本工作在一个它不是的目录中，感觉有点可疑:)。

最后，每次启动 jupyter 服务器时设置环境变量可能会有点无聊。您可以使用 [python-dotenv](https://pypi.org/project/python-dotenv/) 来自动化这个过程。它将搜索一个`.env`文件，首先在本地目录中，然后在它的所有父目录中。一旦它这样做了，它将加载那里定义的变量。如果你喜欢这个想法，请查看项目文档！

# 数据模块

我们使用一个环境变量来保存关于项目配置的信息，并将其公开给笔记本。但是把这个责任转移到别的地方呢？我们可以创建一个模块，它的职责是知道数据目录，以及数据集在哪里。我更喜欢这种方法，因为它将使数据集成为代码中的显式符号。

我们需要一个`project_package`文件夹来表示项目的包。在里面，我们将创建一个`project_data.py`模块:

我们使用返回当前文件路径的`__file__` [dunder](https://www.geeksforgeeks.org/__file__-a-special-variable-in-python/) 方法，以及内置的`[Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)` 类来浏览目录。我们通过在`project`文件夹中创建`setup.py`使这个包[可安装](https://python-packaging.readthedocs.io/en/latest/minimal.html):

我们快到了！现在，我们在开发模式下安装刚刚创建的包，这样对包的修改就不需要重新安装了:

`python -m pip install -e .`

这应该会安装`project_package`包，可以从笔记本上访问:

这样，任何环境和位置的任何笔记本电脑都可以使用相同的方法访问数据。如果数据位置改变，我们只需要改变一个位置:模块`project_data.py`。