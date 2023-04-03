# 使用诗歌的 Python 打包和依赖性管理

> 原文：<https://medium.com/analytics-vidhya/python-packaging-and-dependency-management-using-poetry-b7a7b3dd8e60?source=collection_archive---------5----------------------->

# 介绍

依赖关系管理是一种以自动化方式声明、解析和使用项目所需依赖关系的技术。有几种不同的方法来处理 Python 中的依赖关系。在本文中，我将向您展示如何使用**诗歌**来管理项目的依赖关系。**poems**是 Python 中进行依赖管理和打包的工具。它允许您声明项目的依赖项，并为您管理它们。

# 先决条件

**诗词**要求 [Python](https://www.python.org/downloads/) 2.7 或 3.4+。它是多平台的，目标是让它在不同的操作系统上都能很好地工作。

# 装置

**poems**提供了一个定制的安装程序，它将通过出售诗歌的依赖项来安装与系统其余部分隔离的诗歌。这是推荐的装诗方式。或者，您可以使用 ***pip install*** 命令将它安装在您现有的 python 之上。

## 在 Linux/macOS 上

```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

## 在 windows 上(PowerShell)

```
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python
```

安装程序将`poetry`工具安装到 poems 的`bin`目录中。在 Unix 上，它位于`$HOME/.poetry/bin`，在 Windows 上，它位于`%USERPROFILE%\.poetry\bin`，在任一操作系统上，您都应该看到以下输出:

```
Retrieving Poetry metadata# Welcome to Poetry!This will download and install the latest version of Poetry,
a dependency and package manager for Python.It will add the `poetry` command to Poetry’s bin directory, located at:$HOME/.poetry/binThis path will then be added to your `PATH` environment variable by
modifying the profile file located at:$HOME/.profileYou can uninstall at any time by executing this script with the — uninstall option,
and these changes will be reverted.Installing version: 1.0.5
 — Downloading poetry-1.0.5-linux.tar.gz (23.00MB)Poetry (1.0.5) is installed now. Great!To get started you need Poetry's bin directory ($HOME/.poetry/bin) in your `PATH`
environment variable. Next time you log in this will be done
automatically.To configure your current shell run `source $HOME/.poetry/env`
```

使用`poetry --version`查看安装是否成功。您应该会看到类似于`Poetry 1.0.5`的结果。

# 使用

poem 帮助您声明、管理和安装 Python 项目的依赖项，确保您在任何地方都有正确的堆栈。

## 安装依赖项并创建虚拟环境

poems 只使用一个文件(`pyproject.toml`)来管理你的项目依赖关系。换句话说，诗用`pyproject.toml`代替`setup.py`、`requirements.txt`、`setup.cfg`、`MANIFEST.in`、`Pipfile`。

```
[*tool*.*poetry*]name = "test-package"version = "0.1.0"description = "The description of the package goes here"license = "MIT"authors = ["MehrdadEP <mehrdadep@outlook.com>"]readme = 'README.md'  # Markdown files are supportedrepository = "https://github.com/python-poetry/poetry"homepage = "https://github.com/python-poetry/poetry"keywords = ['packaging', 'poetry'] [*tool*.*poetry*.*dependencies*]python = "~2.7 || ^3.2"  # Compatible python versions must be declared here# Dependencies with extrasrequests = { version = "^2.13", extras = [ "security" ] }# Python specific dependencies with prereleases allowedpathlib2 = { version = "^2.2", python = "~2.7", allow-prereleases = true }# Git dependenciescleo = { git = "https://github.com/sdispater/cleo.git", branch = "master" } # Optional dependencies (extras)pendulum = { version = "^1.4", optional = true } [*tool*.*poetry*.*dev-dependencies*]pytest = "^3.0"pytest-cov = "^2.4"
```

第一次使用 ***install*** 命令创建虚拟环境并安装所有依赖项

```
poetry install
```

## 其他有用的命令

1.  使用 ***add*** 命令向现有环境添加一个新的依赖项

```
poetry add pendulum
```

2.使用 ***build*** 命令从您的项目中构建一个包

```
poetry build
```

3.使用 ***发布*** 命令将您的项目发布到 PyPI

```
poetry publish
```

4.使用 ***show*** 命令跟踪项目的依赖项并洞察它们

```
poetry show --tree
poetry show --latest
```

5.使用 ***新建*** 命令创建一个带有诗歌的 Python 项目(该命令创建一个空项目和所需文件)

```
poetry new test-project
```

6.使用项目根目录中的 ***shell*** 命令激活一个现有的虚拟环境

```
poetry shell
```

# 卸载诗歌

如果您认为诗歌不适合您，您可以通过使用`--uninstall`选项再次运行安装程序，或者在执行安装程序之前设置`POETRY_UNINSTALL`环境变量，将它从您的系统中完全删除。

```
python get-poetry.py --uninstall
POETRY_UNINSTALL=1 python get-poetry.py
```