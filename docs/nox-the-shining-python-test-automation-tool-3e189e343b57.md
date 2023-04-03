# Nox:闪亮的 python 测试自动化工具

> 原文：<https://medium.com/analytics-vidhya/nox-the-shining-python-test-automation-tool-3e189e343b57?source=collection_archive---------3----------------------->

![](img/ef706381fcce064bf218482f709141a4.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Franck V.](https://unsplash.com/@franckinjapan?utm_source=medium&utm_medium=referral) 拍摄的照片

上周我发现了 [nox](https://nox.thea.codes/en/stable/index.html) 这是一个命令行工具，可以在多种 python 环境中自动测试，类似于 [tox](https://tox.readthedocs.org/) 。它被他的创造者认为像托克斯的*后代*。

对我来说，它相对于 tox 的主要优势是，您可以在自动化过程中利用 python 语言的所有功能，但它还有其他很酷的特性。它也被一些伟大的项目所使用，如 [google-cloud-python](https://github.com/googlecloudplatform/google-cloud-python) 或 [urllib3](https://github.com/urllib3/urllib3) 。

# 使用

## 基础

对于这个例子，我们将想象一个简单的项目，它有一个模块 example.py，其中有一个函数将两个数相加。

```
def addition(a, b):
    return a + b
```

为了测试这个文件，我们还将创建一个 test_example.py 文件。

```
import pytest

from .example import addition

@pytest.mark.parametrize(('a', 'b', 'c'), [
    (0, 0, 0),
    (1, 2, 3)
])
def test_addition_returns_correct_result(a, b, c):
    assert c == addition(a, b)
```

现在我们将创建一个 noxfile.py 来测试这个文件。

```
import nox

@nox.session
def lint(session):
    session.install('flake8')
    session.run('flake8', 'example.py')

@nox.session
def tests(session):
    session.install('pytest')
    session.run('pytest')
```

Nox 与**会话**的概念一起工作。session 是一个用 **nox 修饰的可调用函数。session** 接受一个 *session* 参数，我们用它来执行各种操作。您通常会执行的两个主要动作是**安装**和**运行**，但还有[其他](https://nox.thea.codes/en/stable/config.html#module-nox.sessions)。

在前面的例子中，我们定义了两个会话，一个用于检查模块 example.py 是否符合 python 编写标准，以及优秀的 [flake8](http://flake8.pycqa.org/en/latest/) 。Nox 将按照我们第一个命令***session . install(‘flake8’)***的要求，在一个 ***默认创建的新虚拟环境中安装 flake 8 包。nox 项目顶部的*** 文件夹，但是你可以改变它的。注意，这个*安装*命令将把它所有的参数传递给 ***pip 安装*** 命令。第二个命令使用参数 example.py 运行 flake8。还要注意，如果您想要运行类似于 ***foo -x bar*** 的命令，您应该将命令的每个字符串像参数一样传递给 session.run 命令，即 ***session.run('foo '，'-x '，')*** 。

第二个会话使用 pytest 库检查我们的脚本是否按照我们期望的方式工作。

要列出所有可用的会话，您可以运行以下命令。

```
nox -l
Sessions defined in ..* lint
* tests
```

如果您想运行所有会话，只需运行 *nox* ，您将得到如下输出:

```
nox > Running session lint
nox > Creating virtual environment ..
...nox > Ran multiple sessions:
nox > * lint: success
nox > * tests: success
```

## 针对多个版本的 python 进行测试

您可以指定多个 python 版本进行测试，或者使用 session decorator 指定一个特定的版本。

单一版本

```
@nox.session(python="3.8")
def test(session):
    ...
```

多个版本

```
@nox.session(python=["3.6", "3.7", "3.8"])
def tests(session):
    ...
```

如果你再次检查会话列表，你会看到它变得更加丰富。

```
nox -l
..
* lint
* tests-3.6
* tests-3.7
* tests-3.8sessions marked with * are selected, sessions marked with - are skipped.
```

请注意，您需要在计算机上安装指定的 python 版本，以便会话能够正确运行。此外，如果您不想运行所有会话，您可以通过“-s”选项指定您想要的会话。要检查您是否将运行您想要的会话，您还可以组合“-l”选项。例如，在前面的例子中，如果我想运行 lint 和 tests-3.7 会话，我将首先使用命令:

```
nox -l -s lint tests-3.7
..* lint
- tests-3.6
* tests-3.7
- tests-3.8sessions marked with * are selected, sessions marked with - are skipped.
```

请注意，所选会话前面有一个**星号**，那些不运行的会话前面有一个**减号**。

您还可以使用环境变量 NOXSESSION 来实现相同的目标。

```
export NOXSESSION=lint,tests-3.7
# you can use set command on Windows
nox -l
...
* lint
- tests-3.6
* tests-3.7
- tests-3.8sessions marked with * are selected, sessions marked with - are skipped.
```

## 参数化

nox 的另一个很酷的特性是参数化。例如，您可以轻松地创建一个适合您需求的测试矩阵。在这里，我们创建了一个由四个会话组成的矩阵，用两个不同的数据库测试 django 的两个版本:

```
@nox.session
@nox.parametrize('django', ['1.9', '2.0'])
@nox.parametrize('database', ['postgres', 'mysql'])
def tests(session, django, database):
    ...
```

这将创建以下会话:

```
nox -l
...
* tests(database='postgres', django='1.9')
* tests(database='mysql', django='1.9')
* tests(database='postgres', django='2.0')
* tests(database='mysql', django='2.0')
```

## 改变 nox 文件中的 NOx 行为

你可以改变 noxfile 中的一些行为，比如将要创建虚拟环境的文件夹，或者重用现有的虚拟环境，而不是系统地创建一个新的。

```
nox.options.envdir = ".cache"
nox.options.reuse_existing_virtualenvs = True
```

请注意，在命令行上传递的参数优先于在 noxfile 中定义的参数。有关可用于更改行为的选项的完整列表，请参考此[文档](https://nox.thea.codes/en/stable/config.html#modifying-nox-s-behavior-in-the-noxfile)。

## 将 tox 文件转换为 nox 文件

如果您正在使用 tox，并且希望转换到 nox，后者提供了一个简单的脚本来从您的 tox 文件创建一个 noxfile。您需要额外依赖地安装 nox。

```
pip install --upgrade nox[tox_to_nox]
```

然后，您只需要在包含您的 tox 文件的目录中运行脚本 tox_to_nox。

```
tox_to_nox
```

注意，您可能需要检查生成的 noxfile 并手工修复它，因为到目前为止对 tox 转换的支持还很少。

这就是这篇教程的全部内容，希望你喜欢。如果你想了解更多关于 nox 的知识，我邀请你查阅[官方文件](https://nox.thea.codes/en/stable/index.html)。

这是我的第一个教程，所以如果你对如何改进它有任何意见，我很乐意听到他们:)