# Python 项目的模板

> 原文：<https://medium.com/analytics-vidhya/a-template-for-python-projects-3539a5cbeaf1?source=collection_archive---------9----------------------->

![](img/c864dc02e9ab0add59d8762214d51de9.png)

我上周吃的午餐。完全不相关。

我是一个懒惰的人。每次我发现自己做同样的事情超过两次，我就自动执行。一开始需要付出努力，但从长远来看是值得的。开始一个新的 Python 项目就是其中之一，今天我想和你们分享我的蓝图。可以在 github 上找到[完整模板。对于中小型代码库来说，这种设置是一个很好的起点，它可以做一些常见的事情:](https://github.com/gabrieleangeletti/python-package-template)

*   设置开发环境
*   管理依赖关系
*   格式化您的代码
*   运行林挺、静态类型检查和单元测试

在接下来的部分中，我将描述如何在模板中设置这些东西。请注意，还缺少一些东西，我打算接下来添加:

*   部署脚本
*   持续集成管道

# 管理 python 版本— Pyenv

管理 python 的多个版本，或者任何语言的多个版本，都是一种痛苦的经历，原因有很多:无法触及的系统版本、2 对 3 的噩梦、需要不同解释器的两个不同项目等等。 [Pyenv](https://github.com/pyenv/pyenv) 解决了这个问题:它是一个版本管理工具，在很多方面让你的生活更轻松。如果你来自 Javascript /节点世界，这个工具类似于流行的 [n](https://github.com/tj/n) 。

使用`pyenv`,您可以轻松:

*   安装新版本:`pyenv install X.Y.Z`
*   将版本设置为全局:`pyenv global X.Y.Z`
*   通过覆盖`PYENV_VERSION`环境变量来设置当前 shell 的版本
*   通过创建一个`.python-version`文件来设置特定于应用程序的版本

# 管理依赖关系— Pipfile

如果处理版本是痛苦的，那么处理依赖关系就更糟糕了。任何重要的应用程序都依赖于外部包，而外部包又依赖于其他包，确保每个人都获得相同的版本是相当具有挑战性的。在 python 世界中，依赖关系传统上是通过`requirements.txt`文件来管理的。它包含您的应用程序所需的软件包，可选地包含所需的版本。问题是这个文件不处理*递归*依赖，也就是你的应用程序依赖的依赖。Pipfile 是一个新的规范，旨在解决这个问题。它比要求有许多优点。目前最大的一个是*确定性构建*。`Pipfile`和它的合作伙伴`Pipfile.lock`包含了在任何地方安装*相同环境*所需的所有信息。

举个例子吧。考虑以下场景:我们的应用程序`Ninja ducks`依赖于包`ninja`的版本`1.2.3`，而包`ninja`又依赖于另一个名为`requests`的包。

## 示例— requirements.txt

一个`requirements.txt`文件应该是这样的:

运行`pip install -r requirements.txt`时，我们安装`ninja`的`1.2.3`版本，因为需求上是这么说的，还有`requests`的`2.7.9`版本，因为那是当时最新的公开版本。几周后，我们部署了应用程序，但同时`requests`被升级到了`3.0.0`。如果`ninja`使用了`requests`中已经被改变或删除的特性，我们的应用程序将会崩溃。我们可以通过将`requests`添加到需求文件中来解决这个问题，但是你可以自己看到这个解决方案并没有真正的扩展。

## 示例— Pipfile

一个`Pipfile`应该是这样的:

```
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"[packages]
ninja = {version = "==1.2.3"}
```

由此我们可以运行`pipenv lock`来生成一个`Pipfile.lock`:

```
{ "_meta":
  {
    "hash": {
      "sha256": "[long string]"  
    },
    "pipfile-spec": 6,
    "sources": [{
        "name": "pypi",
        "url": "https://pypi.org/simple",
        "verify_ssl": true
      }]
    },
    "default": {
      "ninja": {
        "hashes": [
          "[long string]",
          "[long string]"
        ],
        "version": "==1.2.3"
      },
      "requests": {
        "hashes": [
          "[long string]",
          "[long string]"
        ],
        "version": "==2.7.9"
      }
    }
  }
}
```

如你所见，`requests`是存在的，即使我们在`Pipfile`中没有提到它。这是因为`Pipfile`通过`Pipfile.lock`文件处理递归依赖关系。在部署期间，当我们运行`pipenv install --deploy`来安装依赖项时，将会安装正确版本的`requests`，而不管公共注册表中可用的最新版本。

*   注 1:在上面我使用了几个`pipenv`命令，这是`Pipfile` [规范](https://github.com/pypa/pipfile)的[参考实现](https://pipenv.readthedocs.io)
*   注意 2:您需要将`Pipfile`和`Pipfile.lock`都添加到您的存储库中，否则您将无法恢复相同的环境
*   注 3:如果你目前正在使用`requirements.txt`并想迁移到`Pipfile`，这里有一个[关于如何做的便捷指南](https://pipenv.readthedocs.io/en/latest/basics/#importing-from-requirements-txt)

在[模板](https://github.com/gabrieleangeletti/python-package-template)中，`pyenv`和`pipenv`都可以通过提供的`./setup.sh`脚本进行安装。仅支持 Linux 和 MacOS(部分软件包需要在 Linux 上手动安装)。

# 管理代码——我最喜欢的工具

以下是我在 Python 项目中经常使用的代码质量工具的列表，没有特定的顺序。

## 格式—黑色

根据我最近读的这本书，意志力是一种有限的资源。它就像一块肌肉，你不可能一整天都保持专注，并期望一直保持同样的生产力水平。这就是为什么在编程时，我想把时间花在重要的事情上，而不是缩进、括号等等。所有能自动化的事情都必须自动化。我认为代码格式化至少有两大好处:

*   您将对格式规则的控制权交给了该工具，这意味着您不再考虑它
*   由于每个人都在船上，你不再和你的团队讨论完美的线路长度应该是 42、79 还是 110(或者至少你在开始时只有一个大的讨论)

[Black](https://github.com/psf/black) 称自己是“不折不扣的 Python 代码格式化程序”，它是我最喜欢的格式化工具。使用起来超级简单，只需运行:

```
black {source directory}
```

黑色有很多可配置的选项。我唯一用的是线长 110。如果您查看完整的[项目模板](https://github.com/gabrieleangeletti/python-package-template)，我已经包含了一个方便的`./format_code.sh`脚本，它将在一个命令中格式化您的代码。

## 林挺—薄片 8

林挺是一个相当基本的代码质量检查，有助于防止代码中的简单错误。诸如打字错误、格式错误、未使用的变量等等。对我来说，林挺非常有用，因为:

*   你不必检查次要的细节，因此你节省了时间
*   其他开发人员不必检查次要的细节，因此他们节省了时间

我用`flake8`代表林挺。我特别喜欢的一个特性是忽略特定警告和错误的能力。例如，我使用 110 的线长，这与 [PEP8](https://www.python.org/dev/peps/pep-0008/) 样式指南(推荐 79)相反。通过关闭相应的错误`E501`,我可以安全地使用任意长度的`flake8`。

在[项目模板](https://github.com/gabrieleangeletti/python-package-template)中，你可以用`./test.sh lint`对你的包运行`flake8`。

## 类型检查— Mypy

这是迄今为止我最喜欢的。这个工具为 python 世界带来了静态类型检查。自从发现`mypy`以来，我从未写过一行非类型化的 python。我不打算深入静态类型检查的细节，我只给你看一个从 mypy 的网站[偷来的简单例子:](http://mypy-lang.org/)

标准 python:

```
def fibonacci(n):
  a, b = 0, 1
  while a < n:
    yield a
    a, b = b, a+b
```

类型化 python:

```
def fibonacci(n: int) -> Iterator[int]:
  a, b = 0, 1
  while a < n:
    yield a
    a, b = b, a+b
```

这非常有用，因为:

*   通过查看函数的签名，我更有可能理解函数是做什么的
*   我甚至可以在运行代码之前就发现很多错误
*   我可以检查我是否正确地使用了第三方库

在[项目模板](https://github.com/gabrieleangeletti/python-package-template)中，你可以用`./test.sh type_check`对你的包运行`mypy`。

## 测试— Pytest

Pytest 是 python 最好的测试框架。它给你测试失败原因的详细信息，可以根据名称自动发现你的测试，对[夹具](https://docs.pytest.org/en/latest/fixture.html#fixture)有惊人的支持，还有很多有用的插件。使用`pytest`编写测试超级简单。考虑以下模块`my_module.py`:

```
def my_func(x: int) -> int:
  return x ** 2
```

为了测试这个函数，我们创建了一个名为`my_module_test.py`的模块:

```
from . import my_moduledef test_my_func():
  expected = 9
  actual = my_module.my_func(3)
  assert actual == expected
```

我在`pytest`中使用的主要功能，按随机顺序排列如下:

*   `pytest-cov`:为你的代码生成多种格式的测试覆盖报告的插件
*   `pytest-mock`:为[猴子打补丁](https://stackoverflow.com/questions/5626193/what-is-monkey-patching)增加夹具的插件。用法:

```
def test_my_func(mocker):
  mocker.patch(...)
```

*   `pytest-xdist`:并行运行单元测试。对于大型代码库尤其有用
*   pytest 的标记功能。您可以使用装饰器来标记测试:

```
import pytest@pytest.mark.integration
def test_my_integration_test():
  ...
```

那么您可以只运行标记为`integration`的测试。

在[项目模板](https://github.com/gabrieleangeletti/python-package-template)中，你可以用`./test.sh unit_tests`对你的包运行`pytest`。

## 其他人

其他开发人员。我在项目中使用的工具有:

*   `isort`:自动对导入进行分类，并将其分成多个部分:内部、第一方、第三方等。再说一次，我完全是关于自动化的，这个工具从我的脑海中移除了另一件事
*   `vulture`检查死代码的工具，如未使用的函数、未使用的常数等。保持房子整洁是件好事，尤其是如果你知道破窗理论

请在下面的评论中告诉我你的观点。完整的代码可以在[这里](https://github.com/gabrieleangeletti/python-package-template)找到(说明在自述文件中)。

*原载于*[*https://gabrieleangeletti . github . io*](https://gabrieleangeletti.github.io/blog/python-project-template/)*。*