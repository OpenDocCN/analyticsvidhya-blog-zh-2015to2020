# 林挺 Python 和 Travis 一起编码更好

> 原文：<https://medium.com/analytics-vidhya/linting-python-codes-better-with-travis-ab166034400?source=collection_archive---------15----------------------->

![](img/3432b5ab7c845e6d3e7f5a7600007204.png)

在为开源组织做贡献时，人们会注意到需要遵循编码标准和惯例来审查和合并您的拉式请求。成熟的公司和许多初创公司都是如此:他们都喜欢遵循自己选择的编程语言的编码标准。这个过程就是所谓的 T2 林挺 T3。当组织放弃他们的代码时，每个人都很高兴，因为编码标准被遵循了。
但是我们能在我们的私人账户里有同样的设置吗？可以像这些组织一样运作。答案是肯定的，我们可以，如果你留下来，我会带你去看。

在本教程结束时，我们将:

*   了解林挺是什么
*   设置 TravisCI 并将其集成到我们的存储库中
*   使用 Travis CI 通过 flake8 和 pylint 来 lint Python 代码

# 什么是林挺

林挺是指分析源代码以标记编程错误和风格错误的过程。如果操作得当，林挺可以检测程序中的格式差异和错误，并确保您遵守编码标准和惯例。Linters(林挺的一个工具)确保你遵循最佳实践，使你的代码可读，易于维护，并可能节省时间。
组织使用 CI 工具来自动化他们的林挺，并确保未来代码库的贡献者遵循 lint 规则。

每种语言都使用自己的一套行，所以我在下面只列出了三种编程语言的三个 linters:

*   Python: pylint，flake8，bandit
*   go lang:Go-critical、go-vet、golangci-lint(包含 Go 语言中的所有 linters)
*   JavaScript: ESLint，JSLint。

对于本教程的重点，我们将集中在林挺 python 代码。

# 设置 TravisCI 并将 TravisCI 集成到您的存储库中

在上一节中，我们讨论了组织使用 CI 工具来自动化他们的林挺。但是 CI 是什么呢？

持续集成是一种软件开发实践，开发人员定期将他们的代码变更合并到一个中央存储库中，然后运行自动化构建和测试。
持续集成通常是指软件发布过程的构建或集成阶段，需要自动化组件(如 CI 或构建服务)和文化组件(如经常学习集成)。
Travis CI(拜托，顾名思义)是组织用来自动测试 lint 的工具之一。许多开源组织使用 Travis 来实现自动化，我们将模仿他们来自动检查 lint。

Travis CI 在他们的文档中有所有需要设置的步骤，因此在这里输入这些步骤是多余的。请点击查看他们的文档。完成后，Travis 就设置好了，并与您的帐户关联。
要将 Travis 与您的存储库集成，您需要在项目级目录中添加一个 **.travis.yml** 文件来触发 Travis CI 构建。

# 林挺 Python 代码与特拉维斯 CI

现在我们前面提到了 Python 中的 linters，所以我们从中挑选两个来使用:flake8 和 pylint。我们将检查包中的以下代码。

```
import json
class Dog:def __init__(self, name, age):
        """Initialize name and age attributes."""
        self.dog_name = name
        self.dog_age = age
        self.name = "Rambo"
        self.age = 12def sit(self):
        print(f"{self.dog_name} is now sitting")def roll_over(self):
        print(f"{self.dog_age} rolled over!")my_dog = Dog("Maxwell", 6)
```

将上面的代码保存在一个名为`app.py`的新文件中，然后我们用下面的内容创建一个`requirements.txt`文件

```
pytest-pylint==0.16.1
pytest-flake8==1.0.5
```

我们选择这些包是因为 Travis CI 需要运行一个测试，以及使用哪个包更好。
Pytest 用于运行测试，将在后续文章中详细介绍。

> ***注意*** *:以下环节需单独完成。首先使用 flake8 包，观察结果，修复后，将代码返回到上面的格式，并用 pylint 进行测试。
> 这样做是为了让你看到两个包装之间的差异，并选择你最喜欢的一个。*

# 薄片 8

Flake8 是一个用于实施样式指南的工具。这意味着 flake8 的主要目的是确保你的代码风格正确。
是“验证 pep8、pyflakes 和循环复杂度的包装器”。要在 Travis 中设置它，按照上面的说明创建一个 Travis YAML 文件，并向其中添加以下代码。

```
language: python
python: "3.8"install:
    - pip install -r requirements.txt
scripts:
    - pytest --flake8
```

*   语言:这告诉 Travis 使用 python 作为这次构建的编程语言。
*   python:在这里你选择你想要测试代码的版本，它支持从 2.7 到现在的所有 python 版本，但是我们用 python 3.8 版本测试我们的代码。
*   安装:我们正在安装我们已经在`requirements.txt`文件中定义的包。我们希望将包安装在 Travis 环境中，以便可以访问它。
*   脚本:这里，我们正在执行我们安装的`pytest --flake8`包。这个包裹集 pytest 和 flake8 于一身。这意味着它将针对 flake8 测试代码中的所有`.py`文件。

# 皮林特

Pylint 与 flake8 有点不同，它实际上是完整的:它检查样式指南、编程错误，并遵循最佳实践。有些人甚至说 pylint 的规则太严格了(这是真的，你会看到的)，所以有些人用`black`代替 pylint。
我们要做和上面一样的设置
编辑上面的文件`.travis.yml`来使用 pylint，

```
language: python
python: "3.8"install:
    - pip install -r requirements.txt
scripts:
    - pytest --pylint
```

它遵循与上面相同的解释，这次我们执行 pylint 来检查，而不是 flake8。

# 结果

当我们将上述代码提交到存储库时，Travis 构建了测试 flake8 或 pylint 代码所需的环境。
如果您使用 flake8 包，您应该会得到以下错误:

如果您使用 pylint 包，您应该会得到以下错误:

现在 pylint 有一个烦人的`C103: Constant name "my_dog" doesn't conform to UPPER_CASE naming style (invalid-name)`规则。
我不喜欢这样，所以我们会选择忽略那个特定的错误，我们通过...

因此，通过 flake8 和 pylint 的最终代码如下:

```
"""This module is an attempt to practise class in python."""class Dog:
    """A simple attempt to model a dog."""def __init__(self, name, age):
        """Initialize name and age attributes."""
        self.dog_name = name
        self.dog_age = age
        self.name = "Rambo"
        self.age = 12def sit(self):
        """Simulate a dog sitting in response to a command."""
        print(f"{self.dog_name} is now sitting")def roll_over(self):
        """Simulate a dog sitting in response to a command."""
        print(f"{self.dog_age} rolled over!")my_dog = Dog("Maxwell", 6)
```

现在，如果我们将代码推送到 git 存储库，测试就成功通过了。如果您检查与您的 git 帐户相关的电子邮件，您应该会看到一条祝贺消息，说您已经修复了构建。

# 结论

我们已经了解了什么是林挺，如何将 Travis CI 添加到我们的 git 环境中，并使用 Python 中的 linters 根据 Python 规则测试了我们的代码。
您应该知道，您还可以使用 Travis 来测试您的代码是否做了它应该做的事情。这被称为测试驱动开发，我们将在即将到来的 Pytest 系列中对此进行大量讨论。

在那之前，享受自动化的力量吧！

最初发布于[发展至](https://dev.to/edeediong/using-travisci-to-write-better-python-codes-27kg)。