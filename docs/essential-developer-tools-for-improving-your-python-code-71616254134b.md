# 用于改进 Python 代码的基本开发工具

> 原文：<https://medium.com/analytics-vidhya/essential-developer-tools-for-improving-your-python-code-71616254134b?source=collection_archive---------9----------------------->

![](img/5097726cec363c5a4f265c8c608c8e5f.png)

由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [Louis Hansel @shotsoflouis](https://unsplash.com/@louishansel?utm_source=medium&utm_medium=referral) 拍摄的照片

在我多年使用 Python 作为编程语言开发软件的过程中，我越来越喜欢一些开发工具，它们可以帮助我编写更好的代码。特别是当与团队中的其他开发人员一起工作时，这些工具可以用来创建统一的代码库，这些代码库经过了良好的测试，更加安全，对所有合作开发人员来说都是可读的。

在本文中，我将分享我喜欢的开发人员工具，并向您展示如何使用 Git 预提交钩子在每次 Git 提交或推送操作中自动运行这些工具。

## 内容

1.  你为什么需要这些工具？
2.  工具
3.  代码样式和格式化工具(isort、Flake8、Bugbear、Black、Mypy)
4.  软件测试工具(Pytest，coverage.py)
5.  安全工具(土匪，安全)
6.  自动化(使用预提交挂钩)
7.  奖金

# 为什么？

现在你可能会想:为什么我需要所有这些工具？而最重要的答案是:节省时间。手动完成所有这些检查需要很多时间(除非你是完美的程序员)，你可以通过自动化来节省时间。此外，编写错误代码会导致在调试和修复错误上浪费时间。在开始之前，让我们先回顾一下我将讨论的不同类别的工具。

*代码风格和格式* 毫无疑问，任何开发者都可以享受一个可读的、格式良好的、统一的代码库。在 Python 生态系统中，有这样一个概念，即编写代码时充分利用 Python 的习惯用法(也称为编写“Python”代码)，最好符合 Python 的[禅和](https://www.python.org/dev/peps/pep-0020/) [PEP 8 标准](https://pep8.org)。对于任何开发人员来说，主动检查自己或他人代码的一致性都是一个相当乏味的挑战。

*软件测试*
作为一名开发人员，当代码库的软件测试(例如单元测试)最好在每次代码变更时自动运行时，我对任何代码的结果都更有信心。此外，Python 不是一种静态类型的编程语言，这一事实在不主动检查类型和测试失败的情况下，给编写无错误代码带来了一些挑战。

*安全*
开发人员可以得到帮助的第三个领域是安全领域。大量使用开源软件库或组件会使应用程序容易受到使用这些库或组件所带来的任何安全威胁。如果一个开发者被告知在一个依赖的软件包中存在已知的安全漏洞，这不是很酷吗？

所以我们将要讨论的工具集中在上面描述的“代码风格和格式”、“软件测试”或“安全性”类别中的一个。我们开始吧！

# 工具

*项目配置* 当然，需要进行一些配置来调整所有工具并加入或排除一些参数。每个工具都有自己的一套配置参数和配置格式，但是有一种趋势是工具都符合`pyproject.toml`格式(更多信息见 [PEP 518](https://www.python.org/dev/peps/pep-0518/) )。对于下面的每个工具，我将尽可能在`pyproject.toml`部分提到相关的配置参数。为了方便起见，我把最后一个`pyproject.toml`放在了 Github 的仓库里。

**注意:下面例子中的‘app’文件夹指的是我的项目文件夹。*

# 代码样式和格式

对于代码风格和格式，我使用的是 isort、Flake8、Bugbear、Black 和 Mypy。我将在下面详细讨论它们。

## 伊索特

[Github 库](https://github.com/PyCQA/isort)

*它有什么作用？* 它将你的`import`语句从这里排序:

```
from pytz import timezone
from app.const import NAME

from datetime import datetime as dt
```

对此:

```
# Standard library imports
from datetime import datetime as dt

# Third party imports
from pytz import timezone

# Local application imports
from app.const import NAME
```

很漂亮吧？

*配置*

```
[tool.isort]
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
line_length = 88
skip = []
import_heading_stdlib = 'Standard library imports'
import_heading_thirdparty = 'Third party imports'
import_heading_firstparty = 'Local application imports'
import_heading_localfolder = 'Local folder imports'
known_first_party = ['app']
```

*如何安装运行？*

```
pip install isort
isort .
```

## 薄片 8

[Github 资源库](https://github.com/PyCQA/flake8)

它是做什么的？
Flake8 是围绕 [pyflakes](https://github.com/PyCQA/pyflakes) 、 [pycodestyle](https://github.com/PyCQA/pycodestyle) 和 Ned Batchelder 的 [McCabe 脚本](https://github.com/PyCQA/mccabe)的包装器，本质上检查是否符合 [PEP 8](https://pep8.org) 代码风格标准。
如果检测到任何不符合项，工具将输出带有错误代码的警告。以下是它输出的一些错误示例:

```
app/main.py:14:7: E111 indentation is not a multiple of four
app/main.py:14:7: E117 over-indented
app/main.py:15:7: E111 indentation is not a multiple of four
app/const.py:1:20: W292 no newline at end of file
```

注意:flake8 有几个扩展其功能的插件(包括这里提到的一些其他工具，例如“flake8-isort”)，但是我更喜欢将这些工具的执行分开，这样配置和执行更容易。

*配置* Flake8 还不支持`pyproject.toml`配置，所以我们需要在项目根目录下创建一个`.flake8`配置文件:

```
[flake8]
ignore = E203, E266, E501, W503
max-line-length = 88
max-complexity = 18
select = C,E,F,W,B
filename = app/*.py
```

注:`B`选择是针对 Bugbear 错误代码的。

*如何安装运行？*

```
pip install flake8
flake8 <project_folder_name>
```

## 使人过分害怕的东西

[Github 库](https://github.com/PyCQA/flake8-bugbear)

*它有什么作用？Bugbear 是 flake8 的一个插件，包含一些不适合 pyflakes 或 pycodestyle 的错误和警告规则。这方面的一个例子是规则 B009:*

> 不要调用`getattr(x, 'attr')`，而是使用正常的属性访问:`x.attr`。对于不存在的属性，缺省为`getattr`将导致`AttributeError`被引发。如果您事先知道属性名，使用`getattr`没有额外的安全性。

*配置* 只需在 flake8 的选择配置参数中添加`B`即可。

*如何安装运行？*

```
pip install flake8-bugbear
flake8 <project_folder_name>
```

## 黑色

[Github 资源库](https://github.com/psf/black)

它是做什么的？
Black 是一个(牛逼的)坚持 PEP8(尽可能)的固执己见的代码格式化程序。

它变成了这样:

```
def print_hi(name):
 date_time = tz.localize(
     dt.now())
 print(
     f"Hi, {name} it is currently {date_time}"
 )
def   print_hello(
        name
):
  print(
   f"Hello, {name} "
   f"how are you?"
  )
```

变成这样:

```
def print_hi(name):
    date_time = tz.localize(dt.now())
    print(f"Hi, {name} it is currently {date_time}")

def print_hello(name):
    print(f"Hello, {name} " f"how are you?")
```

✨魔法！✨

*配置*

```
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs          # exclude a few common directories in the
  | \.git           # root of the project
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
  | migrations

  # The following are specific to Black, you probably don't want those.
  | blib2to3
  | tests/data
)/
'''
```

*如何安装运行？*

```
pip install black
black <project_folder_name>
```

## Mypy

[Github 库](https://github.com/python/mypy)

它是做什么的？使用 Mypy，你可以将类型 hinst ( [PEP 484](https://www.python.org/dev/peps/pep-0484/) )添加到你的代码中，并检查你的代码中的类型错误。

有趣的事实:Mypy 的创造者 Jukka Lehtosalo 在 Dropbox 与 Python(吉多·范·罗苏姆)的创造者合作开发 Mypy。更多信息[在这里](https://dropbox.tech/application/our-journey-to-type-checking-4-million-lines-of-python)。

这种类型错误的例子有:

```
app/main.py:15: error: Missing return statement
app/main.py:25: error: Argument 1 to "print_hi" has incompatible type "str"; expected "List[str]"
Found 2 errors in 1 file (checked 3 source files)
```

*配置*

Mypy 还不支持`pyproject.toml`，所以我们必须在项目根目录下创建一个`mypy.ini`文件，其中包含:

```
[mypy]
python_version=3.8
platform=linux

files=app/
ignore_missing_imports=true
```

*如何安装运行？*

```
pip install mypy
mypy <project_folder_name>
```

# 软件测试

对于软件测试，我使用 Pytest 和 coverage.py。

## Pytest

[Github 库](https://github.com/pytest-dev/pytest)

*它有什么作用？* Pytest 是一个软件测试 Python 代码的框架。我最喜欢它的一个特性是，它允许你创建可重用的测试代码片段( [fixtures](https://docs.pytest.org/en/stable/fixture.html) ),这些代码可以被组合在一个测试中。

*配置* Pytest 支持`pyproject.toml`配置:

```
[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra"
testpaths = [
    "app",
]
python_files = "*_test.py"
junit_family = "legacy"
```

*注意:任何文件名以* `*_test.py*` *结尾的文件在运行 pytest 时都会被自动测试。*

*如何安装运行？*

```
pip install pytest
pytest <project_folder_name>
```

成功测试的输出:

```
➜ pytest app -ra
===================== test session starts ========================
platform darwin -- Python 3.8.5, ...,  pluggy-0.13.1
rootdir: /, configfile: pyproject.toml, testpaths: app
collected 1 itemapp/main_test.py .                                          [100%]===================== 1 passed in 0.05s ==========================
```

失败测试的输出:

```
pytest app -ra
===================== test session starts ========================
platform darwin -- Python 3.8.5, ..., pluggy-0.13.1
rootdir: /, configfile: pyproject.toml, testpaths: app
collected 1 itemapp/main_test.py F                                                                                                   [100%]=========================== FAILURES ==============================
________________________ test_print_hi ____________________________capsys = <_pytest.capture.CaptureFixture object at 0x1056d5fa0>def test_print_hi(capsys):
        from app.main import print_hi# Happy flow
        name = "Chris"
        print_hi(name)
        captured = capsys.readouterr()
        assert captured.out.startswith(f"Hi, {name} it is currently")# Invalid type
        name = 1
>       print_hi(name)app/main_test.py:16:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _name = 1def print_hi(name: str) -> str:
        if not isinstance(name, str):
>           raise TypeError("name is of invalid type")
E           TypeError: name is of invalid typeapp/main.py:17: TypeError
===================== short test summary info =====================
FAILED app/main_test.py::test_print_hi - TypeError: name is of invalid type
======================= 1 failed in 0.07s =========================
```

## Coverage.py

[Github 资源库](https://github.com/nedbat/coveragepy)

它是做什么的？使用 coverage.py 您可以测量代码执行的百分比，例如在运行 pytest 时。这些度量给你一个指示，你的代码有多少被软件测试覆盖。当然，覆盖率并不能说明软件测试的质量。

*配置*

```
[tool.coverage.run]
branch = true
command_line = '-m pytest'
source = ['app/']

[tool.coverage.report]
precision = 2
skip_covered = true
fail_under = 90
```

*如何安装运行？*

```
pip install coverage[toml]
coverage run
coverage report -m
```

覆盖率报告的输出示例:

```
coverage report -m
Name          Stmts   Miss Branch BrPart     Cover   Missing
------------------------------------------------------------
app/main.py      14      1      6      1    90.00%   24->25, 25
------------------------------------------------------------
TOTAL            28      1      6      1    94.12%3 files skipped due to complete coverage.
```

# 安全性

为了检查安全漏洞，我使用了 Bandit 和 Safety。

## 强盗

[Github 库](https://github.com/PyCQA/bandit)

它是做什么的？
在您的代码上运行 Bandit 可以让您识别已知的安全问题(例如，使用`assert`语句作为检查，在编译成字节码时会被删除)。所有检查的清单可以在这里找到[。](https://bandit.readthedocs.io/en/latest/plugins/index.html#complete-test-plugin-listing)

*配置*

Bandit 还不支持`pyproject.toml`，所以要配置它，必须在项目文件夹根目录下创建一个`.bandit`文件。

```
[bandit]
targets = app/
recursive = true
skips = B101
```

*注意:B101 测试被跳过，因为它(也)触发* `*_test.py*` *文件中使用的断言语句。当断言在测试文件中时，有一个* [*请求*](https://github.com/PyCQA/bandit/issues/346) *忽略 B101。*

*如何安装运行？*

```
pip install bandit
bandit --ini .bandit -r
```

示例输出:

```
bandit --ini .bandit -r
[main] INFO Using ini file for skipped tests
[main] INFO Using ini file for selected targets
[main] INFO profile include tests: None
[main] INFO profile exclude tests: None
[main] INFO cli include tests: None
[main] INFO cli exclude tests: B101
[main] INFO running on Python 3.8.5
Run started:2020-11-29 15:27:27.049118Test results:
 No issues identified.Code scanned:
 Total lines of code: 28
 Total lines skipped (#nosec): 0Run metrics:
 Total issues (by severity):
  Undefined: 0.0
  Low: 0.0
  Medium: 0.0
  High: 0.0
 Total issues (by confidence):
  Undefined: 0.0
  Low: 0.0
  Medium: 0.0
  High: 0.0
Files skipped (0):
```

## 安全

[Github 资源库](https://github.com/pyupio/safety)

它是做什么的？
它扫描你已安装的依赖项，寻找已知的安全漏洞。默认情况下，它使用带有已知漏洞的`safety-db`数据库。

*配置* 运行安全不需要详细配置。

*如何安装运行？*

```
pip install safety
safety check
```

示例输出:

```
safety check
+==================================================================+
| REPORT                                                                       |
| checked 50 packages, using default DB                                        |
+==================================================================+
| No known security vulnerabilities found.                                     |
+==================================================================+
```

# 自动化

为了有效地使用上述工具，我使用 [Git 预挂钩](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)来自动触发使用`pre-commit`的工具。

## 预提交

[Github 库](https://github.com/pre-commit/pre-commit)

*它有什么作用？* 这是一个用于预提交钩子的(多语言)包管理器。在`pre-commit`配置中定义的脚本将在执行`git commit`或`git push`时触发。
预提交允许你配置脚本从本地包运行，或者直接从它们的仓库运行钩子。

*使用本地包* 的配置为了将所有工具集成到`commit`或`push`钩子中，我们将通过创建`.pre-commit-config.yaml`为`pre-commit`构建一个配置文件。

```
repos:
  - repo: local
    hooks:
      - id: isort
        name: isort
        stages: [commit]
        language: system
        entry: isort
        types: [python]
      - id: black
        name: black
        stages: [commit]
        language: system
        entry: black
        types: [python]
      - id: flake8
        name: flake8
        stages: [commit]
        language: system
        entry: flake8
        types: [python]
      - id: mypy
        name: mypy
        stages: [commit]
        language: system
        entry: mypy
        types: [python]
        pass_filenames: false
      - id: bandit
        name: bandit
        stages: [commit]
        language: system
        entry: bandit --ini .bandit -r
        types: [python]
        pass_filenames: false
      - id: pytest
        name: pytest
        stages: [commit]
        language: system
        entry: pytest app -ra
        types: [python]
        pass_filenames: false
      - id: safety
        name: safety
        stages: [commit]
        language: system
        entry: safety check
        types: [python]
        pass_filenames: false
      - id: coverage
        name: coverage
        stages: [push]
        language: system
        entry: coverage run
        types: [python]
        pass_filenames: false
```

*注意:coverage.py 会在每一个* `*git push*` *之前执行，其余的在一个* `*git commit*` *之前执行。*

*使用非本地存储库的配置* 你也可以利用在线存储库来获取预提交钩子。除了 pytest & coverage.py 之外的所有包都提供了可以使用的在线存储库。

```
repos:
-   repo: https://github.com/timothycrosley/isort
    rev: 5.6.4
    hooks:
    -   id: isort
-   repo: https://github.com/psf/black
    rev: 20.8b1
    hooks:
    -   id: black
-   repo: https://gitlab.com/pycqa/flake8
    rev: 3.8.4
    hooks:
    -   id: flake8
        additional_dependencies: [flake8-bugbear]
-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.790
    hooks:
    -   id: mypy
-   repo: https://github.com/Lucas-C/pre-commit-hooks-bandit
    rev: v1.0.4
    hooks:
    -   id: python-bandit-vulnerability-check
-   repo: local
    hooks:
    -   id: pytest
        name: pytest
        stages: [commit]
        language: system
        entry: pytest -v --showlocals
        types: [python]
        pass_filenames: false
    -   id: coverage
        name: coverage
        stages: [push]
        language: system
        entry: coverage run
        types: [python]
        pass_filenames: false
-   repo: https://github.com/Lucas-C/pre-commit-hooks-safety
    rev: v1.1.3
    hooks:
    -   id: python-safety-dependencies-check
```

*如何安装？*

```
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

*注意:第一条命令安装*`*pre-commit*`*；第二个命令安装 git 预提交挂钩&第三个命令安装配置文件中定义的 git 预推送挂钩。*

*怎么跑？* 安装预提交和预推送挂钩后，工具将在每次提交和推送至 git 存储库时被触发。
您可以通过执行以下命令手动触发预提交挂钩(例如用于测试目的):

```
pre-commit run --all-files
```

示例输出:

```
➜ pre-commit run --all-files
isort.........................................................Passed
black.........................................................Passed
flake8........................................................Passed
mypy..........................................................Passed
bandit........................................................Passed
pytest........................................................Passed
safety........................................................Passed
```

当`isort`或`black`在林挺时修改文件时，会出现一个错误:

```
⇣5% ➜ pre-commit run --all-files
isort....................................................................Failed
- hook id: isort
- files were modified by this hookFixing /Users/chris/Code/C01/developer-tools/app/main.pyblack.........................................................Passed
flake8........................................................Passed
mypy..........................................................Passed
bandit........................................................Passed
pytest........................................................Failed
- hook id: pytest
- files were modified by this hook====================== test session starts =========================
platform darwin -- Python 3.8.5, ... -- /bin/python
cachedir: .pytest_cache
rootdir: /, configfile: pyproject.toml, testpaths: app
collected 1 itemapp/main_test.py::test_print_hi PASSED                                   [100%]====================== 1 passed in 0.03s ===========================safety........................................................Passed
```

重新执行钩子将清除失败(在之前的运行中修改了文件)。

*注意:如果您想在提交或推送时跳过运行预提交钩子，您可以通过将* `*--no-verify*` *添加到* `*git commit*` *或* `*git push*` *命令中来实现。*

# 奖金

为了让你快速入门，我已经编写了这个 Python 框架，包括上面描述的所有工具。您可以通过克隆存储库并遵循`README.md`来测试它。祝你检查的愉快！

你已经到达终点了！如果你喜欢这篇文章，我会很感激鼓掌👏。

你知道任何其他的工具会是一个很好的补充或认为这篇文章可以改进吗？也请让我知道！