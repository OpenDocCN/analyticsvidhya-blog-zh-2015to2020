# 诗歌:最终成为管理 Python 包的一体化工具

> 原文：<https://medium.com/analytics-vidhya/poetry-finally-an-all-in-one-tool-to-manage-python-packages-3c4d2538e828?source=collection_archive---------0----------------------->

![](img/a768091cc20e0065a962961f9e6e4d8d.png)

丹·艾丁格。尼德缪勒

编辑:从诗歌 1.2 开始，关于如何安装包有了一些变化，我在这里[解释了一下](https://lewoudar.medium.com/what-is-new-in-poetry-1-2-67af090f1cb4)。至于其他方面，本教程仍然有效。

如果您是 python 编程的新手，有时在管理包依赖关系和在 pypi 或另一种形式的包注册表上发布项目时，您会感到不知所措。涉及到生态系统的工具太多了: [virtualenv](https://virtualenv.pypa.io/en/latest/) ，pip， [pipenv](https://pipenv.kennethreitz.org/en/latest/) ，[调情](https://flit.readthedocs.io/en/latest/)，[麻线](https://twine.readthedocs.io/en/latest/)等等..这有点令人沮丧，因为您不知道从什么开始，并且您经常不得不在几个工具之间周旋以达到您的目的。

幸运的是，一个新的工具已经到来并解决了所有这些问题，它叫做[诗歌](https://python-poetry.org/)！本教程将基于 2019 年[12 月](https://github.com/python-poetry/poetry/releases/tag/1.0.0)发布的诗歌 1.0.0。

# 装置

在 windows 上，您需要 powershell 来安装它:

```
> (Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python
```

之后，您将需要重启 shell 以使其运行。

在 linux 和其他 posix 系统上(包括 mac):

```
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

要检查其安装是否正确，您可以检查版本:

```
> poetry --version
Poetry version 1.0.0
```

在这里，我们看到我们安装了 1.0.0 版。可能是最新的稳定版本没有安装安装脚本，要更新诗，可以运行:

```
> poetry self update
```

请注意，您可以使用传统的 **pip** 命令安装 poem，但是 poem 将被限制为为已安装它的 python 版本创建虚拟环境。

# 初始化项目

为了研究诗歌，我们将使用 *new* 命令创建一个简单的项目 *poet* 。

```
> poetry new poet
```

创建的项目的树形结构如下所示。

```
├── pyproject.toml
├── README.rst
├── poet
│   └── __init__.py
└── tests
    ├── __init__.py
    └── test_poet.py
```

这里最重要的文件是 *pyproject.toml* ，它包含了管理我们的包所需的所有信息。一开始看起来是这样的:

```
[tool.poetry]
name = "poet"
version = "0.1.0"
description = ""
authors = ["lewoudar <XXX@XXX.com>"]

[tool.poetry.dependencies]
python = "^3.8"

[tool.poetry.dev-dependencies]
pytest = "^5.2"

[build-system]
requires = ["poetry>=0.12"]
build-backend = "poetry.masonry.api"
```

这里我们定义了四个部分:

*   第一部分定义了与依赖包无关的所有元数据，如名称、版本等..
*   第二部分定义了项目所需的所有依赖项
*   第三部分定义了所有开发依赖项，这些依赖项不是构建项目所必需的，而是执行测试、构建、文档等其他操作所必需的..
*   第四部分定义了一个构建系统，如 [PEP 517](https://www.python.org/dev/peps/pep-0517/) 所示。

如果你不喜欢诗歌为你初始化一个项目，或者如果你已经有一个你想用诗歌控制的项目，你可以使用 [*init*](https://python-poetry.org/docs/cli/#init) 命令。您将获得一个交互式 shell 来配置您的项目。

```
> poetry init
```

# 添加包

好的，让我们假设我们想要使用[请求](https://2.python-requests.org/en/master/)来执行一些 api 查询，使用诗歌，添加一个依赖是简单的。

```
> poetry add requests
```

如果我们想安装一个开发依赖项，即不像 pytest 那样与您的项目直接相关，我们可以通过传递 *-D* 选项来实现。

```
> poetry add -D pytest
```

名词（noun 的缩写）b:如果您之前使用了 *new* 命令，并像上面一样添加请求包，那么您可能已经安装了 pytest，因此您将得到一个值错误，表明该包已经存在。所以不要惊讶:)

# 列出依赖关系

如果我们想在某个时候发现所有已安装的依赖项，我们可以使用 *show* 命令。

```
> poetry show --tree
pytest 5.3.2 pytest: simple powerful testing with Python
|-- atomicwrites >=1.0
|-- attrs >=17.4.0
|-- colorama *
|-- more-itertools >=4.0.0
|-- packaging *
|   |-- pyparsing >=2.0.2
|   `-- six *
|-- pluggy >=0.12,<1.0
|-- py >=1.5.0
`-- wcwidth *
requests 2.22.0 Python HTTP for Humans.
|-- certifi >=2017.4.17
|-- chardet >=3.0.2,<3.1.0
|-- idna >=2.5,<2.9
`-- urllib3 >=1.21.1,<1.25.0 || >1.25.0,<1.25.1 || >1.25.1,<1.26
```

前面的命令画出了我们所有依赖项以及依赖项的依赖项的图形..

如果我们不确定我们是否有一个依赖项的最新版本，我们可以通过使用““*latest”*选项*告诉 poems 检查我们的包存储库是否有新版本。*我们会得到列出了当前版本和最新发布版本的包。

```
> poetry show --latest
atomicwrites   1.3.0      1.3.0      Atomic file writes.
attrs          19.3.0     19.3.0     Classes Without Boilerplate
certifi        2019.11.28 2019.11.28 Python package for providing Mozilla's CA Bundle.
chardet        3.0.4      3.0.4      Universal encoding detector for Python 2 and 3
colorama       0.4.3      0.4.3      Cross-platform colored terminal text.
idna           2.8        2.8        Internationalized Domain Names 
...
```

# 安装项目

如果我们想在可编辑模式下安装项目，我们可以使用*安装*命令。

```
> poetry install
```

请注意，如果我们有额外的依赖项，它们将不会随前面的命令一起安装。pyproject.toml 文件中的额外依赖项如下所示(摘自官方文档):

```
[tool.poetry.dependencies]
# These packages are mandatory and form the core of this package’s distribution.
mandatory = "^1.0"

# A list of all of the optional dependencies, some of which are included in the
# below `extras`. They can be opted into by apps.
psycopg2 = { version = "^2.7", optional = true }
mysqlclient = { version = "^1.3", optional = true }

[tool.poetry.extras]
mysql = ["mysqlclient"]
pgsql = ["psycopg2"]
```

因此，如果我们想安装项目和 postgresql 客户端，我们应该在终端中输入以下内容:

```
> poetry install -E pgsql
```

如果我们想要所有的数据库客户端:

```
> poetry install -E mysql -E pgsql
```

# 处理虚拟环境

当我们使用诗歌时，虚拟环境是在我们系统的某个地方自动创建的。确切的位置取决于我们的操作系统，但是我们可以通过运行 *env* 命令和查找路径信息来找到它。

```
> poetry env info
Virtualenv
Python:         3.8.0
Implementation: CPython
Path:           C:\Users\rolla\AppData\Local\pypoetry\Cache\virtualenvs\poet-6mnhVs-j-py3.8
Valid:          TrueSystem
Platform: win32
OS:       nt
Python:   C:\Users\rolla\AppData\Local\Programs\Python\Python38-32
```

事实上，我们可以在前面的命令中添加““*path”*选项，以便只获取虚拟环境的路径；)

```
> poetry env info --path
C:\Users\rolla\AppData\Local\pypoetry\Cache\virtualenvs\poet-6mnhVs-j-py3.8
```

如果我们想运行一个由我们的依赖项安装的脚本，我们需要一种方法来访问我们的虚拟环境，否则我们的操作系统会抱怨它不知道命令。为此，poem 提供了一个 *run* 命令来解决这个问题。它将在虚拟环境中自动运行传递给它的参数。

```
> poetry run pytest
===================================================================================== test session starts =====================================================================================
platform win32 -- Python 3.8.0, pytest-5.3.2, py-1.8.1, pluggy-0.13.1
rootdir: D:\projets\python\tutorials\poet
collected 1 itemtests\test_poet.py .                                                                                                                                                                     [100%]
..
```

例如，我们也可以通过运行 pip 命令在可编辑模式下卸载项目。

```
> poetry run pip uninstall poet
```

如果我们发现总是在我们想要使用的所有脚本命令前面添加一个*poem run*很烦人，poem 提供了一个 *shell* 命令，它可以直接在虚拟环境中生成一个新的 shell。这样我们现在就可以调用 *pytest* 命令，而不用在它前面运行*诗歌*。

```
> poetry shell
> pytest
===================================================================================== test session starts =====================================================================================
platform win32 -- Python 3.8.0, pytest-5.3.2, py-1.8.1, pluggy-0.13.1
...
```

要离开这个 shell，我们可以运行 *exit。*

如果你想管理多个 python 解释器，可以看看这个[文档](https://python-poetry.org/docs/managing-environments/)。

# 更新包

为了更新所有的依赖关系，我们可以运行 *update* 命令。

```
> poetry update
```

如果我们只想更新一些包，我们可以将它们指定为 *update* 命令的参数。

```
> poetry update package1 package2
```

# 取出包裹

由于有了 *remove* 命令，删除一个包也很简单。

```
> poetry remove requests
```

如果是开发包，我们必须将-D 选项传递给命令。

```
> poetry remove -D pytest
```

# 构建最终的包

## 向 pyproject.toml 添加一些有用的元数据

在构建项目之前，我们需要添加一些有用的元数据来分发和解释如何使用我们的包，如许可证、一些关键字和我们希望在最终构建中包含的包。我们的 pyproject 现在看起来像这样:

```
[tool.poetry]
name = "poet"
version = "0.1.0"
description = ""
authors = ["lewoudar <XXX@XXX.com>"]
license  = "Apache-2"
keywords = ["poet", "requests"]
readme = "README.rst"

packages = [
    { include = "poet" }
]

[tool.poetry.dependencies]
python = "^3.8"
requests = "^2.22.0"

[tool.poetry.dev-dependencies]
pytest = "^5.2"

[build-system]
requires = ["poetry>=0.12"]
build-backend = "poetry.masonry.api"
```

如果您想了解更多关于传递给 pyproject.toml 的不同元数据的信息，请看这个[文档](https://python-poetry.org/docs/pyproject/)。

## 建设

为了构建一个包，我们运行..*建立*司令部！

```
> poetry build
Building poet (0.1.0)
 - Building sdist
 - Built poet-0.1.0.tar.gz- Building wheel
 - Built poet-0.1.0-py3-none-any.whl
```

如果我们想将构建限制到特定的类型，我们可以只使用-F 选项。

```
> poetry build -F wheel # or sdist
```

# 部署包

## 配置存储库和凭据

如果我们没有在传统的 [pypi](https://pypi.org/) 上部署，我们应该配置我们想要上传包的仓库。为此目的创建了*配置*命令。

```
> poetry config repositories.priv https://private.repository
```

在上面的例子中，我们用 URL“https://private . repository”配置了名为“priv”的存储库。

如果我们想要存储这个私有存储库的凭证，我们也可以使用 *config* 命令。

```
poetry config http-basic.priv username password
```

在上面的命令中，“用户名”和“密码”对应于我们的..登录我们的私有存储库的用户名和密码。请注意，如果我们不指定密码，当我们部署包时，系统会提示我们输入密码。

对于 pypi，我们也可以使用前面的命令来配置凭证，并将“priv”替换为“pypi ”,但是现在建议使用 [API 令牌](https://pypi.org/help/#apitoken)来通过 pypi 进行身份验证。我们可以这样配置它:

```
poetry config pypi-token.pypi my-token
```

“my-token”代表我们在 pypi 上创建的令牌。

注意，我们也可以使用[环境变量](https://python-poetry.org/docs/repositories/#configuring-credentials)来配置我们的凭证。这在使用 CI 工具时特别有用。

## 部署

当我们完成配置后，我们可以使用 *publish* 命令简单地部署我们的包。

```
poetry publish
```

如果我们使用私有存储库，我们必须指定我们之前通过-r 选项配置的名称。

```
poetry publish -r priv
```

请注意，我们可以使用“— build”选项在部署之前构建包。

# c 扩展

目前对 C 扩展的支持还不能完全运行。你可以通过阅读这期[文章](https://github.com/python-poetry/poetry/issues/11)得到一个草案解决方案，但是我会建议你继续使用像 pip 或 pipenv 这样的标准工具，除非 poem 完全支持它。

本教程到此为止。希望你喜欢并学到很多:)