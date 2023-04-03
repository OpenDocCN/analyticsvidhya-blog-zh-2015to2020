# 使用诗歌的最佳实践

> 原文：<https://medium.com/analytics-vidhya/best-practice-for-using-poetry-608ab6feaaf?source=collection_archive---------2----------------------->

对于 python 库开发，强烈推荐使用`[poetry](https://python-poetry.org/)`,因为它为依赖性管理、构建和分发提供了一站式解决方案

最近迷上了它，真的希望它能成为默认的 python 打包和依赖管理系统。

关于基本的诗歌设置，请参考[诗歌文档](https://poetry.eustace.io/docs/pyproject/)。

```
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
```

今天我想谈谈我是如何解决诗歌的一些问题的。

# 兼容其他传统项目。

## 可编辑模式

很多人问的一个问题是如何在 pip 中保持可编辑模式:

```
pip install -e /path/to/your/project
```

这样，您的代码更改将立即反映出来，而无需重新打包。

[这个问题](https://github.com/python-poetry/poetry/issues/34#issuecomment-378673938)说明没有原生支持这样做。

根据响应，我使用下面的语句来生成设置:

```
export ARTIFACT=$(poetry version | sed "s/ /-/g") && tar xvOf "dist/$ARTIFACT.tar.gz" "$ARTIFACT/setup.py" > setup.py
```

之后，你可以在其他任何地方做同样的`pip install -e /path/to/your/project`。但是请记住，如果您对 pyproject.toml/poetry.lock,做了任何更改，您需要再次重新生成 setup.py。这包括依赖项、入口点等。

注意，如果你是在诗歌项目本身开发，就不需要这个。`poetry install`已经为你但当前项目处于可编辑模式。

## 要求

同样，有时无论出于什么原因，人们都需要使用 requirements.txt。

```
poetry export --without-hashes -f requirements.txt -o requirements.txt
```

有了 hashes，它是确定性的，所以如果你的库做了一些讨厌的事情，你不会有问题，它几乎就像你的 poetry.lock。但是它的可读性较差。

# 使用 direnv 管理 VirtualEnv

我一直在强调为你的每一个项目设立单独的`virtualenv`的重要性。有了诗，那是轻而易举的事。诗歌安装将总是安装一个新的`virtualenv`，除非你定义

```
poetry config virtualenvs.create false
```

这在 docker 设置中非常有用

但是您如何确保每次在目录中使用正确的`virtualenv`？诗已提供`poetry shell`。但是如果你忘记使用它了呢？如果你觉得每次都打出来太麻烦了怎么办？为此我强烈推荐`[direnv](https://direnv.net/)`。

关于`direnv`我不打算一一赘述，请自行查看链接。

但是怎么用在诗词上呢？我正在跟随[这个链接](https://github.com/direnv/direnv/wiki/Python#poetry)并添加我自己的快捷方式来创建一个新的诗歌项目。

## 自定义`direnv`命令

首先，您需要将以下内容添加到`$HOME/.direnvrc`中。

```
layout_poetry() {
  if [[ ! -f pyproject.toml ]]; then
    log_status 'No pyproject.toml found. Will initialize poetry in no-interactive mode'
    poetry init -n -q
    poetry run pip install -U pip wheel setuptools
  fi poetry run echo >> /dev/null
  local VENV=$(dirname $(poetry run which python))
  export VIRTUAL_ENV=$(echo "$VENV" | rev | cut -d'/' -f2- | rev)
  export POETRY_ACTIVE=1
  PATH_add "$VENV"
  if [ ! -L .venv ]; then
    ln -ns $VIRTUAL_ENV .venv
  fi
}
```

注意，我总是确保`wheel`在适当的位置，并且`pip`是最新的。对于一些 IDE 来说，符号链接“`.venv`”使得设置你的环境变得容易。

## 快速将文件夹变成诗歌的捷径

其次，将下面的脚本放在路径中的某个位置，并使其可执行。对我来说是`~/.local/bin/poetry-here`。

```
#!/usr/bin/env bash
echo "layout poetry" >.envrc
direnv allow .
direnv exec . echo
```

然后运行`poetry-here`到任何你想设置诗歌的地方。即使你还没有[用诗歌建立一个新的项目](https://poetry.eustace.io/docs/basic-usage/)。

```
> poetry-env
direnv: loading ~/repos/test/.envrc
direnv: No pyproject.toml found.  Will initialize poetry in no-interactive mode
Looking in indexes: [https://pypi.org/simple](https://pypi.org/simple)
direnv: ([direnv export fish]) is taking a while to execute. Use CTRL-C to give up.
Requirement already up-to-date: pip in /Users/leonmax/Library/Caches/pypoetry/virtualenvs/test-3nDJeJY5-py3.8/lib/python3.8/site-packages (20.2.2)
Requirement already up-to-date: wheel in /Users/leonmax/Library/Caches/pypoetry/virtualenvs/test-3nDJeJY5-py3.8/lib/python3.8/site-packages (0.35.1)
Requirement already up-to-date: setuptools in /Users/leonmax/Library/Caches/pypoetry/virtualenvs/test-3nDJeJY5-py3.8/lib/python3.8/site-packages (49.6.0)
Requirement already up-to-date: keyring in /Users/leonmax/Library/Caches/pypoetry/virtualenvs/test-3nDJeJY5-py3.8/lib/python3.8/site-packages (21.3.0)
direnv: export +POETRY_ACTIVE +VIRTUAL_ENV ~PATH
```

从现在开始，无论你什么时候进入这个目录，你都要使用`virtualvenv`。

```
> cd ~/repos/test
direnv: loading ~/repos/test/.envrc
cdirenv: export +POETRY_ACTIVE +VIRTUAL_ENV ~PATH
```

# 和你的私人 PyPI 一起使用

下面我将以 artifactory 版本的 PyPI 为例指导你使用。

## 先决条件

升级`pip ≥ 20.0.0`现在需要安装大部分软件包，强烈推荐升级你的`wheel`和`setuptools`，正如我在`direnv`部分提到的。到今天为止，最新的 pip 版本是`20.2.2.`

## 安装软件包

将下面几行添加到您的`pyproject.toml`文件中，以访问 artifactory private `PyPI`

```
[[tool.poetry.source]]
name = 'myrepo'
url = '[https://MY_PRIVATE_PYPI_SERVER/](https://artifact.rd.inceptioglobal.ai/artifactory/api/pypi/pypi/simple')[artifactory/api/pypi/pypi](https://artifact.rd.inceptioglobal.ai/artifactory/api/pypi/pypi-local)/[simple'](https://artifact.rd.inceptioglobal.ai/artifactory/api/pypi/pypi/simple')
```

另外，不要忘记添加您的凭证:

```
poetry config http-basic.myrepo <ARTIFACTORY_USERNAME> <ARTIFACTORY_API_KEYS>
```

下面是一个完整的例子

```
[tool.poetry]
name = "my-project"
version = "0.1.0"
description = ""
authors = ["Your Name <youremail>"][tool.poetry.dependencies]
python = "^3.6"
pyyaml = "^5.1"
click = "^7.1.2"[tool.poetry.dev-dependencies]
pytest = "^5.4.3"
pytest-cov = "^2.10.0"
mock = "^4.0.2"[[tool.poetry.source]]
name = 'myrepo'
url = '[https://MY_PRIVATE_PYPI_SERVER/](https://artifact.rd.inceptioglobal.ai/artifactory/api/pypi/pypi/simple')[artifactory/api/pypi/pypi](https://artifact.rd.inceptioglobal.ai/artifactory/api/pypi/pypi-local)/[simple'](https://artifact.rd.inceptioglobal.ai/artifactory/api/pypi/pypi/simple')[build-system]
requires = ["poetry>=0.12"]
build-backend = "poetry.masonry.api"
```

之后，你可以做你典型的`poetry install`

## 上传包

对于第三方包(在[pypi.org](http://pypi.org/)中不可用)，您也可以使用`poetry`上传到 artifactory。

首先，您需要像下面这样添加存储库配置。注意它与源代码的`index-ur`不同，存储库 URL 不应该在末尾包含 **/simple** 。

```
poetry config repositories.myrepo [https://](https://artifact.rd.inceptioglobal.ai/artifactory/api/pypi/pypi-local)[MY_PRIVATE_PYPI_SERVER](https://artifact.rd.inceptioglobal.ai/artifactory/api/pypi/pypi/simple')[/artifactory/api/pypi/pypi-local](https://artifact.rd.inceptioglobal.ai/artifactory/api/pypi/pypi-local)
```

凭证部分与安装包相同，因此您可以跳过。

```
poetry config http-basic.myrepo <ARTIFACTORY_USERNAME> <ARTIFACTORY_API_KEYS>
```

在您的回购中，只需发布:

`poetry publish -r myrepo`

# 在 Gitlab 的 CI/CD 中发布

下面我有一个 gitlab 中 CI/CD 的例子

```
image: python:3.6-slim.test:
  stage: test
  before_script:
    - pip install poetry
    - poetry config virtualenvs.create false
    - poetry install --no-interaction --no-ansi
  script:
    - pytest tests.3.6-slim: { extends: .test, image: 'python:3.6-slim' }
.3.7-slim: { extends: .test, image: 'python:3.7-slim' }.publish-pypi:
  stage: deploy
  script:
    - pip install poetry
    - poetry config virtualenvs.create false
    - export PRERELEASE=$(poetry version prerelease | sed -r "s/^.*to (.*)\.[0-9]+\$/\1/g")
    - export VERSION=${CI_COMMIT_TAG:-"$PRERELEASE.`date +%y%m%d%H%M%S`"} && echo $VERSION
    - poetry version $VERSION
    - poetry config repositories.myrepo $PYPI_REPOSITORY
    - poetry build -f wheel
    - poetry publish -n -r myrepo -u $PYPI_USER -p $PYPI_PASSWORDpublish-pypi:
  environment:
    name: release
  extends:
    - .publish-pypi
  only:
    - tagspublish-pypi-prerelease:
  environment:
    name: prerelease
  extends:
    - .publish-pypi
  only:
    - master
```

这里我有两个环境，`release`和`prerelease`，这两个环境的`$PYPI_REPOSITORY`、`$PYPI_USER`和`$PYPI_PASSWORD`变量是不同的。

此外，我们有两个不同的 python 版本要测试，所以您的`python = "^3.6"`真正适用于 3.6 以后的所有版本

如果您标记了您的版本，`$CI_COMMIT_TAG`将出现，否则，预发行版将在最后给出一个日期。

今天到此为止。如果你有任何想法，请让我知道。快乐的诗！