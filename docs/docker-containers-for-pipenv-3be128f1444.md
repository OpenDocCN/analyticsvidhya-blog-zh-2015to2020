# 管道码头集装箱

> 原文：<https://medium.com/analytics-vidhya/docker-containers-for-pipenv-3be128f1444?source=collection_archive---------6----------------------->

## 关于为使用 pipenv 进行依赖性管理的 python 项目编写 Dockerfile 的快速说明

## pipenv 是什么？

Pipenv 帮助我们在 Python 项目中维护一组确定的外部依赖关系。这很重要，因为当我们的代码在不同的时间安装在不同的机器上时，有可能会安装不同版本的软件包。此外，这会使您的代码崩溃，因为您的依赖项的 API 或行为会发生变化。

## requirements.txt 中具有 pipenv 依赖项的 Dockerfile

因为 Docker 为我们的代码运行提供了一个定义良好的环境，所以 pipenv 的任何可能都应该是 Docker 可以实现的。

为了在容器中安装合适的依赖项，我们可以将依赖项直接存储在`requirements.txt`中，然后从那里直接安装。这不仅创建了一个小得多的映像(因为容器中不再有 pipenv)，而且通过省略`pipenv install`减少了构建映像所需的时间(我们知道 [pipenv 安装很慢](https://github.com/pypa/pipenv/issues/2284))。

```
FROM python:3.6-alpine
...RUN apk update && \
    apk add --no-cache python3-dev && \
    pip3 install pipenv && \
    pipenv lock -r > requirements.txt && \
    pip3 uninstall --yes pipenv && \
    pip3 install -r requirements.txt && \
    apk del python3-dev && \
    ...
```

## 测试容器和开发依赖

在我们之前创建的映像中，我们没有安装开发依赖项。尽管这将使容器变小，但是我们将不再能够在容器内部运行测试。

这个问题可以通过为开发依赖项创建一个单独的需求文件来解决，并在需要的时候安装它

```
FROM python:3.6-alpine
...RUN apk update && \
    apk add --no-cache python3-dev && \
    pip3 install pipenv && \
    pipenv lock -r > requirements.txt && \
    **pipenv lock --dev -r > dev-requirements.txt** && \
    pip3 uninstall --yes pipenv && \
    pip3 install -r requirements.txt && \
    apk del python3-dev && \
    ...
```

要对此映像运行测试，我们可以在 CI 环境中进行以下设置—

```
test_docker:
  image: <*the-above-image-name*>
  before_script:
    - apk add python3-dev
  script:
    - pip3 install -r /src/dev-requirements.txt
    - python manage.py test
```

上面的片段使用了 [Gitlab](https://about.gitlab.com/) [CI](https://en.wikipedia.org/wiki/Continuous_integration) 和 [Django](https://www.djangoproject.com/) 作为例子，但是这个想法可以扩展到任何可能的组合。

## 限制

我一直在 GoSocial 中使用这个设置，没有遇到任何限制。然而，从项目中剥离 pipenv 的一个痛点是失去运行 Pipfile 中定义的[定制脚本的能力。这导致我们需要在 shell 中输入长而复杂的命令，大大增加了人为错误的范围。虽然在大多数情况下，这将是一个罕见的事件，因为我们已经预先编写了这些脚本(如 Procfiles、CI config 等。).](https://pipenv-fork.readthedocs.io/en/latest/advanced.html#custom-script-shortcuts)

pipenv 是一个非常棒的工具，可以为运行我们的 Python 代码建立一个定义良好的环境。但是当我们拥有像 Docker 这样的工具时，我们可以省略 pipenv 来跨机器运行更快、更一致的应用程序版本。这不仅导致了一个更小的容器，而且改进了它的构建时间，这是每个人都喜欢的。