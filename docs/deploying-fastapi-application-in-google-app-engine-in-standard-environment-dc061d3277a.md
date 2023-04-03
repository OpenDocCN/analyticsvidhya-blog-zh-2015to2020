# 在标准环境中在 Google App Engine 中部署 FastAPI 应用程序

> 原文：<https://medium.com/analytics-vidhya/deploying-fastapi-application-in-google-app-engine-in-standard-environment-dc061d3277a?source=collection_archive---------1----------------------->

FastAPI 是一个现代、快速(高性能)的 web 框架，用于基于标准 Python 类型提示用 Python 3.6+构建 API。要了解更多关于 FastAPI 的信息，你可以点击[这里](https://fastapi.tiangolo.com/)访问 FastAPI 的文档。

FastAPI 还在{base_url}/docs 中默认提供 Swagger UI 来测试 API。

## 装置

```
pip install fastapi
```

您还需要一个 ASGI 服务器，用于生产诸如[uvicon](https://www.uvicorn.org/)或 [Hypercorn](https://gitlab.com/pgjones/hypercorn) 之类的产品。我们将在这篇文章中使用 uvicorn。

Uvicorn 是一个快如闪电的 ASGI 服务器实现，使用了 [uvloop](https://github.com/MagicStack/uvloop) 和 [httptools](https://github.com/MagicStack/httptools) 。

直到最近，Python 还缺少一个用于 asyncio 框架的最低限度的底层服务器/应用程序接口。ASGI 规范[填补了这一空白，意味着我们现在能够开始构建一套可以跨所有 asyncio 框架使用的通用工具。](https://asgi.readthedocs.io/en/latest/)

要安装 uvicorn:

```
pip install uvicorn
```

这将安装具有最小(纯 Python)依赖性的 uvicorn。

```
pip install uvicorn[standard]
```

这将安装带有“基于 Cython”的依赖项(可能的话)和其他“可选附件”的 uvicorn。

在这种情况下，“基于 Cython”的含义如下:

*   如果可能，将安装并使用事件回路`uvloop`。
*   如果可能的话，http 协议将由`httptools`处理。

我更喜欢使用 uvicon[standard ],因为它安装了基于 cython 的依赖项，这将防止在生产中运行时出现与 uvloop 和 httptools 相关的错误。

最后，您还应该安装 Gunicorn，因为这可能是在生产环境中运行和管理 Uvicorn 的最简单方式。Uvicorn 包含一个 gunicorn worker 类，这意味着只需很少的配置就可以完成设置。在本地运行时，不需要安装 Gunicorn。

要安装 Gunicorn:

```
pip install gunicorn
```

## 冻结需求文件

在 virtualenv 中安装每个必需的依赖项后，不要忘记在部署之前冻结需求文件以进行更新，因为 App Engine 会从 requirements.txt 文件安装依赖项。

要冻结需求文件:

```
pip freeze > requirements.txt
```

## 配置 app.yaml 文件

您的 python 版本应该高于 3.6，FastAPI 才能工作。以下是我的项目配置:

```
runtime: python37
entrypoint: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
instance_class: F2
```

您可以根据需要选择任何高于 3.6 的 python 版本和 instance_class。下面将用四个工作进程启动 Gunicorn:

`gunicorn -w 4 -k uvicorn.workers.UvicornWorker`

main 是我的 main.py 文件，app 是我的 FastAPI 应用程序的实例。App Engine 处理端口号，但您可以定义您想要的端口号。

## 部署前确认

完成上述所有步骤后，最后一次确认以下内容:

1.  您的 virtualenv 已激活，并且通过激活环境安装了所有要求。
2.  确保您只安装了必要的依赖项并包括。gcloudignore 在部署过程中忽略不必要的文件和文件夹。
3.  在部署之前冻结 requirements.txt 文件，这样您就不会错过在需求文件中添加新安装的依赖项。
4.  您的 app.yaml 文件已正确配置。
5.  您的服务帐户 json 文件具有必要的访问权限。

## 在应用引擎中部署

如果您尚未安装 Google Cloud SDK，那么您必须安装并配置该 SDK。你可以点击[这个链接](https://cloud.google.com/sdk/docs/how-to)来正确配置你的 google cloud sdk。

安装 sdk 后，您需要初始化 sdk。要初始化云 SDK:

从终端运行`gcloud init`。

初始化后，确保选择了正确的项目 id。要从 google cloud 中选择项目，您必须运行`gcloud config set project [project_id]`

最后，在选定的项目 id 中部署您的 FastAPI 应用程序:

```
gcloud app deploy
```

您将获得在终端中查看应用程序的 url。