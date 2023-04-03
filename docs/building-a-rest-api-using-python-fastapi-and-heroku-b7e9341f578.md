# 使用 Python、FastAPI 和 Heroku 构建 REST API

> 原文：<https://medium.com/analytics-vidhya/building-a-rest-api-using-python-fastapi-and-heroku-b7e9341f578?source=collection_archive---------3----------------------->

![](img/1d9f63a1d0109ab759d37530a55a8b18.png)

首先，谁从来不需要在几天甚至几个小时内构建和休息 API？我敢打赌，你这样做了，所以我有这个工作，创建一个地理定位 API，作为一个微服务。

我最近一直在使用 python，所以，第一个想法是使用 flask 来完成这项工作，但是我的老板要求我寻找新的方法，因为我已经有了使用 Flask 的经验。所以在我的研究中，我发现了一些选项，其中两个最高的是 [Django](https://www.djangoproject.com) 和 [FastAPI](https://fastapi.tiangolo.com) 。

首先决定测试 Django，我之前见过的那个，是否符合我们的需求。它最终成为一个非常强大的框架，用来做很多不必要的事情，所以让我们看看 FastAPI 是如何为我工作的。

所以，让我们开始学习如何安装它。在本文中，我将使用 pipenv，但是您可以使用任何您喜欢的包管理器:

```
pipenv install fastapi
pipenv install uvicorn
```

我们将使用 uvicorn 作为 ASGI 服务器。

现在，您已经安装了 uvicorn 和 FastAPI，现在可以开始编码了。

让我们创建架构:

```
├── main.py
├── manager.py
├── app          
│   ├── router.py
│   ├── configs.py         
│   └── geolocation 
│       ├── services.py
│       ├── controller.py
│       ├── models.py
│       └── routes.py
└── pipfile
```

假设每个文件夹都需要一个 __init__。py 文件被视为一个模块。

现在，开始编码。main.py 负责运行和设置您的服务器。

我们将前缀设置为“v1”，因此我们可以有多个版本的 API。

现在设置 manager.py:

这个文件只是我创建的一个有用的文件，用来帮助创建模块，要使用它，您应该运行以下命令:

```
python manager.py module MODULE_NAME
```

在 router.py 文件中，您需要设置所有主路由，每个模块一条。我们使用 API 路由器，这样我们就可以将所有的路由集中在同一个地方。

现在，在地理位置文件夹中创建负责地理位置模块端点的文件 routes.py。

记住，这只是处理端点参数的一种方式，其他方式你可以在[这里找到](https://fastapi.tiangolo.com/tutorial/body/)。

下一步是控制器，这里您需要接收响应体或参数，并对其进行组织以向服务提供数据:

现在，服务，负责所有的魔法，在这里你可以做所有的数据处理，请求，数据库管理，任何你需要的东西。我们将向 IP 提供商发出地理定位请求。

您可以看到，我们使用了一个名为 configs.py 的文件，在这个文件中，我们可以存储任何静态数据，例如 API 令牌。

```
API_TOKEN = "YOUR_API_TOKEN_HERE"
```

现在，完成所有这些编码后，我们可以将这个项目构建到 Heroku 或任何云平台上。创建 Heroku 应用程序后，您需要使用 Procfile 进行部署。

```
web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}
```

Procfile 应该如下所示。现在，在 Heroku 中寻找你的项目的 URL:[https://YOUR-PROJECT-NAME.herokuapp.com/v1/geolocation/](https://hugosteixeira-services.herokuapp.com/v1/geolocation/)

结果是这样的:

```
{"lat":"-8.0539","long":"-34.8811"}
```

现在，您可以改进此服务，保存创建数据库缓存的请求，您可以使用 manager.py 创建其他服务并将其添加到 router.py