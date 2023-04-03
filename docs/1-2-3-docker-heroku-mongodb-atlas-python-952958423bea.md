# 1，2，3: Docker，Heroku，MongoDB Atlas，Python

> 原文：<https://medium.com/analytics-vidhya/1-2-3-docker-heroku-mongodb-atlas-python-952958423bea?source=collection_archive---------9----------------------->

这是一个讲解如何使用 MongoDB Atlas 在 Heroku 上对接和部署 Python 应用程序的演练。

> 为什么是 1–2–3？关注简单的本质方面，清晰的代码样本和相关参考。

![](img/ba22ede9509e18c90aee85612d709741.png)

# 应用程序

这一切都从源代码开始。当然，让我们在 [Flask](https://flask.palletsprojects.com/en/1.1.x/) 框架上创建一个 Python 应用程序。

该应用程序提供了几个 REST 端点和一个 DAO 层来访问数据库。

```
# app.py 
from flask import Flask, request, Response
import logging
import requests, import os

try:
    app = Flask(__name__)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger(**'werkzeug'**).setLevel(logging.ERROR)
except Exception as e:
    logging.exception(**"Error at startup"**)

@app.route(**'/ping'**)
def ping():
    *"""
    Ping the endpoint
    :return:
    """
*    return **"ping Ok"** def get_port():
    *"""
    Retrieves port for env variable
    :return:
    """* return int(os.environ.get(**"PORT"**, 5000))

if __name__ == **'__main__'**:
    app.run(debug=True, port=get_port(), host=**'0.0.0.0'**)
```

验证您可以在您最喜欢的 IDE (PyCharm 是我的)中或从命令行(` *python app.py`* )运行该应用程序，它应该响应[http://localhost:5000/ping](http://localhost:5000/ping)

# MongoDB 地图集

注册 MongoDB Atlas(如果您还没有帐户)并设置您的免费 MongoDB。按照 Atlas 一步一步的向导，这是非常简单的。

![](img/b2ac001c6ed2adbf16d3296feb6ee01b.png)

简而言之，你需要:

*   创建一个集群
*   创建数据库用户
*   允许传入连接(编辑防火墙规则)
*   连接

一个常见的错误是忘记允许您的 IP 地址(上面的步骤 3):输入您当前的 IP 地址，或者如果您的存储库没有存储关键数据，则允许所有的 IP 地址(例如对于本教程)。

![](img/8da05b648cd6aca5c52f4a4a3b5dcfc9.png)

最后也是最有趣的一步是从 Python 应用程序连接到 MongoDB Atlas

```
#db_client.pyfrom pymongo import MongoClientconnect_string = 'mongodb+srv://{db_username}:{db_password}@devcluster.s4lc7.mongodb.net/{db_name}?retryWrites=true&w=majority'client = MongoClient(connect_string)
db = client.get_default_database()

# get user by name
def get_user_by_name(name):
    return db.user.find_one({**"name"**: name})

# add new user
def add_user(name):

    ret = get_user_by_name(name)

    if ret is None:
        new_user = {**"name"**: name}
        x = db.user.insert_one(new_user)
        ret = x.inserted_id

    return ret
```

*`connect_string`* 在源代码中是硬编码的(不要忘记用实际的 MongoDB 用户名、密码和 DB 名称替换占位符)，这样做是为了简化示例。您应该真正避免这种情况，而是将连接字符串(和任何其他配置)定义为环境变量。

# 赫罗库码头工人

我们现在有了一个可以工作的应用程序(你也可以添加一些单元测试)，是时候部署了。

Heroku Docker Registry 是部署您的应用程序的一个很好的解决方案，原因如下:

*   您构建了一个可以在其他 Docker 运行时上部署的通用映像
*   与从 git 存储库部署不同，Docker 映像不受大小限制(最大 500MB)

## 先决条件

首先创建一个新的 Heroku 应用程序(使用 Heroku CLI 或 web 界面),选择一个好名字☺️

其次(也是非常重要的),登录 Heroku 和 Heroku Docker 注册表

```
heroku login
heroku container:login
```

## Dockerfile 文件

这是我们示例的 docker 文件，从`python-slim`基本图像开始。

```
FROM python:3.9-slim

COPY app/ /app

EXPOSE 5000

WORKDIR /app
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["app.py"]
```

## 构建、推送、发布

摇滚的时间到了:让我们将应用程序 Dockerize 并将其推送到 Heroku。注意图像标签(registry.heroku.com)和流程类型(web)。

```
# build the image
docker build -t myapp .
# tag
docker tag myapp registry.heroku.com/myapp/web
# push
docker push registry.heroku.com/myapp/web
# release (when deployment starts)
heroku container:release web -a myapp
```

检查日志，看看它是否真的发生了

```
heroku logs -a myapp
```

调用 https://myapp.herokuapp.com/ping`/ping` 端点[的](https://dockerherokumongotutorial.herokuapp.com/ping)

# **TADAAAA！！！**😀

# 配置变量

最后一步是删除硬编码的数据库连接字符串(调皮的开发者！)并使用环境变量。

```
# read env variable
def get_mongo_connect_string():
    return os.environ.get(**"MONGO_CONNECT_STRING"**, **""**)
```

在您的本地开发环境中，创建一个通常被 gitignored 的`.env`文件并使用 Python [是一个好主意。dotenv](https://pypi.org/project/python-dotenv/) 包来加载这些值。

在 Heroku 上，您希望将环境变量定义为 [Config Vars](https://devcenter.heroku.com/articles/config-vars) :在部署时，Heroku 会将变量(连接字符串、令牌、秘密)注入为您的应用程序提供的 Dyno 中。

![](img/e72422be78a64c9959832b5f01f27b96.png)

# 结论

我希望这对开始使用 Docker 和 Heroku 的开发人员有所帮助:重点主要是提供清晰的代码和配置示例，因为这是最好的学习方法。

欢迎反馈！打 [@beppecatanese](https://twitter.com/beppecatanese) 找我

**参考文献**

在 [GitHub](https://github.com/gcatanese/1-2-3-Heroku-Docker-Python-MongoDB) 上查看演练源代码

devcenter.heroku.com[码头集装箱登记文件](https://devcenter.heroku.com/articles/container-registry-and-runtime)

在[docker.com](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)创建 docker 文件的最佳实践