# Python 中的异步 Web 服务器

> 原文：<https://medium.com/analytics-vidhya/asynchronous-web-server-in-python-eac521fba518?source=collection_archive---------6----------------------->

## 如何使用 aiohttp 和 aiopg 用 Python 创建一个只有 1 个脚本文件的简单异步 web 服务器

![](img/e76556f6860649ef6d932f29eada6294.png)

克里斯托弗·高尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

在过去的一年里，我一直在从事一个项目，该项目需要异步 web 服务器来同时处理数千个用户连接，并向用户实时广播数据。面对学习异步编程和弄清楚如何创建异步 web 服务器的困难时期，我决定分享如何创建它。

本文主要关注使用`aiohttp`和`aiopg`的代码结构。我的目的是揭示异步 web 服务器的结构是什么样子的。如果您想深入了解 Python 中的异步，可以点击*同步与异步*部分的链接。

# 同步与异步

同步程序是一个逐个执行每个任务的程序。它只在另一个任务完成后才执行另一个任务。如果有一个步骤需要等待来自其他进程的返回值(例如，数据库查询、对其他 API 的请求)，它会阻塞程序，直到返回值可用，然后继续执行任务。之后，它执行另一个任务。

另一方面，异步程序是并发执行任务**的程序。**表示当有一个步骤需要等待其他流程的返回值时，它会处理其他任务，并在返回值可用之前返回到该任务。这在我们有一个带有 **I/O 绑定**的项目时很有用(一个有很多 I/O 进程的项目，比如处理很多请求并与数据库或其他服务交互的 web 服务器)。因此，web 服务器可以有效地处理许多请求，最大化 I/O 进程的等待时间。

在 Python 中，我们可以使用`asyncio`库创建异步应用程序。它使用`async`将一个函数转换成一个协程(一个可以在进程中暂停的特殊函数)。每当遇到`await`关键字时，协程将暂停进程，计算机将处理其他协程，然后当等待进程的返回值可用时，返回暂停的协程。你可以在这里阅读更多关于`asyncio` [的细节](https://realpython.com/async-io-python/)

# 初始化 web 服务器

我们将用`aiohttp` & `aiopg`创建一个 web 服务器，这是一个异步库，用于创建 web 服务器并使用`asyncio`与数据库交互。这个 web 服务器有 2 个 REST APIs，用于用户登录和用户注销，还有 1 个 PostgreSQL 数据库引擎。首先，我们需要使用 aiohttp 初始化 web 服务器应用程序，并使用 aiopg 创建一个数据库引擎。

```
*# Built-in library
import* json
from uuid import uuid4*# Third-party library
from* aiohttp *import* web
*from* aiopg.sa *import* create_engine*class* WebServer:
    *def* __init__(*self*, **kwargs: dict):
        *self*.app = web.Application()
        *self*.host = kwargs['webserver']['host']
        *self*.port = kwargs['webserver']['port']
        *self*.dbConf = kwargs['db']
        *self*.dbEngine = None
        *self*.sessionToUser = {}
        self.userToSession = {}
```

我们使用面向对象编程(OOP)范例，因为我们需要将 web 服务器应用程序、数据库引擎和用户会话封装在一个地方。它需要一个配置字典，可以是作为字典加载的 JSON 文件，也可以是硬编码的字典。接下来，我们创建用于创建数据库引擎的`initializer`方法，设置 API 路由，并返回 web 应用程序。

```
*async def* initializer(*self*) -> web.Application:
    *# Create a database engine
    self*.dbEngine = *await* create_engine(
        user=*self*.dbConf['user'],
        password=*self*.dbConf['password'],
        host=*self*.dbConf['host'],
        database=*self*.dbConf['database']
    )*# Setup routes and handlers*
    *self*.app.router.add_post('/user', *self*.loginHandler) *self*.app.router.add_delete('/user', *self*.logoutHandler)

    return self.app
```

我们通过使用需要用户、密码、主机和数据库参数的`create_engine`协程来创建数据库引擎。该引擎将用于获取到数据库的连接并执行查询。我们还为具有用于处理登录请求的`self.loginHandler`回调的`POST`方法和具有用于处理注销请求的`self.logoutHandler`回调的`DELETE`方法设置了路由`/user`。

# 创建请求处理程序

接下来，我们需要在`WebServer`类中创建`loginHandler`方法和`logoutHandler`方法。

```
*async def* loginHandler(*self*, request: web.Request) -> web.Response:
    *try*:
        # loads dictionary from JSON-formatted request body
        data = *await* request.json()
    *except* ValueError:
        *return* web.HTTPBadRequest() *if* 'username' *not in* data *or* 'password' *not in* data:
        *return* web.HTTPUnprocessableEntity() username = data['username']
    password = data['password']
    rawSql = 'SELECT password = %s verified FROM users where username = %s;'
    params = (password, username) query = *None
    async with self*.dbEngine.acquire() *as* dbConn:
        *async for* row *in* dbConn.execute(rawSql, params):
            query = dict(row) *if* query *is None*:
        *return* web.HTTPUnauthorized() *if not* query['verified']:
        *return* web.HTTPUnauthorized() sessionId = str(uuid4())
    *self*.userToSession[username] = sessionId
    *self*.sessionToUser[sessionId] = username
    response = {'session_id': sessionId}*return* web.json_response(response)
```

`loginHandler`方法接受 JSON 格式的请求体。在处理它之前，我们需要检查请求体。然后，我们使用`async with`和`async for`语句向数据库验证用户名-密码组合。`async with`语句是一个上下文管理器。在这种情况下，当查询过程完成时，它会自动断开数据库连接。`async for`语句用于异步迭代可迭代对象。在这种情况下，它用来异步地逐行查询数据库中的数据。因此，web 服务器可以在查询过程中处理其他请求。然后，如果验证过程成功，我们通过使用`uuid`库创建一个惟一的会话 ID，在`sessionToUser`和`userToSession`属性中将它与用户进行映射，并以 JSON 格式将其附加到响应对象。

```
*async def* logoutHandler(*self*, request: web.Request) -> web.Response:
    sessionId = dict(request.headers).get('Authorization') *if* sessionId *not in self*.sessionToUser:
        *return* web.HTTPUnauthorized() username = *self*.sessionToUser[sessionId] *self*.sessionToUser.pop(sessionId)
    *self*.userToSession.pop(username) *return* web.HTTPOk()
```

`logoutHandler`方法接受授权头中的会话 ID。如果会话 ID 被 web 服务器识别，则 web 服务器将从内存中删除该会话 ID。

# 运行 web 服务器

最后，我们需要创建运行 web 服务器的`run`方法，并在主函数中实现`WebServer`类。

```
*def* run(*self*):
    web.run_app(*self*.initializer(), host=*self*.host, port=*self*.port)
```

在 main 函数中，我们只需要加载 JSON 配置文件，用它来构造`WebServer`对象。然后使用`run`方法运行`WebServer`。

```
*if* __name__ == '__main__':
    *with* open('config.json') *as* fp:
        cfg = json.load(fp) webserver = WebServer(**cfg)
    webserver.run()
```

如果您想使用 JSON 文件进行配置，这里是 JSON 配置的样子:

```
{
  "webserver": {
    "host": "localhost",
    "port": 8000
  },"db": {
    "user": "<user>",
    "password": "<password>",
    "host": "<host>",
    "database": "<database>"
  }
}
```

# 最后的话

就是这样！只有一个脚本文件，我们可以使用 Python 中的`aiohttp`和`aiopg`创建一个异步 web 服务器。本文中的代码可以作为项目的样板。希望这篇文章能帮助你入门异步编程。

谢谢你，

伊克万·里兹基·努尔扎曼