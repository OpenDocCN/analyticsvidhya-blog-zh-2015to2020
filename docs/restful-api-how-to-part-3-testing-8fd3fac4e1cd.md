# RESTful API，如何|第 3 部分—测试

> 原文：<https://medium.com/analytics-vidhya/restful-api-how-to-part-3-testing-8fd3fac4e1cd?source=collection_archive---------21----------------------->

![](img/426628d467e0fc9849bee7e97a40a1ba.png)

设计和实施服务是我日常工作的一部分，我想分享一些最佳实践和技巧，可以帮助你的工作。

在这个关于 RESTful API 的系列文章中，我将讨论几个主题:

*   设计
*   履行
*   **测试**
*   部署

# 一些信息

> *我们将使用*[***swagger***](https://editor.swagger.io/)*来设计我们的 API，*[***python***](https://www.python.org/)*语言来创建微服务，最后*[***Docker***](https://www.docker.com/)*来交付最终的解决方案。所有的代码都在这个* [***回购***](https://github.com/dandpz/restfulapi-howto) *中。*

在之前的文章中，我们看到了如何设计一个简单的 RESTful API 以及如何实现它。现在是添加一些测试的时候了，但是首先让我们谈谈测试，它们有什么用，以及为什么你应该总是将它们添加到你的开发流程中。

在本文中，我们将讨论**单元测试**，它们对于测试应用程序的单个组件非常有用。他们可以帮助你在投入生产之前发现错误，如果你以正确的方式编写它们，他们还可以检查在代码改变后功能是否仍然工作。我们现在正在考虑测试后，我们已经开发了几乎所有的代码。也有像 TDD 这样的模式，首先编写测试，运行它们(它们会失败)，最后编写代码使测试通过。

从 swagger 生成的代码已经包含了一些测试，如果我们转到 tests 文件夹，我们可以看到类似这样的内容:

```
from swagger_server.test import BaseTestCase

class TestTodoController(BaseTestCase):
    *"""TodoController integration test stubs"""* def test_create_todo(self):
        *"""Test case for create_todo

        Create a new to-do
        """* body = Todo()
        response = self.client.open(
            "/v1/todo",
            method="POST",
            data=json.dumps(body),
            content_type="application/json",
        )
        self.assert200(response, "Response body is : " + response.data.decode("utf-8"))
```

这些测试存根对于测试客户端和我们刚刚创建的 API 之间的交互非常有用。我们不会关注这些测试，因为我们有一个数据库要与之交互，所以我们不能说这些测试是单元测试，事实上它们的结果取决于 HTTP 通信和数据库可用性。这些测试可以在**集成测试**环境中触发。

# 写作测试

让我们在同一个目录中创建一个新文件，我们将创建所有的测试用例来检查我们的 DAO 方法是否可以工作，但是首先，让我们稍微修改一下 **BaseTestCase** 类，以便它可以与 DB 一起工作。

```
import logging
import os

from flask_testing import TestCase

from swagger_server.__main__ import create_app

class BaseTestCase(TestCase):

    os.environ["APPLICATION_ENV"] = "Testing"

    def create_app(self):
        logging.getLogger("connexion.operation").setLevel("ERROR")
        app = create_app()

        return app.app
```

由于我们应用程序的配置，我们只能改变 **APPLICATION_ENV** 变量，现在我们已经准备好编写我们的第一个测试:

下面是第一个测试，这是一个如何创建一个案例的例子，测试数据库内的记录的创建。我们使用的流程如下:

*   我们首先创建一个测试对象
*   我们使用之前实现的 DAO 方法将它保存在 DB 中，
*   为了不使用代码中的其他方法，我们使用 SQL 原始查询来查询数据库
*   我们检查记录是否已经正确保存

```
class TestDaoController(BaseTestCase):

    def setUp(self):
        with self.app.app_context():
            db.create_all()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_save_todo_create_a_record_inside_db(self):
        test_todo = TodoModel(
            name="test_name",
            description="some desc",
            due_date=datetime.utcnow(),
            status=Status("to do")
        )
        test_todo.save()
        raw = text("select * from todo where name = '{}'".format("test_name"))
        result = db.session.execute(raw)
        results = []
        for r in result:
            results.append(r)
            self.assertTrue(r["name"] == "test_name")
        self.assertTrue(len(results) == 1)
```

两个方法 **setUp** 和 **tearDown** 值得注意，因为它们分别在每个测试用例之前和之后运行，确保每个用例有一个空的和干净的数据库，以便使每个测试用例独立。

完整的系列测试请查看[库](https://github.com/dandpz/restfulapi-howto)。

# 运行测试

现在是运行测试的时候了，由于生成了代码，我们已经有了一个几乎可以运行的环境。

如果我们已经正确地激活了虚拟环境并在其中安装了所有的需求，我们只需要在终端中键入 **tox** 并运行它。

我们将看到测试的结果。

# 一些考虑

单元测试是开发的一个重要部分，它有多个层次，试图在这一系列文章中涵盖这个主题的所有方面是错误的，因为它的复杂性，它确实值得一个专门的系列。

从这篇文章中最重要的是为你的应用程序编写测试的重要性，因为它们可以节省调试的时间，主要是在代码库被其他人共享的情况下。

用 **python** 测试**的一些**有用的**链接有:**

*   [https://tox.readthedocs.io/en/latest/](https://tox.readthedocs.io/en/latest/)
*   [https://docs.pytest.org/en/latest/](https://docs.pytest.org/en/latest/)
*   [https://flask.palletsprojects.com/en/1.1.x/testing/](https://flask.palletsprojects.com/en/1.1.x/testing/)
*   [https://docs.python.org/3/library/unittest.html](https://docs.python.org/3/library/unittest.html)

在本文中，我们介绍了单元测试的主题，如何编写单元测试以及如何在合适的环境中运行单元测试，我鼓励你探索这个主题，因为它看起来很难，实际上有时也很难，但这是一个实践，一个开发人员应该能够掌握。

在下一篇也是最后一篇文章中，我们将看到我们的应用程序在类似生产环境中的部署。

> ***提醒:*** *你可以在**[***这个 GitHub 资源库找到所有更新的代码！***](https://github.com/dandpz/restfulapi-howto)*
> 
> **链接往期文章:*[https://medium . com/analytics-vid hya/restful-API-how-to-part-2-implementation-E3 BCA 6072 b 70](/analytics-vidhya/restful-api-how-to-part-2-implementation-e3bca6072b70)*