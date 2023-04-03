# 一个简短但完整的 Python Flask 应用程序开发指南

> 原文：<https://medium.com/analytics-vidhya/a-short-but-complete-guide-to-python-flask-app-development-9b493f960bd1?source=collection_archive---------5----------------------->

***目标:***

*   用 Python Flask framework 编写一个基本的 REST 应用程序。
*   提供基本特性的框架代码，这样你就不必在很多地方寻找基本的东西。
*   创建一个 Docker 映像，它可以很容易地部署在任何服务器上(机器或虚拟机)。
*   只为*提供概念的基本原则*，但为有用的设计模式提供工作代码。和链接，了解更多信息。

***注意:*** *这可能感觉像是一篇很长的文章，它的目的是以简洁的方式来实现完整性。目标受众:初学者(但不是绝对的初学者)，或者中级。*

F 第一件事: ***设置开发环境。***

要轻松编写代码(自动完成、语法高亮等):使用 [IntelliJ](https://www.jetbrains.com/idea/) (带 Python 插件)，或者 [PyCharm](https://www.jetbrains.com/pycharm/) IDE，或者其他你喜欢的东西。

运行应用程序代码:基本上，您所需要的就是在您的机器上安装 python 解释器，但是建议您设置一个[“虚拟环境”](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system.)(简称*，*这样，当您有多个 Python 项目，或者在您的系统上安装了 python2 并且想要在您的项目中使用 Python 3*时，就不太可能出错)。*

*要使用 python 库、框架(例如 Flask):你需要一个包管理器——Pip。*

*为您的项目设置 *Python 解释器*(使用 IntelliJ/PyCharm 应该很容易，只需按两次 Shift 键并搜索“Python 解释器”，它会打开一个窗口，您可以在其中设置解释器，您甚至可以创建一个虚拟环境)。创建虚拟环境有 3 个选项/工具: **virtulenv** (最老的一个) **pyenv** (基本就是 pip+virtualenv)**conda。** [见这里的区别](https://stackoverflow.com/questions/38217545/what-is-the-difference-between-pyenv-virtualenv-anaconda)。关于这一点，有大量令人困惑的工具。请访问此链接了解详情。*

*Anaconda 和 Miniconda 中都包含了 pip 和 conda，所以不需要单独安装。不要混合 pip 和 conda，这可能会意外损坏东西。*

*现在让我们使用`[venv](https://docs.python.org/3/library/venv.html)`工具。*

*通过运行一个简单的 python 脚本，确保到目前为止一切正常。*

*第二件事:测试框架。*

*如果你是一个初学者，从测试开始是“最好的”,这样你就能建立起编写测试所需的肌肉记忆，并且不会感到前进的沉重。*

*如果你是有经验的程序员，你一定已经对 TDD ( [测试驱动开发](https://en.wikipedia.org/wiki/Test-driven_development))有偏见，你可以试图说服自己支持或反对 TDD，但是无论如何你都需要测试。所以，让我们马上设置它。*

*Pytest 是一个很棒的/丰富的框架，用于在 python 中运行测试。*

*要使用它，在你的项目文件夹中创建一个名为`requirements`的文件夹，并在这个文件夹中创建两个文件`development.txt`和`production.txt`。这些文件将交给包管理器(pip)来安装库，包括`pytest`(我们称之为依赖)。但是为什么是两个文件呢:因为有些库，比如 pytest，在你创建一个可部署的工件(比如一个可以在服务器上部署的 docker 映像)时是不需要的，它只在开发时需要。以下是到目前为止的`development.txt`文件:*

```
*-r production.txt # Whatever library required in production, would also be required while developing, so include it here.pytest*
```

*创建一个名为`tests`的文件夹。Python 框架不强制/标准化任何特定的文件夹结构(不像 Java 世界中的框架，例如 maven/gradle)。但是`project_folder/app, project_folder/tests`、`project_folder/integration-tests`，看起来结构足够好。*

*确保在 tests 文件夹中编写一个虚拟测试，并使用 pytest 运行它。*

*要运行虚拟测试，首先运行`pip install -r requirements/development.txt`，在安装完成后，运行`pytest` pytest 自动检测测试文件， ***你必须用*** `***test_***` ***前缀开始你的测试文件和测试方法。****

*你可以看到我们目前拥有的:[这里](https://github.com/trexsatya/sample-flask-app/tree/Step_1)。*

*第三件事:休息控制器*

*我们希望为用户提供一个 REST API。(REST 基本上意味着资源/事物的标准化表示*，您可以在它上面学习更多内容*)*

*也就是说，我们希望我们的应用程序处理这些 HTTP URLs:*

```
*GET /users, GET /users/{user_id}
POST /users
....*
```

*让我们先写一个测试，*

```
*class TestUserController:
    def test_that_application_handles_user_request(self):
       response = testapp.get("/api/v1/users")
       assert response is not None*
```

*这是我们第一次测试中最不需要的。但是，我们没有建议下一个行动项目的`tesatpp`，我们需要设置我们的应用程序。*

*为此我们需要安装[烧瓶](https://flask.palletsprojects.com/en/1.1.x/)，在`requirements/production.txt`文件中添加新的一行`Flask==1.1.2`。我们还需要`WebTest`库。*

*然后再次做`pip install`后，我们可以创建一个应用程序来测试。为此，我们将使用 [Pytest 夹具](https://docs.pytest.org/en/stable/fixture.html)。基本上，fixtures 允许你设置对象(服务，应用程序等等)，并把它作为测试函数的一个参数。*(这种设置测试夹具的风格不同于其他编程语言中的 JUnit、xUnit 框架。)**

*这是 testapp *(因此，整个应用程序是一个单独的对象)*的测试夹具，我们将在我们的测试方法中使用它。*

```
*import pytest
from flask import Flask
from webtest import TestApp

class TestConfig:
    pass

def create_app(config_object):
    _app = Flask(__name__.split(".")[0])
    _app.config.from_object(config_object)
    return _app

@pytest.fixture(scope='function')
# scope='function' means that this object would be created when it is required by test function,
# and will be destroyed when the function completes (i.e. fixture goes out of scope)
def app():
    *"""An application for running tests"""* _app = create_app(TestConfig)

    with _app.app_context():
        pass

    ctx = _app.test_request_context()
    ctx.push()

    yield _app
    # Below is the tear-down code run after the fixture goes out of scope
    ctx.pop()

@pytest.fixture(scope='function')
def testapp(app): # app comes from the fixture defined above
    *"""A Webtest app."""* return TestApp(app)*
```

*这里有一些重要的事情需要注意:*

1.  *这是一种[依赖注入](https://en.wikipedia.org/wiki/Dependency_injection)，你定义/告诉应用程序对象应该如何被创建，当需要时框架自动创建/注入它，当超出范围时销毁它*，你不必创建，只需在需要时询问*。*
2.  *这看起来像注释(如果你熟悉其他编程语言的话)，但 python 中的机制与注释不同。这些被称为[装饰者](https://wiki.python.org/moin/PythonDecorators)又称包装者。基本上，每当调用`testapp`函数时，python 解释器实际上会调用`pytest.wrapper`并将`testapp`函数作为参数传递。我们将创建自己的定制装饰器，这将使它更加清晰。*(注意:这不同于装饰模式，在装饰模式中，你通过将一个对象封装到另一个具有相同接口的对象中来修改它的行为，这里你是动态地修改一个函数/类。* [*亦见此线程*](https://stackoverflow.com/questions/8328824/what-is-the-difference-between-python-decorators-and-the-decorator-pattern) *。**
3.  *我们有最小的配置对象，稍后我们将填充它。*

*现在，如果您运行`pytest`命令，您应该会看到错误*

```
*webtest.app.AppError: Bad response: 404 NOT FOUND*
```

*这是因为 WebTest framework 会对所有不是 200 或 300 左右的响应抛出错误，*除非被告知要这样做。*所以，你只要在调用 get 方法的时候加一个参数`status='*'`就可以了。*

```
*class TestUserController:
    def test_that_application_handles_user_request(self):
       response = testapp.get("/api/v1/users", status="*")
       assert response is not None*
```

*现在，你的测试应该通过了。但是，您实际上需要 200 个响应，而不是 404，因此我们需要通过编写一个控制器来处理这个 URL。*

*向您的测试添加一个新行，它应该会失败。*

```
*assert response.status_code == 200*
```

*注意:你需要一个`tests`文件夹下的`__main__.py`文件来使用`app`文件夹中的代码。*

*文件:app/controllers/user _ controller . py*

```
*from flask import jsonify, Blueprint
from flask.views import MethodView

FAKE_DATA = [{"user_id": 1, "value": "duck"}, {"user_id": 2, "value": "cat"}]

user_blueprint = Blueprint("users_api", __name__)

class UserController(MethodView):
    def get(self, user_id):
        if user_id is None:
            return jsonify(FAKE_DATA)
        else:
            return jsonify(list(filter(lambda x: x['user_id'] == user_id, FAKE_DATA)))

user_view = UserController.as_view("api")
user_blueprint.add_url_rule(
        "/api/v1/users", defaults={"user_id": None}, view_func=user_view, methods=["GET"]
)
user_blueprint.add_url_rule("/api/v1/users/<int:user_id>", view_func=user_view, methods=["GET"])*
```

*这里，我们使用了[烧瓶蓝图](https://flask.palletsprojects.com/en/1.1.x/tutorial/views/)和[方法视图](https://flask.palletsprojects.com/en/1.1.x/api/#flask.views.MethodView)。实际上，我们不需要蓝图和 MethodView，因为 Flask 需要的只是一个函数，它可以为配置的 URL 返回响应。但是，当您有很多 API 时，使用 Blueprint 会使事情易于管理，而使用 MethodView 会使它更适合 REST 实现。*

*但是，如果你了解 Spring/SpringBoot 控制器，你会注意到我们必须在这里做一些额外的编码，而不仅仅是声明 API。这可以简化，有一个库可以做到这一点，但是我们现在不打算使用它。*

*现在，您已经在蓝图中配置了路由(URL ),但是 Flask 仍然不知道它们。为此，我们必须用从`create_app`函数返回的应用程序注册这个蓝图。但在此之前，让我们进行重构，将该函数从 tests 目录中取出，因为它将用于测试和作为服务器实际运行我们的应用程序。*

*你可以看到我们到目前为止所拥有的，[这里](https://github.com/trexsatya/sample-flask-app/tree/Step_2)。*

*第四件事:使用数据库*

*让我们添加 api 来添加新用户，然后将其保存到数据库中。首先，编写一个添加用户的测试。*

```
*def test_that_invalid_data_is_not_allowed_in_creation(self, testapp):
    response = testapp.post_json("/api/v1/users", {"name": 1231, "email": "invalid_mail"}, status="*")
    assert response.status_code == 400

def test_that_we_can_create_new_user(self, testapp):
    response = testapp.post_json("/api/v1/users", {"name": "ivhas", "email": "abc@mail.com"}, status="*")
    assert response.status_code == 200
    assert response.json["name"] == "ivhas"*
```

*你应该看到"<405 METHOD NOT ALLOWED..” on running 【 now.*

*We need to create some class for transferring request and response body for users, which also has information about what kind of data is allowed.*

*For that we’ll use another library [棉花糖](https://marshmallow.readthedocs.io/en/stable/quickstart.html)。它有不同的内置类型，例如电子邮件，日期时间等。基本上，它是一个用于序列化/反序列化目的的库。但是，它的工作模式又不同于 Java world(像 Jackson 这样的库)。这里我们添加了另外两个库(到 requirements/production . txt):`marshmallow`和`flask-apispec`*

*我们创建一个新文件`app/dto/user.py`来包含用户的数据传输对象模型类:*

```
*from marshmallow import post_load, fields, Schema
import datetime as dt

class UserData:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.created_at = dt.datetime.now()

    def __repr__(self):
        return "<User(name={self.name!r})>".format(self=self)

class UserSchema(Schema):
    name = fields.Str(required=True)
    email = fields.Email(required=True)
    created_at = fields.DateTime()

    @post_load # So that when deserialising, you get UserData object
    def make_user(self, data, **kwargs):
        return UserData(**data)*
```

*需要注意一些事情:*

1.  *Marshmallow 定义了模型的模式，实际的 DTO 模型是 UserData。Marshmallow 允许您对字段进行验证，我们将为`name`字段添加一个自定义验证器。*
2.  *您可以使用`dumps`和`load`方法来反序列化为 JSON 字符串，并从 JSON 字符串序列化。但是，为了在我们的控制器中使用它，我们将只直接使用`load`方法，并将使用`flask-apispec`库来发送响应。*

```
*from flask import jsonify, Blueprint, request
from flask.views import MethodView
from flask_apispec import marshal_with
from app.dto.user import UserSchemauser_blueprint = Blueprint("users_api", __name__)

user_schema = UserSchema()

class UserController(MethodView): #... @marshal_with(user_schema)
    def post(self):
        user_data = user_schema.load(request.get_json())
        return user_data

user_view = UserController.as_view("api")
...
user_blueprint.add_url_rule("/api/v1/users", view_func=user_view, methods=["POST"])*
```

*现在，你的一个测试应该通过了，当你的请求中有无效数据时，一个测试会失败，但是它实际上给出了 500 个响应，而不是 400 个。*

*所以，我们需要解决这个问题。*

```
*def validate_with(schema: Schema):
    req_json = request.get_json()
    errors = schema.validate(req_json)
    if errors:
        resp = make_response(jsonify(errors), 400)
        abort(resp)*
```

*你可以看到目前为止的代码- [这里](https://github.com/trexsatya/sample-flask-app/tree/Step_3)。*

*现在，我们开始添加数据库。*

*首先，添加一个测试*

```
*def test_that_user_is_saved_to_database_after_creation(self, testapp, db):
    response = testapp.post_json("/api/v1/users", {"name": "ivhas", "email": "abc@mail.com"}, status="*")
    assert len(db.session.query(User).all()) == 1*
```

*在这里，`db`就像 app 一样是一个固定物。我们需要配置它，我们还需要将它连接到 Flask app。我们将使用[flasksqlanchy](https://flask-sqlalchemy.palletsprojects.com/en/2.x/)进行[对象关系映射](https://en.wikipedia.org/wiki/Object%E2%80%93relational_mapping)(这样您就可以在代码中将 DB 表视为对象，*这主要是因为应用程序开发人员不想处理 SQL 代码*)。*

*这将进入`conftest.py`文件*

```
*@pytest.fixture(scope="function")
def db(app):
    # By this fixture, we'll have a clean DB for each test method.
    database.app = app
    with app.app_context():
        database.create_all()

    yield database

    database.session.close()
    database.drop_all()*
```

*这里使用的`database`将是一个全局对象，它将绑定到我们在`create_app`函数中的应用程序，并且它还将绑定到所有的实体模型(在数据库中表示我们的表的类)。*

*所以，把这段代码放在 app 文件夹的某个地方，我已经把它放在一个新的文件里`app/extensions.py`*

```
*database = SQLAlchemy(model_class=CRUDMixin)*
```

*`CRUDMixin`是一个实用程序类，为了简单起见，它可以使用[活动记录](https://en.wikipedia.org/wiki/Active_record_pattern)模式。*

*我们的实体模型很简单:`app/entities/user.py`*

```
*from sqlalchemy import String, Integer
import datetime as dt
from app.extensions import database

class User(database.Model):
    __tablename__ = "users"
    id = database.Column(Integer, primary_key=True)
    name = database.Column(String(10), nullable=False)
    email = database.Column(String(50), nullable=False)

    created_at = database.Column(database.DateTime, nullable=False, default=dt.datetime.utcnow)
    updated_at = database.Column(database.DateTime, nullable=False, default=dt.datetime.utcnow)

    def __init__(self, **kwargs):
        database.Model.__init__(self, **kwargs)*
```

*还有，你需要设置`SQLALCHEMY_DATABASE_URI = ‘sqlite://’` conftest.py 文件，`TestConfig`类。这意味着我们将使用内存中的 [SQLite](https://www.sqlite.org/index.html) 数据库进行测试。*

*看到目前为止的代码— [这里](https://github.com/trexsatya/sample-flask-app/tree/Step_4)。*

*第五件事:服务、事务和依赖注入*

*基本上，服务充当控制器和数据库之间的中介。它们用于将一些相关的事情组合在一起，例如，创建新用户时应该发生的所有事情。他们还负责交易。*(事务是指一堆事情，要么全部成功，要么都不生效)*。*

*首先，让我们将用户创建代码重构为一个单独的类`UserService`*

*`app/services/user.py`*

```
*from app.dto.user import UserData
from app.entities.user import User

class UserService:
    def __init__(self):
        pass

    def create_user(self, user_data: UserData):
        user = User(name=user_data.name, email=user_data.email).save()

        return user*
```

*现在，我们需要另一个库来进行平滑的依赖注入:`Flask-Injector`*

*在这之后，我们需要配置哪些类/对象将由依赖注入来管理。更改`create_app`功能:-*

```
***from flask_injector import FlaskInjector
from injector import singleton**def create_app(config_object):
    _app = Flask(__name__.split(".")[0])
    _app.config.from_object(config_object)
    _app.register_blueprint(user_blueprint)

    **def configure_di(binder):
        binder.bind(ExternalService, to=ExternalService, scope=singleton)
        binder.bind(UserService, to=UserService, scope=singleton)**

 **FlaskInjector(app=_app, modules=[configure_di])**
    database.init_app(_app)

    return _app*
```

*我创建 ExternalService 只是为了演示，它可以是用户服务使用的任何服务，例如进行 API 调用。*

*现在，我们需要在任何需要的地方注入 服务。首先，UserController 中需要 UserService*

```
***from injector import inject**
class UserController(MethodView):
 **@inject**
    **def __init__(self, user_service: UserService):
        self.user_service = user_service** @marshal_with(user_schema)
    def post(self):
        validate_with(user_schema)
        user_data: UserData = user_schema.load(request.get_json())
 **user = self.user_service.create_user(user_data)**
        return user*
```

*运行`pytest`所有测试都应通过。所有的功能都是一样的。*

*现在，让我们设置事务。对于这里的演示，我们将在创建用户时调用 ExternalService，如果该调用失败，则不应创建用户，因为事务意味着*“要么事务下的每个任务都成功，要么都不成功”*。所以，让我们写一个测试*

```
*def test_that_user_is_not_saved_to_database_if_transaction_fails(self, testapp, db, monkeypatch):
    def mock_call():
        raise Exception("Unknown")
    monkeypatch.setattr(ExternalService, "call", mock_call)
    response = testapp.post_json("/api/v1/users", {"name": "ivhas", "email": "abc@mail.com"}, status="*")
    assert len(db.session.query(User).all()) == 0, "Shouldn't have been saved!"*
```

*这里我们使用 pyests `monkeypatch`来代替对 ExternalService 的实际调用，这个`mock_call`函数将被调用，我们可以在测试本身中控制它。运行测试，这个应该会失败。*

*为了使用事务，我们将使用另一个名为`transaction`和`zope.sqlalchemy`的库(注意:zope 来自 [ZODB](http://www.zodb.org/en/latest/) ，但它实际上对它的依赖性很小，可以在我们的应用程序中不使用 ZODB 而使用)。*

*`transaction` package 提供了一种方法，可以用来将我们的 SQLAlchemy 数据库会话绑定到它的事务管理器。*

*在`app/extensions.py`文件中:-*

```
*register(database.session)*
```

*另外，用户服务中的`create_user`函数变成:-*

```
*def create_user(self, user_data: UserData):
    **transaction.begin()**
    user = User(name=user_data.name, email=user_data.email).save(**commit=False**)
    self.external_service.call()
 **transaction.commit()**
    return user*
```

*现在，我们还需要一件事，当一些异常发生时，我们需要中止事务。为了简单起见，我们将使用 Flask 在请求级别处理它。也就是说，我们将向我们的应用程序注册一个错误处理程序，并在那里中止事务。*(注:可能有更复杂的方式，对交易有更复杂的要求)**

*在我们的`app/main.py`文件中添加这个:-(如果出现意外情况，这个函数也会导致应用程序返回 JSON 错误消息作为响应)*

```
*def generic_error_handler(exception):
    *"""
    This will handle all uncaught exceptions while Flask is processing a request.* ***:param*** *exception:* ***:return****:
    """* 
 **transaction.abort()**
    trace = "\n".join(traceback.format_exception(etype=type(exception), value=exception, tb=exception.__traceback__))
    print(trace)
    return {"message": "Error"}, 500*
```

*并在`create_app`功能中添加:-*

```
*_app.errorhandler(Exception)(generic_error_handler)*
```

*就是这样。现在，如果你运行`pytest`，所有的测试都应该通过。*

*注意:这一行在`app/extensions.py`文件中也发生了变化，因此在`Session.commit()`操作之后，会话中的对象不会[过期](https://docs.sqlalchemy.org/en/13/glossary.html#term-expired)，如果随后访问它们的属性，就会导致延迟加载。*

```
*database = SQLAlchemy(model_class=CRUDMixin, **session_options={"expire_on_commit": False}**)*
```

*看到目前为止的代码— [这里](https://github.com/trexsatya/sample-flask-app/tree/Step_5)。*

*第九件事:建立实际的应用程序，并创建一个可部署的 Docker 映像。*

*下面是我们将在这一部分实现的内容列表:-*

1.  *使应用程序可运行，这样你就可以从浏览器、终端(使用 curl 等)访问它*
2.  *将应用程序连接到一个真实的数据库(目前是 MySQL)*
3.  *使应用程序可部署。*

*为了使应用程序可运行，我们需要使用`create_app`函数创建一个应用程序实例。我们在一个新文件`<project_root>/autoapp.py`中这样做，并把它放在那里:-*

```
*from flask.helpers import get_debug_flag

from app.configs import DevConfig, ProdConfig
from app.main import create_app

CONFIG = DevConfig if get_debug_flag() else ProdConfig

app = create_app(CONFIG)*
```

*为此，我们还需要在一个新文件`app/configs.py`中创建单独的配置对象*

```
*import os

from dotenv import load_dotenv

load_dotenv()
# This will load .env file containing environment variables

class Config(object):
    *"""Base Configuration"""* APP_DIR = os.path.abspath(os.path.dirname(__file__))  # This directory
    PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, os.pardir))

class ProdConfig(Config):
    ENV = 'prod'
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.getenv("DB_URL")

class DevConfig(Config):
    ENV = 'dev'
    DEBUG = True
    DB_NAME = 'dev.db'
    DB_PATH = os.path.join(Config.PROJECT_ROOT, DB_NAME)
    SQLALCHEMY_DATABASE_URI = os.getenv('DB_URL',
                                        'sqlite:///{0}'.format(DB_PATH)  # default
                                        )*
```

*我们已经使用 [Python-dotenv](https://github.com/theskumar/python-dotenv) 将所有环境变量放在一个名为`.env`的文件中*

```
*DB_URL=mysql+mysqlconnector://user:password@localhost:3306/db*
```

*现在，可以使用以下命令启动该应用程序*

```
*export FLASK_APP=./autoapp.py
flask run*
```

*你可以尝试访问网址，你应该看到错误。我们还没有建立数据库。*

*你可以在你的系统上安装 mysql，并创建一个名为`db`的数据库，但是现在我们将使用 MySQL 的 docker 镜像。*

*在项目根目录下创建一个文件【T5:】*

```
*version: "3"

networks:
  backend:
    driver: bridge

volumes:
  my-db:

services:
  mysql:
    image: mysql:8.0.22
    container_name: mydb
    ports:
      - "3306:3306"
    environment:
      MYSQL_DATABASE: 'db'
      *# So you don't have to use root, but you can if you like* MYSQL_USER: 'user'
      *# You can use whatever password you like* MYSQL_PASSWORD: 'password'
      *# Password for root access* MYSQL_ROOT_PASSWORD: 'password'
    networks:
      - backend
    expose:
      - '3306'
    volumes:
      - my-db:/var/lib/mysql*
```

*您需要在系统上安装 [DockerCompose](https://docs.docker.com/compose/install/) 。*

*运行命令`docker-compose up`它将从 DockerHub 下载 MySQL 的 docker 镜像并创建一个容器(这意味着镜像被启动/启动)。现在你有了一个我们的应用程序可以访问的数据库。*

*但是，如果您再次运行应用程序并尝试访问 API，它仍然会抛出一个错误，因为我们仍然没有在数据库中创建表。*

*为此，我们将使用另一个库 [Flask-Migrate](https://docs.docker.com/compose/install/)*

*它需要一个关键的改变(除了将它添加到 requirements/production.txt 文件中):-*

*我们需要创建一个迁移对象，并将其连接到我们的应用程序和数据库。*

*所以，先把这一行加到`app/extensions.py`*

```
*migrate = Migrate()*
```

*接下来，将这一行添加到`create_app`函数中:-*

```
*migrate.init_app(_app, db=database)*
```

*然后，我们必须使用这个迁移框架在数据库中创建表。运行以下命令:-*

```
*flask db init
flask db migrate
flask db upgrade*
```

*第一个命令创建迁移所需的文件(在项目目录中)。*

*第二个命令检测应用程序中关于表的更改，并创建一个迁移脚本文件。该文件的名称将是随机散列。*

*第三个命令将迁移脚本应用于数据库，并创建/修改表等。*

*(注意:此框架的自动检测并不是 100%可靠的，因此请确保验证生成的迁移脚本文件中的更改。第一次没有必要检查，除非你想定制)*

*现在，如果您重新运行应用程序，一切都应该工作正常。*

*最后，我们需要将应用程序部署为 docker 映像。*

*在项目目录中创建一个名为`Dockerfile`的文件:-*

```
*FROM python:3.8-slim-buster
# This will serve as base image on which other things can be added via commands, basically you get a lightweight operating system with Python installed on it

WORKDIR /src
#sets the working directory for any RUN, CMD, ENTRYPOINT, COPY and ADD instructions, a src folder will be created in docker image

ARG *BUILD_ENV* ENV *FLASK_ENV* $*BUILD_ENV* ENV *FLASK_APP* autoapp.py

#RUN mkdir requirements

COPY app/ ./app
COPY autoapp.py .
COPY requirements/ ./requirements
COPY run_tests.sh .
# Copy required source code
COPY tests ./tests

RUN ls .
# just for debugging to see what's ther in the WORKDIR

RUN if [ "$*BUILD_ENV*" = "test" ]; then pip install -r requirements/development.txt ; fi
RUN if [ "$*BUILD_ENV*" = "test" ]; then export FLASK_DEBUG=true ; fi
RUN if [ "$*BUILD_ENV*" != "test" ]; then pip install -r requirements/production.txt ; fi

RUN chmod 777 run_tests.sh
RUN ./run_tests.sh
ENTRYPOINT flask run

# A brief about RUN, CMD, and ENTRYPOINT
# Use RUN to do the installation, making changes etc
# Use ENTRYPOINT to run a command used to start the service
# Use CMD to pass arguments to that command, they get appended automatically; You can override CMD while running docker image.*
```

*如果你在理解 docker file[时遇到困难，访问官方网站，文档是很好的选择。](https://docs.docker.com/get-docker/)*

*此外，将以下内容添加到我们之前创建的`docker-compose`文件中:-*

```
*flask_app:
    build:
      dockerfile: Dockerfile
      context: .
      args:
        BUILD_ENV: $BUILD_ENVcontainer_name: flask_app
    expose:
      - '5000'
    environment:
      - FLASK_DEBUG=$FLASK_DEBUG
      - DB_URL=$DB_URL
    ports:
      - "5000:5000"
    network_mode: bridge*
```

*取消之前运行的命令，并再次运行 compose 命令:-*

```
*BUILD_ENV=test docker-compose up --build*
```

**注意:从浏览器或 Mac 终端访问 docker 中运行的应用程序可能会遇到困难。我还是没想出问题。**

*使用上面的命令为你的应用程序创建一个 docker 镜像，并标记为`latest`，你可以使用下面的命令查看它*

*`docker images`*

*您可以将该映像推送到某个存储库(例如 Docker Hub)，然后在任何服务器(例如另一台机器或 Google Cloud 中的一台虚拟机器)上将其拉回，然后使用 docker compose 运行该应用程序。*

*为了[将图像推送到 Docker Hub](https://docs.docker.com/engine/reference/commandline/push/)*

```
*docker push sample-flask-project_flask_app:latest*
```

*然后，在你想运行的服务器上，你只需要复制`docker-compose`文件并注释掉 flask_app 的`build`部分，这里是完整的`docker-compose.yml`文件*

```
*version: "3"

networks:
  backend:
    driver: bridge

volumes:
  my-db:

services:
  mysql:
    image: mysql:8.0.22
    container_name: mydb
    ports:
      - "3306:3306"
    environment:
      MYSQL_DATABASE: 'db'
      *# So you don't have to use root, but you can if you like* MYSQL_USER: 'user'
      *# You can use whatever password you like* MYSQL_PASSWORD: 'password'
      *# Password for root access* MYSQL_ROOT_PASSWORD: 'password'
    networks:
      - backend
    expose:
      - '3306'
    volumes:
      - my-db:/var/lib/mysql

  flask_app:
*#    build:
#      dockerfile: Dockerfile
#      context: .
#      args:
#        BUILD_ENV: $BUILD_ENV* **image: sample-flask-project_flask_app:latest**
    container_name: flask_app
    expose:
      - '5000'
    environment:
      - FLASK_DEBUG=$FLASK_DEBUG
      - DB_URL=$DB_URL
    ports:
      - "5000:5000"
    network_mode: bridge*
```

*然后你只需要运行命令`docker-compose up -d`*

*(-d 表示它将作为守护进程在后台运行)。*

*以下是您可能会用到的有用的 docker 命令列表:-*

```
*# To see all the images on your system 
docker images# To build the images (if a build config is specified in docker-compose.yml)
docker-compose build --no-cache# To login to a running container, let's say we wan to see our database in mysql container named 'mydb'
docker exec -it mydb /bin/bash
# then you can run mysql command i.e mysql -u user -p*
```

*[这是目前为止的完整代码](https://github.com/trexsatya/sample-flask-app/tree/Step_6)。*

*第一个现在**安全***

*所以我们几乎完成了 flask 应用程序的基本设置。*

*还剩下什么？*

1.  *缓存。但是在 flask app 中添加缓存相当简单，内置了对 Memcache、simple cache、Redis cache 的支持。*
2.  *部署等基础设施。CI/CD 设置。但这对 Flask 应用程序来说是不可知的。*
3.  *一个重要的东西:安全。*

*为了保护应用程序代码中的应用程序，我们需要设置身份验证和授权。*

*我们将使用另一个库名`Flask-HTTPAuth`进行认证。该库支持基本身份验证、基于摘要的身份验证和基于令牌的身份验证。还有另外一个库名叫做 **flask_jwt_extended** 专门用于基于令牌的 auth，实际上有很多选项(见这里的)。*

*我们的自定义身份验证方案是基于社交登录的(目前只有 google 支持)，它是这样工作的:-
1。你从 google 获得访问令牌(*这可以很容易地通过在 UI 端集成 Google*)
2。访问令牌被发送到我们的应用程序进行注册，或登录到应用程序。*

*我们的应用程序验证令牌，然后创建自己的访问令牌，在用户登录时返回给用户。那么这个访问令牌必须作为访问 API 的承载令牌在报头中传递。*

*因此，首先让我们通过使 API 要求认证来保护它们。*

*`user_controller.py`文件的变化:-*

```
***@auth.login_required()**
def get(self, user_id):
    .....**@auth.login_required(role="admin")**
def post(self):
    .....*
```

*这里的`auth`是我们在*

*`app/extensions.py`*

```
***...****auth = HTTPTokenAuth('Bearer')***
```

*然后，我们需要为这个库提供一个函数，它可以验证请求头中的传入令牌。*

*我们在一个新的文件`app/services/auth_service.py`中这样做:-*

```
*from flask import request

from app.entities.user import User
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
import os
from datetime import timedelta

from app.extensions import auth

token_serializer = Serializer(os.getenv("SECRET_KEY"), expires_in=3600, algorithm_name="HS256")

class AuthService:
    @staticmethod
    def verify_token(token):
        try:
            data = token_serializer.loads(token)
        except Exception as e:  
            print("verify_token():", token, e, request.headers)
            return False
        return data

    @staticmethod
    def create_token(user: User):
        payload = {"email": user.email, "name": user.name, "roles": ["user"]}
        print("payload", payload)
        return token_serializer.dumps(payload).decode('utf-8')

@auth.verify_token
def verify_token(token):
    return AuthService.verify_token(token)

@auth.get_user_roles
def get_user_roles(user):
    print("get_user_roles", user)
    return user["roles"]*
```

*`verify_token`在调用处理程序方法之前，函数将在每个请求上被自动调用。*

*如果向`@auth.login_required`传递了名为`role`的参数，将自动调用`get_user_roles`。我们已经在`post`方法的`user_controller.py`文件中完成了这项工作(现在创建用户，需要管理员角色)。*(注意:代码中的角色已经静态分配)**

*现在，如果您运行测试，它们应该会由于 401 响应而失败。*

*接下来的步骤是相应地修改测试，并为登录和注册编写一个控制器。*

*增加安全性的另一个步骤是增加对 CORS 的限制。*

*更改`app/configs.py`文件中的配置类:-*

```
*class Config(object):
    *"""Base Configuration"""* APP_DIR = os.path.abspath(os.path.dirname(__file__))  # This directory
    PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, os.pardir))
    **CORS_ORIGIN_WHITELIST = [
        'http://0.0.0.0:4100',
        'http://localhost:4100',
        'http://0.0.0.0:8000',
        'http://localhost:8000',
        'http://0.0.0.0:4200',
        'http://localhost:4200',
        'http://0.0.0.0:4000',
        'http://localhost:4000',
    ]***
```

*看到目前为止的代码— [这里](https://github.com/trexsatya/sample-flask-app/tree/Step_6)。*

*认证/安全部分已经完成。*

*现在还剩下什么？*

*对弗拉斯克来说没什么。如果我们谈论一个项目设置，当然还有代码质量的实现/改进、CI/CD 的基础设施设置(可以在 pipeline 本身中添加对库进行漏洞检查的步骤)、架构模式、更多功能等等。*

*我们还可以改进 REST 的实现(我们也可以为 REST 实现 [HATEOAS](https://en.wikipedia.org/wiki/HATEOAS) )。有一个名为 [Flask-Restx](https://github.com/python-restx/flask-restx) 的库会有所帮助。*

*出于演示的目的，SQLAlchemy 和 ORM 缺少的一点就是建立关系，但这对于 SQLAlchemy 的 api 来说也很容易。*

*剩下的一件小事是日志记录。Flask 默认提供了一个记录器:*

```
*from flask import current_app ....
current_app.logger.info("log")*
```

*另一个小问题是单元测试(我们还没有编写任何复杂的单元)。Python 的`[Unittest](https://docs.python.org/3/library/unittest.html)`包(Python 附带的)非常丰富，也提供了模仿功能。*

*关于 Python 应用部署的一个主要问题是，在生产中我们必须使用 [WSGI 服务器](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface)(基本上是将请求转发给 Python 框架的 web 服务器)。有几个选项包括 Nginx，Gunicorn，NginxUnit 等。*