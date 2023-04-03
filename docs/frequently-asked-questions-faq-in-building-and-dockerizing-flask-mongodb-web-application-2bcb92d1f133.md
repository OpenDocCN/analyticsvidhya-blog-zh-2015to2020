# 构建和对接 Flask-MongoDB Web 应用程序的常见问题(FAQ)

> 原文：<https://medium.com/analytics-vidhya/frequently-asked-questions-faq-in-building-and-dockerizing-flask-mongodb-web-application-2bcb92d1f133?source=collection_archive---------27----------------------->

![](img/1375098580d743cd9855d565ef3bc6be.png)

Jules Bss 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

有一个主题的常见问题总是很有帮助的。

我们的文章是一个简单的 FAQ，包含了用 flask 和 mongoDB 构建和编写 web 应用程序时常见的令人困扰的错误(FDE)。

> 开发者永远不会满足于单一的框架。他们正在寻求将多种技术融入他们的学习堆栈中。

# 常见问题解答

## 为什么是烧瓶？

Flask 是一个构建 web 应用程序的简单 web 框架。

Flask 不像 Django 那样是全栈 web 框架。但是它是一个依靠第三方库来服务的微观框架。

有了 flask，你将有很好的学习机会和对你的应用程序的控制，例如，构建功能的方式。

## 为什么是 MongoDB？

我们日常生活中的数据没有很好的组织。所以我们的数据库必须准备好处理不同的数据类型。MongoDB 是一个面向文档的数据库，为高效的数据处理提供了多种选择。

## 为什么要码头化？

Docker 容器维护运行应用程序的完整环境。

有了 docker，你可以用更少的资源运行任何版本的 python 和任何操作系统的组合。

## PyMongo vs MongoEngine

PyMongo 返回 JSON 格式的输出或字符串引用的字典。

MongoEngine 允许您创建一个模式来更好地处理数据。这使得你的代码清晰。

所以比起 pymongo 我更喜欢 MongoEngine。我将在文章的最后提供一个 github 链接，链接到用 mongo engine 开发的 web 应用程序。

# 场减速器(Field Decelerator)

## 缩进误差

尽管 python 中没有分号和大括号，但一个简单的空格在这里也能派上用场。

使用一个好的 IDE 来避免这个错误。

## 无模函数

“pip 安装<module>”是解决没有找到模块错误的终极答案。</module>

## 找不到路线

在你的“routes.py”中完美提供路线。

```
[@app](http://twitter.com/app).route("/logout")
def logout():
    flash(f"{session.get('username')}, you are successfully logged   
              out!", "success") session.pop('username',None)
    session.pop('role',None)
    return redirect(url_for('login'))
```

## 渲染模板错误

渲染模板时最常见的错误是在 render_template 前缺少“return”。

将所有必需的参数传递给模板。

```
return render_template("customer/delete_customer.html",form=form,title="Delete customer")
```

## 不起作用的 URL

使用 url_for 函数引用路由，而不是将 url 写入字符串。

```
return redirect(url_for('index'))
```

## 重复的函数名

每个回调函数都有一个唯一的函数名。

以下代码将引发错误

```
@app.route('/in')
def fun():
    ....@app.route('/out')
def fun():
    ....
```

## 无法连接到服务器

如果你在本地用数据库运行你的应用，那么确保在另一个终端运行“mongod”。

如果是最新版本的 Mac，你会得到一个简单的“mongod”错误。

```
cd Documents(or any other preferred location)
mkdir data
cd data
mkdir db
cd ..
cd..
mongod --dbpath ~/documents/data/db
```

## 密钥丢失

请确保将您的密钥设置到您的 flask 应用程序，以便进行有效的会话管理。

```
#config.py
class Config(object):
    SECRET_KEY=os.environ.get("SECRET_KEY") or "secret_string"#app.pyapp = Flask(__name__)
app.config.from_object(Config)
```

## 使用中的端口

要检查正在运行的端口，请使用以下命令

```
sudo lsof -iTCP -sTCP:LISTEN -n -P
```

使用“sudo kill <id>”终止 id 为的 mongod 进程。现在用“mongod”命令重新启动。</id>

## 停靠后的意外行为

别忘了链接 docker-compose.yml 里面的 db。

另外，确保在 docker-compose.yml 的卷中指定正确的目录名。

```
#Dockerfile FROM python:3.7
ADD . /bank
WORKDIR /bank
RUN pip install -r requirements.txt#docker-compose.ymlweb:
  build: .
  command: python -u main.py
  ports:
    - "5000:5000"
  volumes:
    - .:/bank
  links:
    - db
db:
  image: mongo:4.2.5
```

## 对接时出现端口错误

如果对接后端口有任何错误，用以下代码修改

```
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
```

## 停靠后数据库的连接问题

确保 config.py 中有 mongodb 设置，如下所示

```
#config.pyclass Config(object):
    SECRET_KEY=os.environ.get("SECRET_KEY") or "secret_string"
    MONGODB_SETTINGS = { 'db' : 'MY_BANK' ,'host' :   
         os.environ["DB_PORT_27017_TCP_ADDR"] , 'port': 27017}#app.pyapp = Flask(__name__)
app.config.from_object(Config)db = MongoEngine()
db.init_app(app)
```

## 模板中缺少标记

确保已经为模板中的条件语句和循环给出了结束标记。

```
{{ if k }}
....
{{ endif }}
```

## 丢失的 CSRF 令牌

CSRF 令牌由服务器端应用程序生成，并发送给客户端，包含在请求中。

您将看不到应用程序的预期行为。要知道发生了这个错误，需要打印“form.errors”。

```
#include following anywhere in your template{{ form.hidden_tag() }}
```

# 完整的 Web 应用程序

这是一个用 flask 和 mogodb 制作的完整的 dockerized bank web 应用程序的链接。

[](https://github.com/maheshsai252/bank-flask) [## maheshsai252/bank-flask

### 一个基本的 flask web 应用程序，可以执行所有的银行操作。

github.com](https://github.com/maheshsai252/bank-flask) 

感谢您的阅读:)