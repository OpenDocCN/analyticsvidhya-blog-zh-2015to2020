# 学习烧瓶框架[第 1 部分]

> 原文：<https://medium.com/analytics-vidhya/learning-flask-framework-part-1-13be25f30fa6?source=collection_archive---------24----------------------->

当你试图学习 flask 时，这些笔记可能会帮到你。

> "无欲无求的学习会破坏记忆，它不会保留任何它所吸收的东西."
> —达芬奇

我将不再只是跟随一个教程，而是同时尝试做一个项目，你可以在这里找到。如果你想一起去，欢迎你来！

## 烧瓶是什么？

[Flask](https://flask.palletsprojects.com/en/1.1.x/quickstart/) 是一个以 python 为基础开发的微型 web 框架。它是由 Pocoo 的阿明·罗纳切尔创建的，Pocoo 是一个 Python 爱好者的国际组织，成立于 2004 年，作为对 [Werkzeug](https://palletsprojects.com/p/werkzeug) 和 [Jinja](https://palletsprojects.com/p/jinja) 的简单包装，它已经成为最受欢迎的 Python web 应用程序框架之一。

## **先决条件**

1.  Python 3
2.  通过 pip 安装软件包
3.  对 HTML 和 CSS 有基本的了解

## 安装烧瓶

安装 flask 相当简单，因为它没有依赖项。你必须使用 conda 或 venv 创建一个虚拟环境。你可以在这里查阅原始文档[。](https://flask.palletsprojects.com/en/1.1.x/installation/#install-flask)

```
pip install **flask**
```

好吧！我们完成了烧瓶的安装。我们来编码吧！

## 创建应用程序

```
# app.py
from flask import Flask
app = Flask(__name__)[@app](http://twitter.com/app).route("/")
def home():
    return "Hello World!"
if __name__ == '__main__':
    app.run(debug = True)
```

安装 flask 后，转到您的项目目录，创建一个名为 **app.py** 的文件，粘贴上面的代码并执行它。您可以在 [***查看结果 http://127 . 0 . 0 . 1:5000/***](http://127.0.0.1:5000/)*或将 cmd 中显示的 ip 地址复制到您的浏览器中。浏览器中的输出将是“hello world ”,如 hello 函数中所示。*

*让我们检查一下代码，带有大写字母“F”的 Flask(如:`from flask import **Flask**`)创建了一个对象，它引用了整个应用程序本身:当我们声明`app = Flask(__name__)`时，我们正在创建代表我们的应用程序的变量 app。因此，当我们配置变量应用程序时，我们正在配置整个应用程序的工作方式。例如，设置`app = Flask()`可以接受几个属性:*

```
*from flask import Flask
app = Flask(__name__,
            instance_relative_config=False,
            template_folder="templates",
            static_folder="static")*
```

*这些属性是配置文件的位置、存储 HTML 页面模板的 template_folder 和存储前端资产(Js、CSS、供应商、图片等)的 static_folder。)*

*`@app.route("/")`是一个 [Python 装饰器](http://book.pythontips.com/en/latest/decorators.html), Flask 提供它来轻松地将我们应用中的 URL 分配给函数。装饰者告诉我们的`@app`,每当用户在给定的`.route()`访问我们的应用程序域(*myapp.com*或*http://127 . 0 . 0 . 1:5000/)*时，执行`home()`函数。
每个 web 框架都是从在给定 URL 提供内容的概念开始的。*路线*指一个 app 的 URL 模式(如*myapp.com****/首页*** 或*myapp.com/****关于*** )。*视图*指的是在这些 URL 上提供的内容，无论是网页还是 API 响应等。我们可以用一个函数处理多个路由，只需要在任何路由上叠加额外的路由装饰器！下面是提供相同的“Hello World！”的有效示例 3 条不同路线的消息:*

```
*[@app](http://twitter.com/app).route("/")
[@app](http://twitter.com/app).route("/home")
[@app](http://twitter.com/app).route("/index")
def home():
    return "Hello World!"*
```

## ***路由 HTTP 方法***

*`@app.route()` decorator 还接受了第二个参数:一个已接受的 HTTP 方法列表[。默认情况下，Flask 视图接受 GET 请求，除非我们明确列出视图应该具有哪些 HTTP 方法。最好传递一个名为 methods 的参数，它包含我们需要的 HTTP 方法列表。](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)*

```
*[@app](http://twitter.com/app).route("/example", methods=['GET', 'POST', 'PUT'])*
```

## ***动态路线***

*让我们来谈谈基于一个概要名称或一篇文章的标题来动态改变 URL。为此，我们在尖括号`<>`内创建一个变量，并将其传递给`route()`*

```
*[@app](http://twitter.com/app).route('/<error>')
def pageNotFound(error):
    return ('Page '+ error +' not found! Please check the url!')
[@app](http://twitter.com/app).route('/loggedin/<username>')
def profile(username):
    return (username +' has logged in')*
```

*在提到变量的类型并使用冒号将其与变量名分开后，也可以传递特定类型的变量。*

```
*[@app](http://twitter.com/app).route('/<int:year>/<int:month>/<title>')
def article(year, month, title):
    return (title + 'was published in' + month + year)*
```

## ***渲染 HTML 页面***

*Flask 使用 jinja2 模板渲染网页，要做到这一点我们必须先导入 Flask 的一个内置函数`**render_template**`
`**render_template()**`用于返回 html 页面到一个路由。*

```
*from flask import Flask, render_template
app = Flask(__name__,
           template_folder="templates")*
```

*现在让我们创建一个名为`**index.html**`的 HTML 文件，并粘贴下面的代码。*

```
*<!DOCTYPE html>
<html>
<head>
 <title>
  Flask Tutorial
 </title>
</head>
<body>
 <h1> HTML template Works! </h1>
</body>
</html>*
```

*`**index.html**`有一个基本的 html 布局，带有一个包含一些文本的标题< h1 >。让我们更改 app.py 中的代码来渲染`**index.html**`复制下面的代码并粘贴到`**app.py**`*

```
*from flask import Flask, render_template
app = Flask(__name__,
            template_folder="templates")[@app](http://twitter.com/app).route("/")
def home():
    """Serve homepage template."""
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug = True)*
```

*我们还可以将*值*作为关键字参数传递给模板。例如，如果我们想通过视图设置模板的标题和内容，而不是硬编码到模板中，我们可以这样做:*

```
*from flask import Flask, render_template
app = Flask(__name__,
            template_folder="templates")[@app](http://twitter.com/app).route("/")
def home():
    return render_template('index.html',
                           title='Flask-Login Tutorial.',
                           body="You are now logged in!")*
```

*让我们暂时结束它，我将在这里连接我的下一部分。*

*使用的一些资源有:*

1.  *[blog.miguelgrinberg.com](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)*
2.  *[官方烧瓶文件](https://flask.palletsprojects.com/en/2.0.x/quickstart/)*
3.  *[黑客和黑客](https://hackersandslackers.com/flask-routes/)*

*[](https://github.com/BlankRiser) [## 空白冒口-概述

### 在 GitHub 上注册你自己的个人资料，这是托管代码、管理项目和构建软件的最佳地方…

github.com](https://github.com/BlankRiser)  [## 档案

### 你好，世界，我是拉姆·尚卡尔。一个有创造力的开发者和一个有逻辑的设计师。我热衷于打造优秀的…

blankriser.github.io](https://blankriser.github.io/personalDossier/)*