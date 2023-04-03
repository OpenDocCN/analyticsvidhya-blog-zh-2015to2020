# 用 Django REST 框架、React 和 JWT 认证的流行文化指南

> 原文：<https://medium.com/analytics-vidhya/the-pop-culture-guide-to-authentication-with-django-rest-framework-react-and-jwt-22ab67159e9b?source=collection_archive---------15----------------------->

你好。

[来源](https://giphy.com/gifs/mrw-top-escalator-Nx0rz3jtxtEre/links)

厌倦了所有的指南需要 500 个框架和依赖项来获得合理简单的工作吗？那么让我们把这些都扔掉，只使用我们需要的东西(但是加入一些愚蠢的流行文化参考)！在本指南中，我将带你了解开始设置一条简单的私人路线所需的最基本的东西，包括用户登录、注册和一些添加到酷炫的 h'API 视图中的功能！

如果你被卡住了，请查看[这里](https://github.com/shewitt93/DRF-React-auth-guide)

**要求**

*   Python 3.7 +版本
*   [反应](https://reactjs.org/versions/) 16.12
*   姜戈
*   优选地，本教程中的 VScode 将利用一些生活质量特性，例如终端
*   [Pip](https://github.com/pypa/pip) (将使你的 Python 安装 etc 更加容易)
*   对非常糟糕的迷因的容忍

你会发现缩写

*   JSON web 令牌( [JWT](https://jwt.io/) )
*   Django REST 框架( [DRF](https://www.django-rest-framework.org/) )

# 1。让这艘船离开地面

让我们从创建 Django 项目开始，创建一个项目文件夹，在代码编辑器中打开它，并在那里打开您的终端。插入下面的片段，我们就可以开始了！

让我们在这里安装我们的依赖项和我们的 pip 环境

```
pipenv installpipenv shellpipenv install django djangorestframework djangorestframework-jwt django-cors-headers please ensure you add the . at the end of the next part!django-admin startproject <project-name> .
```

然后安装这个 Javascript libr… jk jk 的列表，除非？

让我们编辑在<**project-name>**/settings . py 中找到的设置文件，我在添加的部分旁边写了一些简短的描述，这样你就知道每个部分是做什么的了！

快跑！([敦敦敦敦敦](https://www.youtube.com/watch?v=xdAyC1t5ZvA&ab_channel=NickS%C3%B8rensen))

```
p*ython manage.py runserver*
```

去 [http://localhost:8000](http://localhost:8000/) 然后…！

接下来让我们转到 urls.py，这里我们已经添加了一个 admin 和 token-auth 的路径。Admin 是 Django 的内置特性，非常有用！Django REST 框架为我们提供了非常有用的 JWT 路径，并且进一步简化了事情。这些都在这里，所以在我们刚刚设置的服务器上，我们可以有替代的 url 视图！

现在进入你的终端

```
*python manage.py migrate*
```

创建一个超级用户(这基本上是一个拥有完全管理权限的人，这是一件非常好的事情！)

```
python manage.py createsuperuser
follow the terminal instructions here
```

让我们访问我们创建的网址，玩一玩吧！

```
[http://localhost:8000](http://localhost:8000/)/token-auth/
```

输入您刚刚创建的超级用户的详细信息**瞧，你有一个 JWT 令牌了！**

这就是 REST 框架如此优秀的原因，它提供了许多内置特性，使得后端变得轻而易举！

现在我们需要允许其他人注册，这样他们也可以得到他们的！

# 2。包尾恶作剧

我们要去冒险了！让我们开始后端部分，我们有我们的序列化程序，模型和其他认证和数据类型的东西，所以让我们快速创建模板

```
*python manage.py startapp <project-name-auth>*
```

让我们将其添加到的“已安装的应用程序”部分

```
<project-name>/settings.py
```

> INSTALLED_APPS = [
> 
> …'<**project-name-auth**>. apps .<**project-name-auth**>Config '(你可以在<**project-name-auth**>的 apps.py 文件中找到这个，第一个字母经常是大写的，大概最好复制粘贴)
> 
> …]

否则你可能会遇到一个问题*我认识的一个人*遇到过，感到失落、困惑并且看不到任何错误…

![](img/bb982d0bd884a136a86e2bba7082632c.png)

让我们开始创建我们的序列化器，这样我们的用户模型就可以理解在 JSON 中来回传递的数据到“可消化的”python 大小的块。

在**<project-name-auth>**中创建名为 serializers.py 的文件

关于我添加的评论，有一些额外的内容，但这是你在这里需要的一切

所以让我们快速过一遍。我们为数据创建了两个序列化器，一个用于在我们登录后获取用户数据。第二个用于我们注册和登录时传递 JSON 令牌和用户数据。我们使用来自 ModelSerializer 抽象的某些内置调用。我们添加了 User 元类来告诉序列化程序它将处理哪个模型，以及我们希望解析其中的哪些字段。

我们还需要告诉它通过这个模型中的令牌，因为这不包括在“用户”类的内部工作中，所以我们创建 def get_token 方法来为我们处理这个问题，充分利用我们之前安装的 JWT 包。

我们还需要密码保持安全，首先要存储它，但也不要将它包含在返回的 JSON 中。我们开始时只写数据，并在字段中指定这是一个密码。现在我们添加。set_password 函数，确保密码通过散列和加盐得到适当的保护(此处添加一个注释，将密钥保存在. env 文件中的 settings.py 文件夹中，以确保它是隐藏的！)

让我们在 views.py 中给它一个杀戮的视角吧！

我觉得我需要解释这一切，所以系好安全带，孩子！因此，current_user 视图是我们成功登录时看到的视图(您可以在登录到管理站点后转到 localhost:8000/users，您应该会在这里看到您的姓名和详细信息，但只有在完成接下来的几个步骤之后)。由于我们的设置方式，这需要通过验证过程，以确保存在有效的 JWT 令牌，并允许我们的前端从该视图中“获取”数据(目前仅限于“获取”请求，但这是可以改变的)。)

接下来我们有我们的 UserList 类，这允许你将成功注册的数据发布到后端，以确认并存储用户到站点，我们已经将权限类设置为 any，以确保你不必注册就可以注册，那不是很奇怪吗！这将通过我们之前创建的序列化程序进行传递，在将该用户添加到确认列表之前确保其有效性。我们还在这里添加了一个 get 请求，这样当您访问 localhost:8000/user-list 时，您可以看到所有用户！

现在我们的网址显示你从上面**所做的所有努力！**

**在您的 **< project-name-auth >文件夹**中创建另一个名为 urls.py 的文件**

**让我们回到 urls.py 中的 **<项目名称>** 文件夹，并将它们作为视图添加到其中，这样我们就可以访问我们的 localhost:8000**

> **从 django.urls 导入路径，包括**
> 
> **urlpatterns = [**
> 
> **…path((<**项目名称授权**>/’)，include(<**项目名称授权** >)。网址'))**
> 
> **]**

**我们还要确保用户数据也随着登录信息一起被传递！**

**在 **<项目名称>** 文件夹中创建 utils.py，然后输入**

**现在将 settings.py 添加到 CORS 白名单下**

> **JWT _ AUTH = { ' JWT _ 响应 _ 有效载荷 _ 处理程序':'<**项目名称**>. utils . my _ jwt _ 响应 _ 处理程序' }**

# **对于这两者之间的一些化学反应，让我们让事情发生反应(ive)**

# **3。反应过来，最后的边疆？**

**让我们用快速的方法来做这件事，但是如果你有使用 React 的经验，只要将它整合到你现有的设置中，你就可以开始了！**

**在根目录中(或文件夹的顶层)**

```
**npx create-react-app "name of app"**
```

**因此，让我们创建我们的注册页面，在 src 部分创建一个 components 文件夹，然后在那里添加我们的 registration.js 文件！**

**在这里，您可以看到一个标准的输入表单，它接受用户的电子邮件、姓名和两个密码必须相同的条件，以防止任何意外(我还添加了一些错误处理，只是为了让您对这种情况有所了解)。这被设置为状态，然后数据被发送回我们的“用户”API，它将存储这个用户。注册后，如果您转到 localhost:8000/users/上的视图，您应该会在那里看到我们的用户。**

**现在我们的登录页面！**

**同样是一个标准的输入表单，但是这一次我们将数据提交到我们的/ **token-auth** API。如果数据是正确的，它将接受输入(就像我们之前创建的超级用户一样)并给我们*珍贵的，*我是说我们的令牌。一旦成功，我们就将这个令牌添加到本地存储中，这样我们将要创建的函数就可以实现路由私有化了！有关于这种方法安全性的讨论，但那绝对是另一个时间和地点的讨论。**

**现在，我们需要使我们的组件能够环绕路由，以便只有那些拥有有效 JWT 令牌的用户才能访问它们。**

**在这里，我们有效地呈现了我们指定的任何组件，但前提是您已经成功登录，否则您将被重定向到您想要放在这里的任何路由(在我的例子中是登录页面)。PrivateRoute 现在可以像这样环绕任何你想要的路线…**

```
**<PrivateRoute path=”/home” component={home} />**
```

**…将登录和注册页面添加到 app.js 文件，并在每个路由中呈现该组件，如下所示:**

**让我们对 index.js 文件进行快速编辑**

**以及一个快速的 npm 安装来获得我使用过的依赖项，react-router-dom 用于浏览器路由器和链接特性**

```
**npm install react-router-dom in the react folder**
```

**现在进入最后一部分，创建我们的 isLogin 函数，你可以在上面的要点中看到**

**这里，我们从本地存储中获取令牌，确保字段不为空或未定义，然后如果上述操作成功，则添加布尔值 true，从而允许访问 PrivateRoute。对于注销，创建一个从本地存储中删除令牌的函数，并将它附加到一个按钮上(但我会让您自己解决这个问题)。**

**现在使用 **npm** **start** 然后在浏览器中打开 localhost:3000。尝试在 localhost:3000/home 上访问你的 home 组件，不能！登录并注册，然后你就可以了！**

**现在，如果您想将用户数据拉至前端，只需向我们之前创建的 current_user API 视图发出一个获取请求，如下所示**

**恭喜你。Django 后端和 React 前端认证！希望这让你的事情变得简单了。肯定有很多东西可以玩，也有更多的阅读要做，但是这应该可以满足你的需要，直到你开始接近*危险地带*！**

**![](img/fef68e2b899378070b2eaae06934babe.png)**

**来源**

**[](/@dakota.lillie/django-react-jwt-authentication-5015ee00ef9a) [## Django & React: JWT 认证

### 最近，我一直在开发一个应用，前端使用 React，后端使用 Django。我没有…

medium.com](/@dakota.lillie/django-react-jwt-authentication-5015ee00ef9a)  [## Home - Django REST 框架

### Django REST 框架是一个用于构建 Web APIs 的强大而灵活的工具包。您可能想使用 REST 的一些原因…

www.django-rest-framework.org](https://www.django-rest-framework.org/)**