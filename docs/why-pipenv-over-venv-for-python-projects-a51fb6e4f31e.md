# 为什么 Python 项目要用 pipenv 而不是 venv？

> 原文：<https://medium.com/analytics-vidhya/why-pipenv-over-venv-for-python-projects-a51fb6e4f31e?source=collection_archive---------4----------------------->

![](img/42ea55291a9002887bd3ec2549b6765e.png)

默认情况下，系统上的每个 Python 项目都将使用这些相同的目录来存储和检索站点包(第三方库)。
但是您可能会发现，在您的本地机器上管理针对不同 Python 版本的具有不同依赖关系的不同项目是非常痛苦的。

在这种情况下，*虚拟环境*的概念就来帮助我们了。如果你没有使用过或者不知道 Python 中的虚拟环境，让我先为你定义一下。

> 虚拟环境是一种工具，通过为不同项目创建隔离的 Python 虚拟环境，有助于保持不同项目所需的依赖关系分离。

因此，基本上安装在本地机器上的 Python( *父*)会在您的机器上产生另一个 Python( *子*)，您可以将这个环境用于一个特定的项目。您可以在这个虚拟环境中安装或卸载依赖项，而不会影响安装在您的*父* Python 中的依赖项。

创建虚拟环境的一种常见方式是使用`venv`。 *venv* 模块预装了 Python 3.5+版本。下面列出了一些使用 *venv* 创建虚拟环境的重要命令:

## 创建虚拟环境

```
python -m venv environment_name
```

## 激活虚拟环境

*   对于 **MacOS** 和 **Linux** 用户:

```
venv environment_name/bin/activate
```

*   对于 Windows 用户，首先将您的目录更改为新创建的环境文件夹，然后运行以下命令:

```
..\environment_name\Scripts\activate
```

## 停用虚拟环境

```
deactivate
```

## 检查虚拟环境中安装的所有依赖项

*   首先激活虚拟环境，然后使用以下命令查看该虚拟环境中安装的依赖项列表:

```
pip freeze
```

*   生成一个 *requirements.txt* 文件，该文件包含所有与它们各自版本的依赖关系:

```
pip freeze > requirements.txt
```

## 从 requirements.txt 文件安装所有依赖项

```
pip install -r requirements.txt
```

现在让我们转到 [**Pipenv**](https://pipenv.pypa.io/en/latest/) ，这已经成为管理项目依赖关系的推荐方法。代替在你的项目中有一个`requirements.txt`文件，并管理虚拟环境，我们现在将在我们的项目中有一个`Pipfile`自动做所有这些事情。如果您曾经使用过 Node.js，您可能会熟悉 *package.json* 和 *package-lock.json* 文件。 *Pipfile* 和 *Pipfile.lock* 文件与 Python 中的类似。

## 安装管道

让我们首先在本地机器上安装 *Pipenv* 。使用以下命令来完成此操作:

```
pip install pipenv
```

## 第一次使用 pipenv

```
pipenv install
```

上面的命令将寻找一个 *Pipenv* 文件。如果它不存在，它会创造一个新的环境并激活它。运行上述命令后，我们会发现两个新文件: *Pipenv* 和 *Pipenv.lock*

## 激活已经创建的 pipenv 环境

```
pipenv shell
```

## 安装/卸载依赖项

```
pipenv install djangopipenv uninstall django
```

## 安装开发依赖项

```
pipenv install nose --dev
```

## 从 requirements.txt 安装

```
pipenv install -r requirements.txt
```

## 检查安全漏洞

```
pipenv check
```

## 检查依赖图

```
pipenv graph
```

## 运行自定义脚本

与 npm 类似，我们也可以使用 pipenv 运行定制脚本。假设我们正在进行一个 Django 项目，我们必须使用命令:`python manage.py runserver` *来运行服务器。*现在我们可以将该命令添加到我们的 *Pipfile* 中，目前，我们的 *Pipfile* 可能如下所示:

```
[[source]]
url = "[https://pypi.org/simple](https://pypi.org/simple)"
verify_ssl = true
name = "pypi"[packages]
django = "*"[dev-packages]
nose = "*"[scripts]
server = "python manage.py runserver"[requires]
python_version = "3.8"
```

在上面的输出中，我们可以看到我们已经在 scripts 部分添加了脚本。因此，现在我们可以简单地运行下面的命令来运行我们的服务器:

```
pipenv run server
```

## 锁定依赖关系

```
pipenv lock -r
```

Pipfile.lock 文件通常如下所示:

```
{
    "_meta": {
        "hash": {
            "sha256": "627ef89...64f9dd2"
        },
        "pipfile-spec": 6,
        "requires": {
            "python_version": "3.8"
        },
        "sources": [
            {
                "name": "pypi",
                "url": "[https://pypi.org/simple](https://pypi.org/simple)",
                "verify_ssl": true
            }
        ]
    },
    "default": {
        "django": {
            "hashes": [
                "sha256:acdcc1...ab5bb3",
                "sha256:efbcad...d16b45"
            ],
            "index": "pypi",
            "version": "==3.1.1"
        },
        "pytz": {
            "hashes": [
                "sha256:a061aa...669053",
                "sha256:ffb9ef...2bf277"
            ],
            "version": "==2020.1"
        }
    },
    "develop": {}
}
```

# 结论

公平地说， *Pipenv* 在为您的包安装所有必需的子依赖项时，表现就像 pip 一样。但是一旦您解决了这个问题， *Pipfile.lock* 会为每个环境跟踪您的应用程序的所有相互依赖关系，包括它们的版本，因此您基本上可以忘记相互依赖关系。

在实践中，这意味着你可以继续在开发中工作，直到你有了一套适合你的包/版本。现在您可以简单地发出`pipenv lock`和 *Pipenv* 将锁定您的项目需要的所有依赖/相互依赖，锁定它们的版本并散列结果，这样您就可以在生产中确定性地复制您的构建。

最后，我的建议是**停止在虚拟环境中安装 pip，开始安装 pipenv。相信我，你不会后悔的。**

我把一些与 *pipenv* 相关的基本命令放到了 Github Gist 中。你可以在这里参观它们或者去 http://srty.me/pipenv 参观。如果你喜欢这个帖子，请鼓掌或主演 [Github Gist](http://srty.me/pipenv) 来支持。你可以在我的[网站](http://srty.me/ashutosh)联系我。
以上就是我的侧面，感谢阅读！