# Django ORM 内部——环境设置

> 原文：<https://medium.com/analytics-vidhya/django-orm-internals-environment-setup-42cf66dff5b2?source=collection_archive---------20----------------------->

## **背景:**

在本文中，我们将了解 Django ORM 是如何工作的。根据专家的说法，如果我们想学习一些新的东西，最好的方法是，阅读一些现有实现的代码库。
如果你正在开发一些基于 Python 的 ORM &那么你就不会去研究 Django 代码，要么你错过了什么，要么你远远超出了像我这样的普通开发人员。
我想在开始的时候澄清一下，这不是关于任何使用 Django 框架的 web 应用程序开发。我正在学习任何 ORM 是如何工作的&阅读 Django 代码是最好的起点。这将是一系列的职位。如果我们正在寻找一个能够帮助使用 Django 唯一的 ORM 特性的指南，这可能会有所帮助。

## **工具:**

因为目的是了解 Django ORM 的内部，所以第一步应该是从它的开发部门获得源代码。因为每个项目都有一个虚拟环境是一个很好的实践，所以我们将使用 conda 来管理我们的虚拟环境。为了使这篇文章更短，我们不再详细介绍 conda 的安装过程。由于虚拟环境帮助我们隔离 python 开发环境，docker 可以以类似的方式帮助我们处理其他应用程序，因此对于数据库应用程序，我们将使用 docker。同样，我们可以从互联网上的一些其他资源开始使用 docker 和 docker composer。最后，关于 IDE，我正在使用 Pycharm 社区版。

## 设置:

那么，我们已经讨论了正在使用的工具，现在让我们按照步骤开始作为一个 ORM 开发者。

**创建并激活虚拟环境**

```
*~$ conda create -n django-dev python=3.9
~$* conda activate django-dev
```

**从 Github 克隆最新的 Django 源代码**

```
~$ git clone [https://github.com/django/django.git](https://github.com/django/django.git)
~$ cd django
```

**从源代码安装 Django**

```
~$ python -m pip install . # make sure you are inside the django dir
```

**为 ORM 实验**创建一个项目& App

```
~$ mkdir experiment && cd experiment
~$ django-admin startproject ormtest .
~$ python manage.py startapp modeltest
```

**修改设置文件以满足我们的要求**

默认情况下，django 将创建一个 web 应用程序可以使用的设置文件。因为我们只需要数据库连接部分，所以我们将删除其他不相关的配置。最终的设置文件将如下图所示。这里我附上了 docker-composer 文件，以供参考。docker 容器可以通过下面的命令启动。

```
~$ sudo docker-compose up # in the dir with compose.yml 
```

**最后一步**

我们几乎完成了设置。我们可以使用下面的示例运行一个基本的测试代码。在下面的示例中，假设我们在 modeltest 的模型文件中有一个模型“Device”。

```
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ormtest.settings')

application = get_wsgi_application()

from modeltest.models import Device

if __name__ == "__main__":
    d1 = Device.objects.filter(device_id=1)
    print(d1[0].name)
```

这将是一系列的职位。所以请继续关注，我们将了解 ORM 是如何工作的！