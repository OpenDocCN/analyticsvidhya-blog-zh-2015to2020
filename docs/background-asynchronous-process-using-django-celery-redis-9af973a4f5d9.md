# 使用 Django+芹菜+Redis 的后台异步过程

> 原文：<https://medium.com/analytics-vidhya/background-asynchronous-process-using-django-celery-redis-9af973a4f5d9?source=collection_archive---------3----------------------->

在 Web 应用程序开发中创建后台异步进程是不可避免的，芹菜使整个过程变得简单和容易。这个博客的主要重点是使用 Django 和芹菜创建一个后台进程，Redis 作为消息代理。

**1。下载并安装 Redis**

您可以从[](https://redis.io/download)**下载并安装 Redis。*确保 Redis 已启动，并且 Redis 服务器正在运行。*

> *注意:您系统中安装的 Redis 版本应该大于 3.0*

***2 .基本安装***

*作为第一步，创建一个虚拟环境并安装以下内容:*

```
*$ virtualenv scheduler-env
$ cd scheduler-env
$ source bin/activate
$ pip install Django==3.1.3
$ pip install celery==5.0.0
$ pip install django-celery-beat==2.1.0
$ pip install django-celery-results==2.0.0
$ pip install redis==3.5.3*
```

*3.**创建姜戈项目和应用程序，添加基本配置***

```
*$ django-admin startproject testproj
$ cd testproj
$ django-admin startapp testtasks*
```

*在 testproj/testproj/settings.py 中添加以下内容*

```
*#testproj/testproj/settings.pyINSTALLED_APPS += [
    'django_celery_beat',
    'django_celery_results',
    'testtasks'
]
CELERY_RESULT_BACKEND = "django-db"
CELERY_IMPORTS = ('testtasks.tasks')*
```

*4.**运行迁移***

```
*python manage.py makemigrations
python manage.py migrate*
```

*5.**添加芹菜，更新项目配置。***

*在 path testproj/testproj 中创建一个文件芹菜. py，并将以下内容添加到该文件中。*

> *注意:将文件中的“ **testproj** ”替换为您的项目名称*

```
*#testproj/testproj/celery.pyimport os
from celery import Celery#set the default django settings to celery
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'testproj.settings')app = Celery('testproj')app.config_from_object('django.conf:settings', namespace='CELERY')# Load task modules from all registered Django app configs.
app.autodiscover_tasks()#message broker configurations
BASE_REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')app.conf.broker_url = BASE_REDIS_URL#creating a celery beat scheduler to start the tasks
app.conf.beat_scheduler = 'django_celery_beat.schedulers.DatabaseScheduler'*
```

*更新 testproj/testproj/__init__ 中的芹菜项目配置。巴拉圭*

```
*#testproj/testproj/__init__.py
from .celery import app* 
```

*6.**创建任务***

*在 testproj/testtasks 路径中创建一个任务文件“tasks.py ”,并添加以下内容。如果您已经注意到，此任务. py 将在设置. py 中作为芹菜导入。*

```
*#testproj/testtasks/tasks.pyimport random
from celery import Celery
from testproj.celery import app[@app](http://twitter.com/app).task(name='tasks.add')
def add(x, y):
    return x + y*
```

*7.**验证任务***

*打开一个终端并启动工作进程，*

```
*$ celery --app=testproj worker -l INFO*
```

*在另一个终端，打开外壳*

```
*$ python manage.py shell*
```

*在 shell 中执行以下操作，*

```
*>>> from testtasks.tasks import *
>>> add.delay(10,20)
<AsyncResult: 2f7057cb-7c2f-49b8-9f9b-640d54a62b6e>*
```

*当任务被调用时，您可以看到工作进程执行任务，并在工作进程运行的终端中给出结果。*

*8.**创建周期性任务***

*您还可以创建在特定时间间隔内执行任务的定期流程。让我们创建一个周期任务，每 3 秒运行一次。*

*在芹菜中加入下列物质*

```
*#testproj/testproj/celery.pyapp.conf.beat_schedule = {
    'add-every-3-seconds': {
        'task': 'tasks.add',
        'schedule': 3.0,
        'args': (16, 16)
    },
}
app.conf.timezone = 'UTC'*
```

*上面的代码片段每 3 秒运行一次任务“add”。您可以通过在单独的终端中运行工人和节拍来检查以下内容。您可以看到芹菜节拍启动任务，工作进程执行任务。*

> *注意:将“testproj”更改为您的项目名称*

*终端 1-工作进程*

```
*$ celery --app=testproj worker -l INFO*
```

*2 号航站楼-芹菜段*

```
*$ celery -A testproj beat -l INFO --scheduler django_celery_beat.schedulers:DatabaseScheduler*
```

***参考文献:***

1.  *[*https://docs . celeriproject . org/en/latest/django/first-steps-with-django . html*](https://docs.celeryproject.org/en/latest/django/first-steps-with-django.html)*
2.  *[*https://docs . celerrproject . org/en/last/user guide/tasks . html*](https://docs.celeryproject.org/en/latest/userguide/tasks.html)*
3.  *[*https://docs . celery project . org/en/stable/user guide/periodic-tasks . html*](https://docs.celeryproject.org/en/stable/userguide/periodic-tasks.html)*