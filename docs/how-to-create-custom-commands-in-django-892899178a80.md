# 如何在 django 中创建自定义命令

> 原文：<https://medium.com/analytics-vidhya/how-to-create-custom-commands-in-django-892899178a80?source=collection_archive---------17----------------------->

在开发 django 项目时，需要编写一次性的脚本来自动完成特定的任务。在我们继续实现之前，这里有一些我发现自己正在应用的用例。

1.  一起清理错误的数据列。
2.  在一个[多租户应用](https://lewiskori.com/series/intro-to-multi-tenant-apps-with-django/)中迁移多个模式

在 django 中，有两种方法可以运行这些类型的命令。编写一个普通的 python 脚本，然后可以通过运行来调用它，另一种方法是利用 django-admin 命令。这些都是通过调用`python manage.py command_name`运行的。

对于这篇文章，我将用一个只有 3 个数据库表的博客应用程序来演示，用户、类别和文章。我假设你对初始化 django 项目很熟悉，但是如果你不熟悉，这篇文章应该能帮到你。

这篇文章的源代码可以在[这里](https://github.com/lewis-kori/django-commands)找到。

# 普通 python 脚本方法

对于第一个示例，我们将尝试使用下面的脚本列出所有系统用户

```
from django.contrib.auth import get_user_model

User = get_user_model()

# retrieve all users
users = User.objects.all()

# loop through all users
for user in users:
    print(f'user is {user.get_full_name()} and their username is {user.get_username()}')
```

您可以将脚本命名为 list_users.py，并通过`python list_users.py`运行它

一旦运行这个程序，就会遇到一个错误，

`django.core.exceptions.ImproperlyConfigured: Requested setting AUTH_USER_MODEL, but settings are not configured. You must either define the environment variable DJANGO_SETTINGS_MODULE or call settings.configure() before accessing settings.`

有人可能会认为，既然您在 django 的项目目录中，脚本运行起来不会有任何问题。然而，事实并非如此。这是因为脚本不知道该脚本将应用于哪个项目。您可以在一台机器或虚拟环境中拥有多个项目。所以给剧本一些背景是很重要的。

我们将通过稍微修改我们的脚本来做到这一点。

```
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'projectname.settings')

import django
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

users = User.objects.all()

for user in users:
    print(f'user is {user.get_full_name()} and their username is {user.get_username()}')
```

在这里，我们指定项目的设置，不仅如此，调用`django.setup()`方法。该方法配置设置、日志记录并填充应用程序注册表。简而言之，我们让脚本知道我们的项目环境。

如果你对操作系统模块感兴趣，我的[上一篇文章](https://lewiskori.com/blog/how-to-clear-screen-in-python-terminal/)提供了更多的见解。

**请注意，导入顺序很重要，必须保持不变。**

如果我们再次运行这个脚本，我们所有的用户都应该被打印到终端上👯‍♂️.

接下来，我们将通过运行`django-admin startapp posts`初始化一个名为 posts 的应用程序。

该应用程序将容纳我们的博客文章模型。

对于这个例子，我们将从命令行创建一个博客文章的实例。初始化一个脚本，并将其命名为`create_post.py`

```
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'commands.settings')

import django
django.setup()

from django.contrib.auth import get_user_model
from posts.models import Category, Post

User = get_user_model()

def select_category():
    # retrieve categories. (You can create some examples from the django admin)
    categories = Category.objects.all().order_by('created_at')
    print('Please select a category for your post: ')
    for category in categories:
        print(f'{category.id}: {category}')
    category_id = input()
    category = Category.objects.get(id=category_id)
    return category

def select_author():
    # retrieve all users
    users = User.objects.all()
    print('Please select an author for your post: ')
    for user in users:
        print(f'{user.id}: {user}')
    user_id = input()
    user = User.objects.get(id=user_id)
    return user

def create_post():
    title = input("Title of your post: ")
    content = input("Long post content: ")
    category = select_category()
    author = select_author()
    Post(**locals()).save()
    print('Post created successfully!')

if __name__ == "__main__":
    create_post()
```

这里，我们创建了一个博客文章的实例。注意我们如何处理外键关系？确保将相关数据库表的对象实例分配给该字段。

通过运行 python create_post.py，系统会提示我们进行一些输入。

# 编写自定义 django 管理命令的方法

如前所述，django-admin 命令是通过运行`python manage.py command_name`来执行的，例如`runserver, migrate and collectstatic`。要获得可用命令的列表，请运行`python manage.py help`。这将显示可用命令的列表以及它们所在的 django app 文件夹。

要注册自定义管理命令，请在 django app 文件夹中添加一个`management\commands`目录。在我们的例子中，它将位于 posts \ management \ commands 中。

一旦设置完成，我们就可以初始化 commands 文件夹中的自定义脚本了。对于第一个示例，我们将编写一个命令，将之前创建的博客文章标记为已发布。

为此，创建一个文件并命名为`publish_post.py`

```
from django.core.management.base import BaseCommand, CommandError
from posts.models import Category, Post

class Command(BaseCommand):
    help = 'Marks the specified blog post as published.'

    # allows for command line args
    def add_arguments(self, parser):
        parser.add_argument('post_id', type=int)

    def handle(self, *args, **options):
        try:
            post = Post.objects.get(id=options['post_id'])
        except Post.DoesNotExist:
            raise CommandError(f'Post with id {options["post_id"]} does not exist')
        if post.published:
            self.stdout.write(self.style.ERROR(f'Post: {post.title} was already published'))
        else:
            post.published = True
            post.save()
            self.stdout.write(self.style.SUCCESS(f'Post: {post.title} successfully published'))
```

Django 管理命令由一个名为 command 的类组成，该类继承自 BaseCommand。

为了处理参数，该类利用了 [argparse](https://docs.python.org/3/library/argparse.html) 。方法`add_arguments`允许我们的函数接收参数。

在我们的例子中，函数期望一个被分配了键`post_id`的参数

然后,`handle()`函数评估输入并执行我们的逻辑。

在上面的示例中，预期的参数类型称为位置参数，必须为函数的运行提供位置参数。为此，我们运行`python manage.py publish_post 1 (or any post primary key)`

另一种类型的参数称为可选参数，可以应用到方法中，顾名思义，缺少这些不会阻碍函数的执行。

下面提供了一个例子。我们将初始化一个文件，并将其命名为`edit_post.py`。用下面的代码填充它。

```
from django.core.management.base import BaseCommand, CommandError
from posts.models import Category, Post

class Command(BaseCommand):
    help = 'Edits the specified blog post.'

    def add_arguments(self, parser):
        parser.add_argument('post_id', type=int)

        # optional arguments
        parser.add_argument('-t', '--title',type=str, help='Indicate new name of the blog post.')
        parser.add_argument('-c', '--content',type=str, help='Indicate new blog post content.')

    def handle(self, *args, **options):
        title = options['title']
        content = options['content']
        try:
            post = Post.objects.get(id=options['post_id'])
        except Post.DoesNotExist:
            raise CommandError(f'Post with id {options["post_id"]} does not exist')

        if title or content:
            if title:
                old_title = post.title
                post.title = title
                post.save()
                self.stdout.write(self.style.SUCCESS(f'Post: {old_title} has been update with a new title, {post.title}'))
            if content:
                post.content = content
                post.save()
                self.stdout.write(self.style.SUCCESS('Post: has been update with new text content.'))
        else:
            self.stdout.write(self.style.NOTICE('Post content remains the same as no arguments were given.'))
```

这里我们只是编辑一篇博文的标题或内容。为此，我们可以运行`python manage.py edit_post 2 -t "new title"`来编辑标题

或`python manage.py edit_post -c "new content"`仅编辑内容。如果我们希望通过运行`python manage.py edit_post 2 -t "new title again" -c "new content again"`来编辑标题和内容，我们可以提供这两个参数

# 额外资源

1.  [姜戈文件](https://docs.djangoproject.com/en/3.1/howto/custom-management-commands/)。
2.  [简单胜于复杂](https://simpleisbetterthancomplex.com/tutorial/2018/08/27/how-to-create-custom-django-management-commands.html#cron-job)。

# 赞助商

**请注意，下面的一些链接是附属链接，您无需支付额外费用。要知道，我只推荐我个人使用过的并且认为真正有用的产品、工具和学习服务。最重要的是，我从不提倡购买你负担不起或者你不准备实施的东西。**

# 刮刀 API

Scraper API 是一家专注于策略的初创公司，可以缓解你的 IP 地址在网络抓取时被阻止的担忧。他们利用 IP 轮换，所以你可以避免检测。拥有超过 2000 万个 IP 地址和无限带宽。

除此之外，它们还为你提供了 CAPTCHA 处理功能，并启用了一个无头浏览器，这样你就看起来像一个真正的用户，而不会被检测为网页抓取者。用法不限于 scrapy，还可以与 python 生态系统中的 requests、BeautifulSoup 和 selenium 一起使用。还支持与 node.js、bash、PHP 和 ruby 等其他流行平台的集成。您所要做的就是将您的目标 URL 与它们在 HTTP get 请求上的 API 端点连接起来，然后像在任何 web 抓取器上一样继续进行。不知道如何网页抓取？别担心，我已经在[网络抓取系列](https://lewiskori.com/series/web-scraping-techniques-with-python/)中广泛讨论了这个话题。完全免费！

![](img/f59d5bf952c31f95161a65d8a1addbe1.png)

使用[这个 scraper api 链接](https://www.scraperapi.com/?_go=korilewis)和促销代码 lewis10，您将在首次购买时获得 10%的折扣！！你可以随时开始他们慷慨的免费计划，并在需要时升级。

我就说这么多，如果你有任何问题， [twitter dm](https://twitter.com/lewis_kihiu) 随时开放。

*原载于*[*https://lewiskori.com*](https://lewiskori.com/blog/how-to-create-custom-commands-in-django/)*。*