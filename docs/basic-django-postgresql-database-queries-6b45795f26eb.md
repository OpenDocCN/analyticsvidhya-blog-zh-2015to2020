# 基本 Django 数据库查询

> 原文：<https://medium.com/analytics-vidhya/basic-django-postgresql-database-queries-6b45795f26eb?source=collection_archive---------21----------------------->

Django 附带了一个强大的数据库抽象 API，可以让您轻松地创建、检索、更新和删除对象。Django **对象关系映射器**与 MySQL、PostgreSQL、SQLite 和 Oracle 兼容。请记住，您可以在您项目的 *settings.py* 文件的*数据库*设置中定义您项目的数据库。Django 可以同时处理多个数据库，您可以对数据库路由器进行编程，以创建定制的路由方案。

一旦你创建了你的数据模型，Django 会给你一个免费的 API 来与它们交互。你可以在[https://docs.djangoproject.com/en/2.2/ref/models/](https://docs.djangoproject.com/en/2.2/ref/models/)找到官方文档的数据模型参考。

## 创建对象

打开终端并运行以下命令打开 Python shell:

```
**python manage.py shell**
```

然后，键入以下几行:

```
**>>> from django.contrib.auth.models import User**
**>>> from blog.models import Post**
**>>> user = User.objects.get(username='admin')**
**>>> post = Post(title='Another post',**
 **slug='another-post',**
 **body='Post body.',**
 **author=user)**
**>>> post.save()**
```

我们来分析一下这段代码是做什么的。首先，我们检索用户名为 *admin* 的*用户*对象:

```
user = User.objects.get(username='admin')
```

*get()* 方法允许您从数据库中检索单个对象。请注意，该方法期望得到与查询匹配的结果。如果数据库没有返回结果，该方法将引发一个*不存在*异常，如果数据库返回多个结果，将引发一个*多对象返回*异常。这两个例外都是执行查询的模型类的属性。

然后，我们创建一个带有自定义标题、slug 和 body 的 *Post* 实例，并将之前检索到的用户设置为帖子的作者:

```
post = Post(title='Another post', slug='another-post', body='Post body.', author=user)
```

*对象在内存中，没有保存到数据库中。*

最后，我们使用 *save()* 方法将 *Post* 对象保存到数据库:

```
post.save()
```

前面的操作在后台执行一个 *INSERT* SQL 语句。我们必须首先在内存中创建一个对象，然后将它持久化到数据库中，但是我们也可以使用 *create()* 方法在一个操作中创建该对象并将其持久化到数据库中，如下所示:

```
Post.objects.create(title='One more post', slug='one-more-post', body='Post body.', author=user)
```

## 更新对象

现在，将文章的标题改为不同的名称，并再次保存对象:

```
**>>> post.title = 'New title'**
**>>> post.save()**
```

这一次， *save()* 方法执行一个 *UPDATE* SQL 语句。

*在调用****save()****方法之前，您对对象所做的更改不会持久保存到数据库中。*

## 检索对象

Django **对象关系映射** ( **ORM** )是基于 QuerySets 的。QuerySet 是数据库中对象的集合，可以有几个过滤器来限制结果。您已经知道如何使用 *get()* 方法从数据库中检索单个对象。我们已经使用 *Post.objects.get()* 访问了这个方法。每个 Django 模型至少有一个管理器，默认的管理器叫做 **objects** 。您使用模型管理器获得了一个 *QuerySet* 对象。要从表中检索所有对象，只需使用默认对象管理器上的 *all()* 方法，如下所示:

```
**>>> all_posts = Post.objects.all()**
```

这就是我们如何创建一个返回数据库中所有对象的 QuerySet。请注意，这个 QuerySet 尚未执行。django query sets*懒*；只有当他们需要被强迫时，他们才会被评估。这种行为使得*查询集*非常高效。如果我们不将 *QuerySet* 设置为一个变量，而是将它直接写在 Python shell 上，QuerySet 的 SQL 语句将被执行，因为我们强制它输出结果:

```
**>>> Post.objects.all()**
```

## 使用 filter()方法

要过滤一个查询集，可以使用管理器的 *filter()* 方法。例如，我们可以使用以下查询集检索 2020 年发布的所有帖子:

```
Post.objects.filter(publish__year=2020)
```

您也可以筛选多个字段。例如，我们可以用用户名 *admin* 检索作者在 2020 年发表的所有帖子:

```
Post.objects.filter(publish__year=2020, author__username='admin')
```

这相当于构建相同的查询集链接多个过滤器:

```
Post.objects.filter(publish__year=2020) \
            .filter(author__username='admin')
```

*具有字段查找方法的查询是使用两个下划线构建的，例如****publish _ _ year****，但是访问相关模型的字段也使用相同的表示法，例如****author _ _ username****。*

## **使用 exclude()**

您可以使用管理器的 *exclude()* 方法从查询集中排除某些结果。例如，我们可以检索 2020 年发布的标题不以*开头的所有帖子，为什么*:

```
Post.objects.filter(publish__year=2020) \
            .exclude(title__startswith='Why')
```

## **使用 order_by()**

您可以使用管理器的 *order_by()* 方法按不同的字段对结果进行排序。例如，您可以检索按标题排序的所有对象，如下所示:

```
Post.objects.order_by('title')
```

升序是隐含的。您可以用负号前缀表示降序，如下所示:

```
Post.objects.order_by('**-**title')
```

## 删除对象

如果您想要删除一个对象，您可以使用 *delete()* 方法从对象实例中删除它:

```
post = Post.objects.get(id=1)
post.delete()
```

*请注意，删除对象也会删除任何依赖关系。*

# 评估查询集时

您可以将任意数量的过滤器连接到一个 QuerySet，并且在对 QuerySet 进行评估之前，您不会命中数据库。查询集仅在以下情况下计算:

1.  第一次迭代它们时
2.  例如，当您对它们进行切片时， *Post.objects.all()[:3]*
3.  当你腌制或储藏它们时
4.  当你在上面呼叫 *repr()* 或 *len()* 时
5.  当您显式地对它们调用 *list()* 时
6.  当你在一个语句中测试它们的时候，比如 *bool()* ，*或者*，*和*，或者 *if*