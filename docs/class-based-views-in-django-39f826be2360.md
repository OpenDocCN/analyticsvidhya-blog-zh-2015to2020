# Django 中基于类的视图

> 原文：<https://medium.com/analytics-vidhya/class-based-views-in-django-39f826be2360?source=collection_archive---------22----------------------->

![](img/4fae1207e68de9dc02695cbcd2dc667c.png)

[Codementor.io](https://www.codementor.io/@jamesezechukwu/working-with-class-based-views-in-django-5zkjnrvwc)

在 Django 中开发一个 web 应用程序让我们可以使用一个健壮的框架，这个框架具有许多特性，旨在使编程更加高效和直观。其中一个特性是基于类的视图。基于类的视图是从 Django 框架中的视图类扩展而来的 Python 对象。这些对象以其灵活性和可重用性而闻名(想想面向对象的编程)。但是，文档警告说，它们不能代替基于函数的视图，但是可以在特定的情况下为您所用。

> 其核心是，基于类的视图允许您使用不同的类实例方法来响应不同的 HTTP 请求方法，而不是在单个视图函数内使用有条件的分支代码—基于类的视图简介

每个 Django 视图都接受一个 HTTP 请求并返回一个 HTTP 响应。基于类的视图的一个主要好处是，您可以轻松地定义如何响应不同的 HTTP 方法，而无需输入繁琐的条件。看看这个来自[的基于类的视图介绍](https://docs.djangoproject.com/en/3.0/topics/class-based-views/intro/)的例子。

这段代码利用基于函数的视图在返回正确的 HTTP 响应之前检查 HTTP 方法是否是“GET”。

```
**from** **django.http** **import** HttpResponse

**def** my_view(request):
    **if** request.method == 'GET':
        *# <view logic>*
        **return** HttpResponse('result')
```

当我们用基于类的视图替换上面的函数时，我们可以直接使用 get 作为 to 类中的方法来接受请求并返回正确的响应，而不必使用任何条件语句。这是 Django 如何解决基于函数的视图的一些缺点的一个例子。

```
**from** **django.http** **import** HttpResponse
**from** **django.views** **import** View

**class** **MyView**(View):
    **def** get(self, request):
        *# <view logic>*
        **return** HttpResponse('result')
```

基于类的视图的另一个重要好处是类继承。我们可以创建子类来访问父类的属性和方法。当我们扩展我们的 web 应用程序时，这促进了干代码和可重用性。

学习 Django 中基于类的视图的一个真正有趣的部分是利用*通用的基于类的视图* ***。*** 通用的基于类的视图被构建到通用用例的框架中，例如查看对象列表或该对象的特定实例。不同的通用视图从父类继承了很多功能，所以解释发生了什么的代码行更少。程序员发现这个警告既方便又令人困惑，所以在实现一个通用的基于类的视图时参考文档是一个好习惯。[这里的](https://ccbv.co.uk/)是一个小备忘单，以防文档难以理解。

要访问通用的基于类的视图，您必须首先从 Django 导入模块，如下所示。例如，假设我们想要创建一种管理医院记录的方法，并且需要显示病人就诊的列表。我们可以导入通用的 ListView 类，并根据我们的模型轻松地使用它来请求和返回单个患者记录的列表。

```
from django.shortcuts import render
from django.views.generic.list import ListViewclass RecordsListView(ListView):    
""" Renders a list of all Pages. """    
  model = Record def get(self, request):        
""" GET a list of Pages. """        
    records = self.get_queryset().all()        
    return render(request, 'records/records_list.html', 
    {'records': records})
```

如果我们需要查看每个患者记录中包含的信息，该怎么办？Django 还提供了一个通用的 DetailView 类，可以轻松地访问模型的各个实例。您可以指定要呈现的字段以及要使用的唯一标识符。举一个基本的例子，我通过主键返回了单个记录，并且只用了**两行**代码！

```
from django.views.generic.detail import DetailViewclass RecordsDetailView(DetailView):    
""" Renders a specific page based on it's pk."""    
  model = Record
```

决定是使用基于类的视图还是传统的基于函数的视图最终取决于您的个人用例。使用基于类的视图的一些优点是可重用性、可扩展性、干代码和没有条件语句的 HTTP 请求。更难阅读和需要额外输入的事实是一些缺点。选择最能满足您需求的方法。

感谢您阅读这篇文章。如果你喜欢，给它一些掌声。👏查看下面的链接，了解 Django 中基于类的视图的更多信息！

# 资源

*   [让学校 BEW 1.2](https://make-school-courses.github.io/BEW-1.2-Authentication-and-Associations/#/Lessons/04-ViewsURLs)
*   [Django:基于类的视图 vs 基于函数的视图](/@ksarthak4ever/django-class-based-views-vs-function-based-view-e74b47b2e41b)
*   [基于类的视图](https://django-advanced-training.readthedocs.io/en/latest/features/class-based-views/)
*   Django 初学者完全指南—第 6 部分
*   [基于类的视图介绍](https://docs.djangoproject.com/en/3.0/topics/class-based-views/intro/)