# Django 的基于函数和基于类的观点

> 原文：<https://medium.com/analytics-vidhya/djangos-function-based-and-class-based-views-339d0022b542?source=collection_archive---------1----------------------->

# 问题

你有没有想过为什么 Django 教程使用基于函数的视图:

[](https://docs.djangoproject.com/en/2.1/intro/tutorial01/#write-your-first-view) [## 编写您的第一个 Django 应用程序，第 1 部分| Django 文档| Django

### 本教程是为 Django 2.2 编写的，它支持 Python 3.5 及更高版本。如果 Django 版本不匹配，你…

docs.djangoproject.com](https://docs.djangoproject.com/en/2.1/intro/tutorial01/#write-your-first-view) 

```
# polls/views.py
def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")
```