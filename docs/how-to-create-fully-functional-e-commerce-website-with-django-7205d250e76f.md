# 如何用 Django 创建一个功能齐全的电子商务网站

> 原文：<https://medium.com/analytics-vidhya/how-to-create-fully-functional-e-commerce-website-with-django-7205d250e76f?source=collection_archive---------4----------------------->

第 4 步，共 5 步:创建结帐表单

![](img/6fc134ba6f76d6e1138bdb2fe3143e57.png)

[https://www . shuup . com/WP-content/uploads/2017/12/python-plus-django-1 . jpg](https://www.shuup.com/wp-content/uploads/2017/12/python-plus-django-1.jpg)

在本部分教程中，我们将创建订单结账

# 先决条件

*   Django 最新版本
*   Anaconda(可选)创建虚拟环境
*   作为代码编辑器的 Visual Studio 代码(可选)

在开始本教程之前，不要忘记激活您的虚拟环境，以及我们在上一个教程中创建的项目。如果你没有，你可以在这里下载:

[](https://github.com/Andika7/DjangoEcommerce/tree/0bdedf2fe57b3677ec0b1e006d0946cfed10e546) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/tree/0bdedf2fe57b3677ec0b1e006d0946cfed10e546) 

# A.形式

在教程的这一部分，我们将创建一个结帐表单。好的，只要从下面的链接下载名为 checkout.html**的结帐页面表单的模板就可以了:**

[](https://github.com/Andika7/DjangoEcommerce/tree/bd146d530d8bec8a5dea3960d3291fe9b60d2bfe/templates) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/tree/bd146d530d8bec8a5dea3960d3291fe9b60d2bfe/templates) 

**结账**。html 文件已根据结账显示的需要进行了调整。然后我们只需要从核心目录创建一个表单。

然后在核心目录中创建一个名为 **forms.py** 的新文件，该文件将存储与表单相关的所有内容。

导入:

```
from django import forms
from django_countries.fields import CountryField
from django_countries.widgets import CountrySelectWidget
```

您可以看到，我们正在使用国家/地区字段库，这是第三方库，因此我们必须安装它才能使用。要安装国家库，只需运行下面的命令:

```
$ pip install django-countries
```

在 settings.py 中添加以下代码:

```
INSTALLED_APPS = [
   ...
   'django_countries'
]
```

然后我们创建一个名为 **CheckoutForm** 的类表单，作为来自 checkout 页面的表单

有了这个，你就有了结账表格

# B.模型

为了创建一个结帐表单，我们将创建一个新的模型类，用于保存订单的送货地址。

我们将创建的类是 **CheckoutAddress** ,它将存储我们之前创建的表单中订单的发货地址

也不要忘记导入，因为我们将使用国家字段:

```
from django_countries.fields import CountryField
```

不要忘记使用下面的命令迁移您的模型数据库:

```
$ python manage.py migrate
$ python manage.py makemigrations
```

完整的 models.py 代码可以在以下链接中看到:

[](https://github.com/Andika7/DjangoEcommerce/blob/bd146d530d8bec8a5dea3960d3291fe9b60d2bfe/core/models.py) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/blob/bd146d530d8bec8a5dea3960d3291fe9b60d2bfe/core/models.py) 

# C.视图

在视图中，不要忘记添加国家字段库，也不要忘记添加我们刚刚创建的模型和表单

```
from .forms import CheckoutForm
from .models import (
    ...
    CheckoutAddress
)
```

创建签出视图类

在我们之前创建的 forms.py 文件中。我们通过视图将表单发送到模板中。 **post** 函数捕获通过模板中的表单发送的值

完整的 **views.py** 代码可以在下面的链接中看到:

[](https://github.com/Andika7/DjangoEcommerce/blob/bd146d530d8bec8a5dea3960d3291fe9b60d2bfe/core/views.py) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/blob/bd146d530d8bec8a5dea3960d3291fe9b60d2bfe/core/views.py) 

# D.资源定位符

像往常一样，在我们创建视图之后，不要忘记在 core/urls.py 文件中注册它。

导入

```
from django.urls import path
from .views import (
    ...
    CheckoutView
)
```

将视图中的所有类和函数注册到 urls.py 文件中。之后，我们将添加地址

使用下面的代码创建结帐页面的路径 url:

```
urlpatterns = [
   ...
   path('checkout', CheckoutView.as_view(), name='checkout'),
]
```

最后可以访问结帐表单，并将订单数据保存到数据库中

# 完成源代码这一部分:

[](https://github.com/Andika7/DjangoEcommerce/tree/bd146d530d8bec8a5dea3960d3291fe9b60d2bfe) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/tree/bd146d530d8bec8a5dea3960d3291fe9b60d2bfe) 

# 进行下一部分！

我希望本教程的第四部分对你有所帮助。在下一节课中，我们将使用 stripe 处理支付。