# 如何用 Django 创建一个功能齐全的电子商务网站

> 原文：<https://medium.com/analytics-vidhya/how-to-create-fully-functional-e-commerce-website-with-django-997ffaa0f040?source=collection_archive---------3----------------------->

第 3 步，共 5 步:制作订单摘要

![](img/6fc134ba6f76d6e1138bdb2fe3143e57.png)

[https://www . shuup . com/WP-content/uploads/2017/12/python-plus-django-1 . jpg](https://www.shuup.com/wp-content/uploads/2017/12/python-plus-django-1.jpg)

在本部分教程中，我们将总结购物车中的订单

# 先决条件

*   Django 最新版本
*   Anaconda(可选)创建虚拟环境
*   作为代码编辑器的 Visual Studio 代码(可选)

在开始本教程之前，不要忘记激活您的虚拟环境，以及我们在上一个教程中创建的项目。如果你没有，你可以在这里下载:

[](https://github.com/Andika7/DjangoEcommerce/tree/88aecdce1341d871acdfb4b19add1b497170c634) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/tree/88aecdce1341d871acdfb4b19add1b497170c634) 

# A.模型

使用编辑代码打开 core / models.py，并在类模型中添加一些函数，如下所示:

1.  类**项**

在**项目**类中没有任何改变，因为订单汇总与**项目**类没有直接关系

2.类别 **OrderItem**

在 **OrderItem** 类中，我们将添加一些额外的函数如下:

*   **get_total_item_price** ，返回每个产品项目的总价值
*   **get_discount_item_price，**根据折扣价格返回每个产品项目的总价
*   **get_amount_saved，**返回从现有折扣中节省的价格值
*   **get_final_price，**返回用作价格决定因素的函数(使用原价还是折扣价)

3.阶级**秩序**

在 Order 类中添加一个计算订单总价的函数:

*   **get_total_price，**返回所有订购产品项目的总价值

4.迁移模型数据库

不要忘记使用下面的命令迁移您的模型数据库:

```
$ python manage.py migrate
$ python manage.py makemigrations
```

完整的 models.py 代码可以在以下链接中看到:

[](https://github.com/Andika7/DjangoEcommerce/blob/0bdedf2fe57b3677ec0b1e006d0946cfed10e546/core/models.py) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/blob/0bdedf2fe57b3677ec0b1e006d0946cfed10e546/core/models.py) 

# B.视图

在管理我们的模型之后，现在切换到 core/views.py 文件。在此之前，首先使用以下链接放置您模板目录:

[](https://github.com/Andika7/DjangoEcommerce/tree/0bdedf2fe57b3677ec0b1e006d0946cfed10e546/templates) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/tree/0bdedf2fe57b3677ec0b1e006d0946cfed10e546/templates) 

或者您可以只取***order _ summary . html***并将其放在模板目录中。

1.  导入

```
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import ListView, DetailView, View
from django.utils import timezone
from .models import (
    Item,
    Order,
    OrderItem
)
```

在你的 views.py 中导入所有需求库，在 **django.views.generic** 中你可以看到我们导入了**视图**类。对于订单汇总视图，我们将使用这个**视图**类

2.创建**订单摘要视图**

在 **OrderSummaryView** 类中，我们使用订单模型作为模型，使用 **order_summary.html** 作为模板。

3.创建减少数量的产品

在上一个教程中，我们制作了两个函数: **add_to_cart()** 和 **remove_from_cart()。** add_to_cart()使产品的数量增加，而 remove_from_cart()函数将使所有订单删除。因此，我们将创建一个函数来减少产品项目的数量，如果数量等于 0，则删除它

4.更新视图中的重定向 url

在 views.py 的每个函数中，我们都将所有 URL 重定向到产品，但是我们现在有了一个订单摘要，我们重定向它是有好处的

```
// change this code
return redirect("core:product"// to
return redirect("core:order-summary"
```

5.用户必须登录才能下订单

为每个函数添加一个 [@login_required](http://twitter.com/login_required) 注释，如果用户没有登录，在运行函数时，这会将用户抛入登录页面

```
@login_required
def add_to_cart(request, pk):
   ...@login_required
def remove_from_cart(request, pk):
   ...@login_required
def reduce_quantity_item(request, pk):
   ....
```

完整的 views.py 代码可以在下面的链接中看到:

[](https://github.com/Andika7/DjangoEcommerce/blob/0bdedf2fe57b3677ec0b1e006d0946cfed10e546/core/views.py) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/blob/0bdedf2fe57b3677ec0b1e006d0946cfed10e546/core/views.py) 

# C.资源定位符

完成视图后，我们现在将在 core/url.py 文件中添加 OrderSummaryView url 和 reduce_quantity 项函数

1.  导入

```
from django.urls import path
from .views import (
    remove_from_cart,
    reduce_quantity_item,
    add_to_cart,
    ProductView,
    HomeView,
    OrderSummaryView
)
```

添加我们在导入时在 views.py 文件中创建的所有视图和函数，因此我们只添加来自前一个文件的 **OrderSummaryView** 和 **reduce_quantity_item** 。

2.创建路径 Url

创建 **OrderSummaryView** 和 **reduce_quantity_item()** 函数的路径 url，如下所示:

```
urlpatterns = [
   ...
   path('order-summary', OrderSummaryView.as_view(), 
        name='order-summary'),
   path('reduce-quantity-item/<pk>/', reduce_quantity_item,
        name='reduce-quantity-item')
]
```

现在您可以访问视图和另一个功能

# 完成源代码这一部分:

[](https://github.com/Andika7/DjangoEcommerce/tree/0bdedf2fe57b3677ec0b1e006d0946cfed10e546) [## 安集卡 7/DjangoEcommerce

### 没有提供描述、网站或主题。此时您不能执行该操作。您已使用另一个标签登录…

github.com](https://github.com/Andika7/DjangoEcommerce/tree/0bdedf2fe57b3677ec0b1e006d0946cfed10e546) 

# 进行下一部分！

我希望本教程的第三部分对你有所帮助。在下一节课中，我们将制作这个电子商务网站的结帐表单。