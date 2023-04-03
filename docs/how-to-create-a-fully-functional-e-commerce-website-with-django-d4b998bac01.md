# 如何用 Django 创建一个功能齐全的电子商务网站

> 原文：<https://medium.com/analytics-vidhya/how-to-create-a-fully-functional-e-commerce-website-with-django-d4b998bac01?source=collection_archive---------4----------------------->

第 5 步，共 5 步:使用 Stripe 进行支付处理

![](img/6fc134ba6f76d6e1138bdb2fe3143e57.png)

[https://www . shuup . com/WP-content/uploads/2017/12/python-plus-django-1 . jpg](https://www.shuup.com/wp-content/uploads/2017/12/python-plus-django-1.jpg)

在教程的这一部分，我们将使用 stripe 处理支付。

# 什么是条纹？

**Stripe** 是一款支付处理器，这意味着它支持从客户银行(发卡银行)向商户银行(收单银行)进行电子转账，以支付用信用卡购买的商品或服务。有关条带的完整信息可在以下链接中查看:

[](https://stripe.com/) [## 互联网业务的在线支付处理- Stripe

### 互联网业务的在线支付处理。Stripe 是一套支付 API，支持在线商务…

stripe.com](https://stripe.com/) 

# 先决条件

*   Django 最新版本
*   Anaconda(可选)创建虚拟环境
*   作为代码编辑器的 Visual Studio 代码(可选)
*   种类

在开始本教程之前，不要忘记激活您的虚拟环境，以及我们在上一个教程中创建的项目。如果您没有在此下载它:

[](https://github.com/Andika7/DjangoEcommerce/tree/bd146d530d8bec8a5dea3960d3291fe9b60d2bfe) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/tree/bd146d530d8bec8a5dea3960d3291fe9b60d2bfe) 

# 安装条纹

要安装条带，您可以使用以下命令:

```
$ pip install stripe
```

现在 sripe 正在安装到您设备中。在本教程中，我们将使用 stripe API 来处理支付。通过使用以下链接，可以在 Stripe API 文档中看到如何使用 Stripe API:

 [## 条带 API 参考-创建收费- Python

### Stripe API 的完整参考文档。包括我们的 Python 的代表性代码片段和示例…

stripe.com](https://stripe.com/docs/api/charges/create?lang=python) 

确保你使用 python 语言

# 模型

打开您的 models.py 文件，我们将使用 stripe 创建一个支付模型类来保存支付信息:

该模型将存储与使用 stripe 的支付相关的信息，如 stripe 的 ID、支付者、支付金额和支付时间

创建支付模型后，不要忘记使用以下命令迁移您的模型数据库:

```
$ python manage.py migrate
$ python manage.py makemigrations
```

完整的 models.py 代码可以在以下链接中看到:

[](https://github.com/Andika7/DjangoEcommerce/blob/master/core/models.py) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/blob/master/core/models.py) 

# 视图

在视图中，使用下面的代码替换我们之前创建的 CheckoutView 类:

如果选择了 stripe 选项，这一次 Checkout 视图类将重定向到 payment using stripe

然后在下面的链接中找到 payment.html 模板，并把它放在模板目录中

[](https://github.com/Andika7/DjangoEcommerce/tree/master/templates) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/tree/master/templates) 

> 建议:用链接中的文件替换模板目录中的所有文件

之后，我们将创建一个带条纹的 PaymentView 类。对于以下教程:

1.  导入

```
...
from .models import (
   ...,
   Payment 
)import stripe
stripe.api_key = settings.STRIPE_KEY
```

在 import 列中，导入我们之前制作的支付模型，不要忘记导入 stripe，因为我们将在这里使用它

2.设置

打开项目名称目录中的 settings.py 文件，并将该行代码添加到您的底线中:

```
STRIPE_KEY = '<your_api_key_here>'
```

将您条带 API 密码输入 constanta 变量

3.创建付款视图

付款视图的主要部分是 try 部分，而错误处理可以在以下文档中看到:

 [## 条带 API 参考-处理错误- Python

### Stripe API 的完整参考文档。包括我们的 Python 的代表性代码片段和示例…

stripe.com](https://stripe.com/docs/api/errors/handling?lang=python) 

**post** 函数将使用我们之前制作的支付模型的条带来存储支付信息

完整的 **views.py** 代码可以在下面的链接中看到:

[](https://github.com/Andika7/DjangoEcommerce/blob/master/core/views.py) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce/blob/master/core/views.py) 

# 资源定位符

像往常一样，在视图准备好使用之后，现在我们将创建用于执行该功能的 url 地址

在导入中添加 PaymentView

```
from .views import (
   ...,
   PaymentView
)
```

向 PaymentView 类中的视图添加路径 url

```
urlpatterns = [
   ...,
   path('payment/<payment_option>/',
         PaymentView.as_view(), name='payment')
]
```

现在你可以使用 stripe 支付，你可以查看模板目录中的 payment.html**文件**来了解我如何在 stripe 支付表单上调用**并发布**

# 完成源代码这一部分:

[](https://github.com/Andika7/DjangoEcommerce) [## 安集卡 7/DjangoEcommerce

### 在 GitHub 上创建一个帐户，为 Andika7/DjangoEcommerce 的发展做出贡献。

github.com](https://github.com/Andika7/DjangoEcommerce) 

# 结束语

谢谢你把这个教程看完。有了这个教程“*如何用 Django* 创建一个功能齐全的电子商务网站”就完成了。非常感谢阅读这篇文章的人