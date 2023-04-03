# 使用 pytesseract、gensim 和 Django 构建 OCR 并总结 Webapp

> 原文：<https://medium.com/analytics-vidhya/build-an-ocr-and-summarize-webapp-using-pytesseract-gensim-and-django-13d0ff911d93?source=collection_archive---------7----------------------->

![](img/fd7e785c7074c891c1532a2103439385.png)

尼克·莫瑞森在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

在本教程中，我们将构建一个 web 应用程序，它将从图像中提取文本，并使用 gensim 库给出 ocred 文本的摘要。

在这里查看最终的项目:【http://emmamichael.pythonanywhere.com/ 

让我们进去:

# **设置项目和应用**

使用 pycharm 创建新项目，并将项目命名为 **ocr。使用以下命令打开终端并安装 Django:**

```
pip install Django==3.1.3
```

然后运行以下命令创建一个名为**ocrandsummary**的新 django 项目:

```
django-admin startproject ocrandsummarize .
```

运行下面的命令创建一个新的应用程序“oands”。

```
python manage.py startapp oands
```

现在我们需要在 INSTALLED_APPS 中注册新的应用程序。由于我们将使用静态和媒体文件，我们将把 **STATICFILES_DIRS、STATIC_ROOT、MEDIA_URL、MEDIA_ROOT** 设置添加到我们的 settings.py 文件中。打开**ocrandsummary**目录下的 settings.py 文件，添加以下粗体代码。

```
from pathlib import Path
**import os**

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = #your secret key will be here

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    **'oands',**
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'ocrandsummarize.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'ocrandsummarize.wsgi.application'

# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
# https://docs.djangoproject.com/en/3.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/3.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.1/howto/static-files/

STATIC_URL = '/static/'

**# Add these new lines
STATICFILES_DIRS = (
    os.path.join(BASE_DIR, 'static'),
)

STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')**
```

打开您的终端并运行下面的命令来安装 pytesseract、gensim、pillow，因为我们将使用 pytesseract 从图像中提取文本，使用 gensim 总结 ocred 文本，使用 pillow 处理图像:

**pip 安装枕头** ==8.0.1

**pip 安装 pytesserac**= = 0 . 3 . 6

**pip 安装 gensim** ==3.8.3

将以下粗体代码添加到我们项目目录**ocrandsummary 的 urls.py 文件中。**这段代码将包含我们已安装的应用程序的 urls.py 文件，并为我们的媒体文件添加配置。

```
from django.contrib import admin
from django.urls import path**, include**
**from django.conf.urls.static import static ** # new
**from django.conf import settings**  # new

urlpatterns = [
    path('admin/', admin.site.urls),
    **path('', include('oands.urls')),**
]

**if settings.DEBUG:  # new
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)**
```

# 我们应用程序的 URL 配置

在我们的应用程序目录“oands”中创建一个“urls.py”文件。添加下面的代码，为我们的视图函数创建一个名为“index”的资源定位器，它将执行大部分操作，并呈现我们的 html 文件。

```
from django.urls import path
from .views import index

urlpatterns = [
    path('', index, name='index'),
]
```

# **型号**

将下面的代码添加到我们的应用程序的 models.py 文件中，创建一个名为“Ocr”的模型，其中包含一个图像字段。

```
from django.db import models

# Create your models here.
class Ocr(models.Model):
    image = models.ImageField(upload_to='images/')
```

运行下面的命令在我们的数据库中创建模型:

```
python manage.py makemigrationspython manage.py migrate
```

# **Admins.py**

我们需要向管理员注册我们的模型。将粗体代码添加到我们应用程序的 admins.py 中。

```
from django.contrib import admin

# Register your models here.
**from .models import Ocr

admin.site.register(Ocr)**
```

# **forms.py**

在我们的应用程序目录中创建一个 forms.py 文件，并添加下面的代码来创建我们的表单，该表单将从模型中呈现一个字段“image”。

```
from django import forms
from .models import Ocr

class ImageUpload(forms.ModelForm):
    class Meta:
        model = Ocr
        fields = ['image']
```

# 景色

打开我们应用程序的 views.py 文件。这是大部分功能将被执行的地方，首先，我们将导入 pytesseract，总结..，pytesseract 会将图像中的文本转换为字符串，而 summary 会协助汇总转换后的文本。

```
from django.shortcuts import render

# import pytesseract to convert text in image to string
**import pytesseract**

# import summarize to summarize the ocred text
**from gensim.summarization.summarizer import summarize**

from .forms import ImageUpload
import os

# import Image from PIL to read image
from PIL import Image

from django.conf import settings

# Create your views here.
def index(request):
    text = ""
    summarized_text = ""
    message = ""
    if request.method == 'POST':
        form = ImageUpload(request.POST, request.FILES)
        if form.is_valid():
            try:
                form.save()
                image = request.FILES['image']
                image = image.name
                path = settings.MEDIA_ROOT
                pathz = path + "/images/" + image

                **text = pytesseract.image_to_string(Image.open(pathz))**
                **text = text.encode("ascii", "ignore")
                text = text.decode()**

                # Summary (0.1% of the original content).
                **summarized_text = summarize(text, ratio=0.1)**
                os.remove(pathz)
            except:
                message = "check your filename and ensure it doesn't have any space or check if it has any text"

    context = {
        'text': text,
        'summarized_text': summarized_text,
        'message': message
    }
    return render(request, 'formpage.html', context)
```

上面的代码将接受用户上传的图像，将图像中的文本转换为字符串，汇总转换后的文本，然后将转换后的文本和汇总后的文本呈现到用户可以查看的 html 页面。

# **模板**

在我们项目的 app 目录中创建一个名为 templates 的文件夹，然后在该目录中创建一个名为“formpage.html”的 html 文件，并添加以下代码:

```
<!DOCTYPE html>
{% load static %}
<html lang="en">
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>OCR AN IMAGE AND SUMMARIZE</title>
        <link href="{% static 'css/form.css' %}" rel="stylesheet" type="text/css">
        <script src="{% static 'js/form.js' %}"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.css" >
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
    </head>
    <body >
        <div class="container">
            <div class="form-container">
                <p style="color: red;">{{message}}</p>
                <h1>
                    Ocr an Image and Summarize Using Gensim
                </h1>
         **       <form method="POST" enctype=multipart/form-data>
                    {% csrf_token %}
                    <label> Image Upload:</label>
                    <input type="file" class="form-control" id="image" name="image" />
                    <button class="button-primary" type="submit" >Submit</button>
                </form>**
                <label><h2>Ocred Text:</h2></label>
               ** <span class="textarea" role="textbox" contenteditable>{{text}}</span>**

                <label><h2>Summarized Text:</h2></label>
               ** <span class="textarea" role="textbox" contenteditable>{{summarized_text}}</span>**

            </div>
        </div>
    </body>

</html>
```

该表单会将上传的图像发送到我们的名为“索引”的视图，该视图将对图像进行 ocr 并总结文本。原始 ocred 文本和摘要文本都将在“span”标签中显示给用户

# **造型**

在我们的项目目录中创建一个名为**‘静态’**的文件夹，并在该文件夹内创建另一个名为**‘CSS’的文件夹。**

在 css 文件夹中，创建一个名为“form.css”的 css 文件，并添加以下代码

```
@media  only screen and (min-device-width: 768px) 
{
    .form-container {
      padding: 5%;
      background: #ffffff;
      border: 9px solid #f2f2f2;            
      max-width: 520px;
      margin: auto;
    }
}
body
{
    background: #00CED1;
}
h1, p
{
  text-align: center;
}
input, textarea , button
{
  width: 100%;
}    
textarea
{
  height: 200px;
} <span class="textarea" role="textbox" contenteditable>{{text}}</span>
```

上面的代码将为我们的 html 标签添加样式。

**在 twitter 上关注我:** [@emmakodes](https://twitter.com/emmakodes)

**领英:**[https://www.linkedin.com/in/emmanuelonwuegbusi/](https://www.linkedin.com/in/emmanuelonwuegbusi/)

# **结论**

恭喜你，你已经完成了本教程。你可以在这里查看代码:[https://github.com/emmakodes/ocrandsummarize](https://github.com/emmakodes/ocrandsummarize)

并且不要忘记在这里检查现场项目:[http://emmamichael.pythonanywhere.com/](http://emmamichael.pythonanywhere.com/)