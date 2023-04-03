# 如何创建 Django 网站地图

> 原文：<https://medium.com/analytics-vidhya/django-sitemap-8f4ca0538fa?source=collection_archive---------0----------------------->

![](img/633e7b80f35c75ec664246c5a348bc1e.png)

Sitemap 是一个带有 xml 扩展名的文件，包含我们在网站上发布的内容页面。今天，站点地图已经成为每个站点最重要的 SEO 标准之一。

我们想从创建一个新项目开始。

```
$ django-admin startproject django_sitemap
$ cd django_sitemap/
```

我们现在想要创建一个新的应用程序，我将把它称为我的应用程序。

```
$ ./manage.py startapp myapp
```

我们可以将`myapp`添加到我们的安装应用中。

```
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp',
]
```

`django_sitemap/urls.py:`

```
from django.contrib import admin
from django.urls import path, include
urlpatterns = [
    path('', include('myapp.urls')),
    path('admin/', admin.site.urls),
]
```

`myapp/urls.py:`

```
from django.urls import path
from myapp.views import about
urlpatterns = [
    path('about/', about, name='about')
]
```

`myapp/views.py:`

```
from django.http import HttpResponse
def about(request):
    return HttpResponse('about page')
```

有一个叫做`Django contrib sitemaps`的包，开始创建我们的网站地图。

```
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp',
    'django.contrib.sitemaps',
]
```

`StaticViewSitemap()`将是我们所有静态视图的站点地图。每个单独的站点地图都必须有一个名为 item 的函数，并且`location()`应该为每个单独的项目返回一个路径。

`myapp/sitemaps.py:`

```
from django.contrib.sitemaps import Sitemap
from django.shortcuts import reverse
class StaticViewSitemap(Sitemap):
    def items(self):
        return ['about']
    def location(self, item):
        return reverse(item)
```

我们想要导入`static view the site map`，然后我们可以创建一个名为`site maps`的新字典，我们想要给它`static`的键和`static view side map`的值，因为我们需要马上传递这个字典。

`django_sitemap/urls.py:`

```
from django.contrib import admin
from django.contrib.sitemaps.views import sitemap
from django.urls import include, path
from myapp.sitemaps import StaticViewSitemap
sitemaps = {
    'static': StaticViewSitemap
}
urlpatterns = [
    path('', include('myapp.urls')),
    path('sitemap.xml', sitemap, {'sitemaps': sitemaps}),
    path('admin/', admin.site.urls),
]
```

## 运行应用程序

*   `http://127.0.0.1:8000/sitemap.xml`，我们得到一个包含位置的 XML 文件。

```
<urlset ae lk" href="http://www.sitemaps.org/schemas/sitemap/0.9" rel="noopener ugc nofollow" target="_blank">http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>[http://127.0.0.1:8000/about/](http://127.0.0.1:8000/about/)</loc>
    </url>
</urlset>
```

# 模型

我们创建一个片段模型。

```
from django.db import models
from django.utils.text import slugify
class Snippet(models.Model):
    title = models.CharField(max_length=80)
    slug = models.SlugField(blank=True, null=True)
    body = models.TextField()def save(self, *args, **kwargs):
        self.slug = slugify(self.title)
        super().save(*args, **kwargs)def get_absolute_url(self):
        return f'/{self.slug}'
```

使用`save()`方法。因为，我们想从标题中生成 slug。

```
$ ./manage.py makemigrations
$ ./manage.py migrate
$ ./manage.py shell
In [1]: from myapp.models import Snippet
In [2]: Snippet.objects.create(title='t1', body='<h1></h1').save()
In [3]: Snippet.objects.create(title='t2', body='<h2></h2').save()
```

现在，我们想创建一个真实的路径和我们的详细视图，这一个将采取一个鼻涕虫。

`myapp/urls.py:`

```
from django.urls import path
from myapp.views import about, snippet_detail
urlpatterns = [
    path('<slug:slug>/', snippet_detail),
    path('about/', about, name='about'),
]
```

`myapp/views.py:`

```
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from .models import Snippetdef about(request):
    return HttpResponse('about page')def snippet_detail(request, slug):
    snippet = get_object_or_404(Snippet, slug=slug)
    return HttpResponse(f'the detailview for slug of {slug}')
```

## 运行应用程序

*   `http://127.0.0.1:8000/h2/`，h2 段塞流的详细视图。

`django_sitemap/urls.py:`

```
from django.contrib import admin
from django.contrib.sitemaps.views import sitemap
from django.urls import include, path
from myapp.sitemaps import SnippetSitemap, StaticViewSitemapsitemaps = {'static': StaticViewSitemap, 'snippet': SnippetSitemap}urlpatterns = [
    path('', include('myapp.urls')),
    path('sitemap.xml', sitemap, {'sitemaps': sitemaps}),
    path('admin/', admin.site.urls),
]
```

`myapp/sitemaps.py:`

```
from django.contrib.sitemaps import Sitemap
from django.shortcuts import reverse
from .models import Snippetclass StaticViewSitemap(Sitemap):
    def items(self):
        return ['about']def location(self, item):
        return reverse(item)class SnippetSitemap(Sitemap):
    def items(self):
        return Snippet.objects.all()
```

## 运行应用程序

*   我们得到一个包含位置的 XML 文件。

```
<urlset ae lk" href="http://www.sitemaps.org/schemas/sitemap/0.9" rel="noopener ugc nofollow" target="_blank">http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>[http://127.0.0.1:8000/about/](http://127.0.0.1:8000/about/)</loc>
    </url>
    <url>
        <loc>[http://127.0.0.1:8000/h1](http://127.0.0.1:8000/h1)</loc>
    </url>
    <url>
        <loc>[http://127.0.0.1:8000/h2](http://127.0.0.1:8000/h2)</loc>
    </url>
    <url>
        <loc>[http://127.0.0.1:8000/h3](http://127.0.0.1:8000/h3)</loc>
    </url>
</urlset>
```

# 结论

如果你在 [django 网站地图文档](https://docs.djangoproject.com/en/2.1/ref/contrib/sitemaps/)中查找。你可以在网站地图上找到其他属性来告诉搜索引擎更多关于它的信息。在本文中，您已经学习了如何创建站点地图。该应用的源代码可在 [GitHub](https://github.com/erdimollahuseyin/django-site-map) 上获得。