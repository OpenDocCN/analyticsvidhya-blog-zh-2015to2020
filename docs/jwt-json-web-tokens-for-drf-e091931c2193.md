# JWT(JSON 网络令牌)与 DRF

> 原文：<https://medium.com/analytics-vidhya/jwt-json-web-tokens-for-drf-e091931c2193?source=collection_archive---------2----------------------->

![](img/41a2207a8527c6eb4a0bfcb3a369a4fe.png)

# 部分概述

*   构建 django 项目。
*   定义新闻内容模型和管理。
*   配置新闻内容 api。
*   身份验证和权限。
*   添加 JWT(JSON Web 令牌)。
*   招摇博士。

[JSON Web 令牌](https://jwt.io/)是客户端/服务器应用程序使用的基于令牌的认证，其中客户端是使用。JWT 是基于令牌的认证的流行实现。

在本文中，我们将使用它来验证 JWT 与 Django REST 框架一起使用的用户。创建新闻内容时，只有当前登录的用户才能对自己的新闻内容进行读写操作。

# 第一节。构建 Django 项目

首先安装`Django`、`Django REST Framework(DRF)`和`django-cors-headers (CORS)`。

```
$ pip install django django-rest-framework django-cors-headers
```

然后，创建一个新的项目文件夹(`news_contents/`)。

```
$ mkdir news_contents
$ cd news_contents
```

创建 venv( `virtual environment`)并激活它。

```
$ python -m venv venv
$ source venv/bin/activate
```

现在，构建一个 django 项目(`core`)。

```
$ django-admin startproject core
$ cd core
```

创建一个`newscontents` app。

```
$ python manage.py startapp newscontents
$ python manage.py migrate
```

我们现在将配置 django 项目，在`core/settings.py`中使用`CORS`和`DRF`。

```
# Application definitionINSTALLED_APPS = [
    ....
    'rest_framework',
    'corsheaders',
    'newscontents',
]MIDDLEWARE = [
    ....
    'corsheaders.middleware.CorsMiddleware',
]CORS_ORIGIN_ALLOW_ALL = TrueREST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES':['rest_framework.permissions.AllowAny'],
    'DEFAULT_PARSER_CLASSES': ['rest_framework.parsers.JSONParser'],
}
```

让我们运行服务器。

```
$ python manage.py runserver
Watching for file changes with StatReloader
Performing system checks...System check identified no issues (0 silenced).
November 08, 2019 - 07:24:30
Django version 2.2.7, using settings 'core.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```

# 第二节。定义`NewsContent`模型和管理员

在`core/newscontents/models.py`中创建`NewsContent`模型。

```
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import modelsUSER_MODEL = get_user_model()class NewsContent(models.Model):
    reporter = models.ForeignKey(
        to=USER_MODEL,
        on_delete=models.CASCADE,
        related_name='news_contents',
    )
    headline = models.CharField(max_length=255)
    body = models.TextField() def __str__(self):
        return self.headline
```

运行迁移。

```
$ python manage.py makemigrations
$ python manage.py migrate
```

`core/newscontents/admin.py`中`NewsContent`模型的管理站点。

```
from django.contrib import admin
from .models import NewsContentadmin.site.register(NewsContent)
```

创建一个超级用户。

```
$ python manage.py createsuperuser
Username: reporter
Email address: reporter@example.com
Password: 
Password (again):
Superuser created successfully.
```

让我们再次启动服务器。

```
$ python manage.py runserver
Watching for file changes with StatReloader
Performing system checks...System check identified no issues (0 silenced).
November 08, 2019 - 07:35:23
Django version 2.2.7, using settings 'core.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```

# 第三节。配置`News Content` API

添加一些 URL，以便我们可以访问`core/urls.py`中的新闻内容 API。

```
....
from django.urls import include, pathurlpatterns = [
    ....
    path('api/', include('newscontents.urls')),
]
```

然后，在`newscontents/serializers.py`中为`NewsContent`模型创建一个序列化器。

```
from rest_framework import serializers
from .models import NewsContent class NewsContentSerializer(serializers.ModelSerializer):
    class Meta:
        fields = ('id', 'reporter', 'headline', 'body')
        model = NewsContent
```

接下来，我们将在`newscontents/views.py`中创建视图。

```
from rest_framework import viewsets
from .models import NewsContent
from .serializers import NewsContentSerializerclass NewsContentViewSet(viewsets.ModelViewSet):
    queryset = NewsContent.objects.all()
    serializer_class = NewsContentSerializer
```

在`newscontents/urls.py`中创建 URL。

```
from django.urls import path
from rest_framework.routers import SimpleRouter
from .views import NewsContentViewSetrouter = SimpleRouter()
router.register('news-content', NewsContentViewSet, base_name="news_content")
urlpatterns = router.urls
```

# 第四节。身份验证和权限

首先，在管理面板中创建一个新用户。然后，在`core/urls.py`中添加下面一行。

```
urlpatterns = [
    ....
    path('auth/', include('rest_framework.urls')),
]
```

然后，运行到服务器，我们将创建一个新用户，并从管理面板输入新闻内容。

```
$ python manage.py runserver
Watching for file changes with StatReloader
Performing system checks...System check identified no issues (0 silenced).
November 08, 2019 - 11:08:55
Django version 2.2.7, using settings 'core.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
```

您可以通过登录然后转到`[http://localhost:8000/api/news-content/](http://localhost:8000/api/news-content/.)` [来验证这一点。](http://localhost:8000/api/news-content/.)

我们已经用`AllowAny`配置了 rest_framework。因此，每个用户都可以访问新闻内容。现在，我们需要换成你的`core/settings.py`文件:

```
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': ["rest_framework.permissions.IsAuthenticated",],
    'DEFAULT_PARSER_CLASSES': ['rest_framework.parsers.JSONParser'],
}
```

我们需要在`newscontents/views.py`中进行更改，以确保用户只能看到自己的`News Contents`对象，并将用户设置为`News Contents`对象的报告者。

```
from rest_framework import permissions, viewsets
from rest_framework.exceptions import PermissionDeniedfrom .models import NewsContent
from .serializers import NewsContentSerializerclass IsReporter(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        return obj.reporter == request.userclass NewsContentViewSet(viewsets.ModelViewSet):
    serializer_class = NewsContentSerializer
    permission_classes = (IsReporter,) # Ensure a user sees only own News Content objects.
    def get_queryset(self):
        user = self.request.user
        if user.is_authenticated:
            return NewsContent.objects.filter(reporter=user)
        raise PermissionDenied() # Set user as owner of a NewsContents object.
    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)
```

如果你访问`http://localhost:8000/api/news-content/`，你应该只能看到属于当前登录用户的新闻内容。因此，我们可以从`newscontents/serializers.py`中的序列化程序中删除报告字段:

```
from rest_framework import serializers
from .models import NewsContentclass NewsContentSerializer(serializers.ModelSerializer):
    class Meta:
        fields = ('id', 'headline', 'body')
        model = NewsContent
```

我们已经配置了身份验证和权限。但是 DRF 仍然使用会话认证。现在我们将使用 Json 令牌认证(JWT)。

# 第五节。添加 JWT(JSON Web 令牌)

我们将使用一个库(`djangorestframework_simplejwt`)。

```
$ pip install djangorestframework_simplejwt
```

我们将其添加到`core/settings.py`中的`DEFAULT_AUTHENTICATION_CLASSES`。

```
REST_FRAMEWORK = {
 'DEFAULT_PERMISSION_CLASSES':["rest_framework.permissions.IsAuthenticated"],
    'DEFAULT_PARSER_CLASSES': ['rest_framework.parsers.JSONParser'],
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework_simplejwt.authentication.JWTAuthentication',
       ],
}
```

在`core/urls.py`中增加 api 的`token`和令牌的`refresh`的新端点。

```
from rest_framework_simplejwt.views import TokenObtainView, TokenRefreshViewurlpatterns = [
    ....
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain'),
    # get a new token before the old expires.
    path('api/token/refresh/', TokenRefreshView.as_view, name='token_refresh'),
]
```

# 用户注册

我们需要新的终端用户可以注册。现在为`jwtauth`创建一个新的应用。

```
$ python manage.py startapp jwtauth
```

在`core/settings.py`中添加 app。

```
INSTALLED_APPS = [
    ....
    'jwtauth',
]
```

然后，我们将为`User`模型创建一个新的序列化程序。添加`jwtauth/serializers.py`。

```
from django.contrib.auth import get_user_modelfrom rest_framework import serializersUser = get_user_model() class UserCreateSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True, required=True, style={'input_type': 'password'}
    )
    password2 = serializers.CharField(
        style={'input_type': 'password'}, write_only=True, label='Confirm password'
    ) class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password2']
        extra_kwargs = {'password': {'write_only': True}} def create(self, validated_data):
        username = validated_data['username']
        email = validated_data['email']
        password = validated_data['password']
        password2 = validated_data['password2']
        if (email and User.objects.filter(email=email).exclude(username=username).exists()
        ):
            raise serializers.ValidationError(
                {'email': 'Email addresses must be unique.'}
            )
        if password != password2:
            raise serializers.ValidationError({'password': 'The two passwords differ.'})
        user = User(username=username, email=email)
        user.set_password(password)
        user.save()
        return user
```

然后我们将在`jwtauth/views.py`中创建一个新视图。

```
from django.contrib.auth import get_user_model
from rest_framework import permissions
from rest_framework import response, decorators, permissions, status
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import UserCreateSerializerUser = get_user_model()@decorators.api_view(['POST'])
@decorators.permission_classes([permissions.AllowAny])
def registration(request):
    serializer = UserCreateSerializer(data=request.data)
    if not serializer.is_valid():
        return response.Response(serializer.errors, status.HTTP_400_BAD_REQUEST)        
    user = serializer.save()
    refresh = RefreshToken.for_user(user)
    res = {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }
    return response.Response(res, status.HTTP_201_CREATED)
```

我们将视图添加到`jwtauth/urls.py`。

```
from django.urls import path
from .views import registrationurlpatterns = [
    path('register/', registration, name='register')
]
```

接下来，在`core/urls.py`中包含`jwtauth`URL。

```
urlpatterns = [
    ....
    path('api/jwtauth/', include('jwtauth.urls'), name='jwtauth'),
]
```

# 第六节。招摇文件

```
$ pip install django-rest-swagger
```

在`core/settings.py`中将它添加到您的`INSTALLED_APPS`列表中。

```
INSTALLED_APPS = [
    'rest_framework_swagger',
    ....
]REST_FRAMEWORK = {
    ....
 'DEFAULT_SCHEMA_CLASS':'rest_framework.schemas.coreapi.AutoSchema',
}
```

添加以下代码`core/urls.py`。

```
from django.contrib import admin
from django.urls import path, include
from rest_framework_swagger.views import get_swagger_viewschema_view = get_swagger_view(title="News Contents API")urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('newscontents.urls')),
    path('auth/', include('rest_framework.urls')),
    path('api/jwtauth/', include('jwtauth.urls'), name='jwtauth'),
    path('api/docs/', schema_view),
]
```

然后在`jwtauth/urls.py`中放入令牌并刷新 enpoints。

```
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import registrationurlpatterns = [
    path("register/", registration, name="register"),
    path("token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("refresh/", TokenRefreshView.as_view(), name="token_refresh"),
]
```

您可以访问`http://localhost:8000/api/docs/`获得 API 的完整列表，令牌端点现在属于`/jwtauth`分组。

# 参考

*   [https://www.django-rest-framework.org](https://www.django-rest-framework.org)
*   https://jwt.io/introduction/
*   【https://github.com/adamchainz/django-cors-headers 
*   [https://github.com/davesque/django-rest-framework-simplejwt](https://github.com/davesque/django-rest-framework-simplejwt)
*   [https://django-rest-swagger.readthedocs.io/en/latest/](https://django-rest-swagger.readthedocs.io/en/latest/)