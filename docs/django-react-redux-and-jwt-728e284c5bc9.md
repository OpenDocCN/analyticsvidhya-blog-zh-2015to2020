# 姜戈+ React，Redux 和 JWT

> 原文：<https://medium.com/analytics-vidhya/django-react-redux-and-jwt-728e284c5bc9?source=collection_archive---------4----------------------->

## 走向全栈开发

# 前言

在过去的 4 年里，我一直在用 Python 编程，当涉及到 web 时，我总是用 Django + Bootstrap 和 jQuery。我也知道如何用 css 来设计一个页面，但是没有什么特别的。我总是远离现代 spa 和它们的框架/库，如 React、Angular 或 Vue。我试过一次，体验很恐怖。我对 babel、webpack、yarn 以及它们是如何粘在一起的一无所知。更不用说 JS 中的箭头函数和析构了。但最终我决定再试一次，并花了不知多少时间观看 React 上的教程。所以现在这是我尝试让 Django 后端与 React 前端一起工作。

本文的目标是拥有一个带有 JSON Web Token 身份验证的最小后端，一个带有登录/注销功能的简单前端，以及一个只为登录用户提供的受保护页面。对我来说，这主要是为了让一切顺利进行。因此，如果将来我需要重复这些步骤，我可以回顾我所做的并重复这些步骤。因此，我决定:

*   保留默认的 SQLite 数据库，以便可以根据需要替换它
*   不要使用任何 UI 框架或任何样式，因为那样会显得固执己见，而且不适合每个项目

还有一点要注意。我不会详细讨论这里列出的代码。如果你想真正理解事物，有大量有用的信息。我会列出每一个对我有帮助的资源。这只是一个操作指南。完整的代码可以在我的 [github](https://github.com/c-v-ya/djact) 和 [gitlab](https://gitlab.com/c.v.ya/djact) 上找到。

所有这些都过去了，拥抱你自己去做一个长篇阅读吧！我希望这对你有用😊

# 先决条件

您需要在系统上安装以下软件包:python(版本 3，这里没有遗留代码😎)，pip，node，npm，yarn。我使用的是 Arch linux，所以列出的命令应该与任何其他类似 Unix 的系统相同或相似。

让我们从创建一个项目目录开始，将`mkdir djact`和`cd`放入其中。然后用`python -m venv venv`创建虚拟环境并激活它— `source venv/bin/activate`。

# 创建 Django 项目

用`pip install django djangorestframework djangorestframework-simplejwt django-cors-headers`安装 Django、REST 框架和 JWT 处理。最后一个包是允许我们的开发 react 服务器与 Django 应用程序交互所必需的。让我们在安装一些东西之后保存我们的依赖关系:`pip freeze > requirements.txt`。现在开始一个新项目`django-admin startproject djact .`。注意最后的`.`，它告诉 Django 在当前目录下创建项目。

# 应用程序

我喜欢把我所有的应用程序和设置放在一个单独的目录里。那么就让它:`mkdir djact/{apps, settings}`。并将`setting.py`移动到新创建的设置目录中。将`settings`做成一个包`touch djact/settings/__init__.py`，并在其中插入以下代码行:

```
# djact/settings/__init__.py
from .settings import *
```

在这里和每个文件列表中，第一行是一个注释，带有文件的相对路径。只是想让你知道。

这样我们就不需要覆盖`DJANGO_SETTINGS_MODULE`变量。

# 核心

现在为核心应用程序`mkdir djact/apps/core`和应用程序本身`python manage.py startapp core djact/apps/core`创建一个目录。在这个新创建的目录`mkdir {templates,templatetags}`中。
创建一个空的`__init__.py`并在`templatetags`目录中反应加载器模板标签`load_react.py`:

```
# djact/apps/core/templatetags/load_react.py
from django import template
from django.conf import settings
from django.utils.safestring import mark_saferegister = template.Library()@register.simple_tag
def load_react():
    css = load_css()
    js = load_js()
    return mark_safe(''.join(css + js))def load_css():
    return [
        f'<link rel="stylesheet" href="/static/{asset}"/>'
        for asset in load_files('.css')
    ]def load_js():
    return [
        f'<script type="text/javascript" src="/static/{asset}"></script>'
        for asset in load_files('.js')
    ]def load_files(extension: str):
    files = []
    for path in settings.STATICFILES_DIRS:
        for file_name in path.iterdir():
            if file_name.name.endswith(extension):
                files.append(file_name.name)return files
```

我知道有一个 [django-webpack-loader，](https://github.com/owais/django-webpack-loader)但是我更喜欢一个像上面这样简单的方法。

接下来在`templates`目录中创建包含以下内容的`index.html`:

```
{# djact/apps/core/templates/index.html #}
{% load static %}
{% load load_react %}
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8"/>
        <title>Djact</title>
        <link rel="icon" href="{% static 'favicon.ico' %}">
    </head>
    <body>
        <div id="app"></div>
        {% load_react %}
    </body>
</html>
```

# 证明

接下来我们需要一个 app 进行认证，所以`mkdir djact/apps/authentication`和`python manage.py startapp authentication djact/apps/authentication`。在这个目录中编辑`models.py`文件:

```
# djact/apps/authentication/models.py
from django.contrib.auth.models import AbstractUserclass User(AbstractUser):
    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users' def __str__(self):
        return f'<{self.id}> {self.username}'
```

接下来，我们需要一个用户注册的序列化器:

```
# djact/apps/authentication/serializers.py
from rest_framework import serializersfrom .models import Userclass UserSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(required=True)
    username = serializers.CharField()
    password = serializers.CharField(min_length=8, write_only=True) class Meta:
        model = User
        fields = ('email', 'username', 'password')
        extra_kwargs = {'password': {'write_only': True}} def create(self, validated_data):
        password = validated_data.pop('password', None)
        instance = self.Meta.model(**validated_data)
        if password is not None:
            instance.set_password(password) instance.save() return instance
```

然后查看`djact/apps/authentication/views.py`:

```
# djact/apps/authentication/views.py
from rest_framework import permissions
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework.views import APIViewfrom .serializers import UserSerializerclass UserCreate(CreateAPIView):
    permission_classes = (permissions.AllowAny,)
    authentication_classes = ()
    serializer_class = UserSerializer user_create = UserCreate.as_view()class Protected(APIView):
    def get(self, request):
        return Response(data={'type': 'protected'})protected = Protected.as_view()
```

`Protected`视图是检查我们只有登录后才能访问页面。

对于 URL，我们将有到两个视图的路径，还可以获取和刷新 JWT:

```
# djact/apps/authentication/urls.py
from django.urls import path
from rest_framework_simplejwt import views as jwt_viewsfrom . import viewsapp_name = 'authentication'
urlpatterns = [
    path(
        'users/create/',
        views.user_create,
        name='sign-up'
    ),
    path(
        'token/obtain/',
        jwt_views.TokenObtainPairView.as_view(),
        name='token-create'
    ),
    path(
        'token/refresh/',
        jwt_views.TokenRefreshView.as_view(),
        name='token-refresh'
    ),
    path(
        'protected/',
        views.protected,
        name='protected'
    )
]
```

在`djact`处更新主`urls.py`:

```
# djact/urls.py
from django.contrib import admin
from django.urls import path, includeurlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('djact.apps.authentication.urls')),
]
```

# 设置

我喜欢新的`Pathlib`模块，所以让我们用它代替`os`来重写一切。我使用`django-environ`来处理环境变量，所以让我们安装那个`pip install django-environ && pip freeze > requirements.txt`。从现有的配置中复制`DJANGO_SECRET_KEY`，这样你就不需要生成一个新的了(尽管这很容易)。我们将把它放在一个`.env`文件中。

```
# djact/settings/settings.py
import pathlib
from datetime import timedeltaimport environBASE_DIR = pathlib.Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parentenv = environ.Env()# Quick-start development settings - unsuitable for production
# See [https://docs.djangoproject.com/en/3.0/howto/deployment/checklist/](https://docs.djangoproject.com/en/3.0/howto/deployment/checklist/)# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env.str('DJANGO_SECRET_KEY')# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env.bool('DJANGO_DEBUG', False)ALLOWED_HOSTS = env.list('DJANGO_ALLOWED_HOSTS', default=list())# Application definitionINSTALLED_APPS = [
    'djact.apps.authentication',
    'djact.apps.core','rest_framework','django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]ROOT_URLCONF = 'djact.urls'TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
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
]WSGI_APPLICATION = 'djact.wsgi.application'# Database
# [https://docs.djangoproject.com/en/3.0/ref/settings/#databases](https://docs.djangoproject.com/en/3.0/ref/settings/#databases)DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': str(BASE_DIR.joinpath('db.sqlite3')),
    }
}# Password validation
# [https://docs.djangoproject.com/en/3.0/ref/settings/#auth-password-validators](https://docs.djangoproject.com/en/3.0/ref/settings/#auth-password-validators)AUTH_PASSWORD_VALIDATORS = [
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
]AUTH_USER_MODEL = 'authentication.User'REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),  #
}
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=30),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=30),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': False,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'VERIFYING_KEY': None,
    'AUTH_HEADER_TYPES': ('JWT',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
    'TOKEN_TYPE_CLAIM': 'token_type',
}LOGIN_URL = '/login'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/login'# Internationalization
# [https://docs.djangoproject.com/en/3.0/topics/i18n/](https://docs.djangoproject.com/en/3.0/topics/i18n/)LANGUAGE_CODE = 'ru'TIME_ZONE = 'Europe/Moscow'USE_I18N = TrueUSE_L10N = TrueUSE_TZ = True# Static files (CSS, JavaScript, Images)
# [https://docs.djangoproject.com/en/3.0/howto/static-files/](https://docs.djangoproject.com/en/3.0/howto/static-files/)STATIC_URL = '/static/'
STATICFILES_DIRS = [
    PROJECT_ROOT.joinpath('static'),
]STATIC_ROOT = PROJECT_ROOT / 'public' / 'static'
pathlib.Path(STATIC_ROOT).mkdir(exist_ok=True, parents=True)MEDIA_URL = '/media/'
MEDIA_ROOT = PROJECT_ROOT / 'public' / 'media'
pathlib.Path(MEDIA_ROOT).mkdir(exist_ok=True, parents=True)# LoggingLOG_DIR = PROJECT_ROOT / 'log'
LOG_DIR.mkdir(exist_ok=True)LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'console': {
            'format': '%(levelname)-8s %(name)-12s %(module)s:%(lineno)s\n'
                      '%(message)s'
        },
        'file': {
            'format': '%(asctime)s %(levelname)-8s %(name)-12s '
                      '%(module)s:%(lineno)s\n%(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'console',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'file',
            'filename': LOG_DIR / 'django.log',
            'backupCount': 10,  # keep at most 10 files
            'maxBytes': 5 * 1024 * 1024  # 5MB
        },
    },
    'loggers': {
        'django.request': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}LOGGING['loggers'].update(
    {app: {
        'handlers': ['console', 'file'],
        'level': 'DEBUG',
        'propagate': True,
    } for app in INSTALLED_APPS}
)# Load dev configif DEBUG:
    try:
        from .dev import *
    except ModuleNotFoundError:
        print('Dev config not found')
```

我们可以在`djact/settings/dev.py`中覆盖一些设置或添加一些只与开发环境相关的东西，这就是为什么我们需要最后 5 行。我的`dev.py`长这样:

```
# djact/settings/dev.py
from .settings import LOGGING, INSTALLED_APPS, MIDDLEWARELOGGING['handlers']['file']['backupCount'] = 1INSTALLED_APPS += ['corsheaders']
CORS_ORIGIN_ALLOW_ALL = True
MIDDLEWARE.insert(2, 'corsheaders.middleware.CorsMiddleware')
```

这里我们告诉 Django 允许与我们的 react dev 服务器交互，它将运行在不同的端口上，因此被认为是跨源的。

我们的`.env.example`文件是这样的:

```
<!-- .env.example -->
PYTHONDONTWRITEBYTECODE=1DJANGO_SECRET_KEY=random long string
DJANGO_DEBUG=True for dev environment|False or omit completely for production
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1:8000,0.0.0.0:8000
```

因此，用这些变量创建一个`.env`文件。

现在在包含以下内容的`djact/apps/core/`目录中创建`urls.py`:

```
# djact/apps/core/urls.py
from django.urls import re_path
from django.views.generic import TemplateViewapp_name = 'core'
urlpatterns = [
    re_path(r'^.*$', TemplateView.as_view(template_name='index.html'), name='index'),
]
```

并更新主 URL 文件:

```
# djact/urls.py
from django.contrib import admin
from django.urls import path, includeurlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('djact.apps.authentication.urls')),
    path('', include('djact.apps.core.urls')),
]
```

然后运行`python manage.py makemigrations`和`python manage.py migrate`。

我们的目录结构应该如下所示:

```
.
├── djact
│   ├── apps
│   │   ├── authentication
│   │   │   ├── admin.py
│   │   │   ├── apps.py
│   │   │   ├── __init__.py
│   │   │   ├── migrations
│   │   │   │   ├── 0001_initial.py
│   │   │   │   └── __init__.py
│   │   │   ├── models.py
│   │   │   ├── serializers.py
│   │   │   ├── urls.py
│   │   │   └── views.py
│   │   └── core
│   │       ├── admin.py
│   │       ├── apps.py
│   │       ├── __init__.py
│   │       ├── migrations
│   │       │   └── __init__.py
│   │       ├── templates
│   │       │   └── index.html
│   │       ├── templatetags
│   │       │   ├── __init__.py
│   │       │   └── load_react.py
│   │       └── urls.py
│   ├── asgi.py
│   ├── __init__.py
│   ├── settings
│   │   ├── dev.py
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── urls.py
│   └── wsgi.py
├── .env
├── .env.example
├── manage.py
└── requirements.txt
```

# 创建 React 应用程序

让我们`mkdir`看看我们的 React 前端，并深入了解它— `mkdir frontend && cd frontend`。

首先用`yarn init`初始化前端项目，回答问题。以下是我的例子:

```
$ yarn init
yarn init v1.22.4
question name (frontend): djact
question version (1.0.0):
question description: Django + React
question entry point (index.js):
question repository url:
question author: Constantine
question license (MIT):
question private:
success Saved package.json
Done in 34.53s.
```

现在我们可以用`yarn add react react-dom axios react-redux redux redux-thunk reselect`添加依赖项。以及我们对`yarn add -D eslint babel-eslint babel-polyfill eslint-plugin-import eslint-plugin-react eslint-plugin-react-hooks eslint-loader style-loader css-loader postcss-loader webpack-dev-server mini-css-extract-plugin cssnano html-webpack-plugin npm-run-all rimraf redux-immutable-state-invariant webpack webpack-cli babel-loader @babel/core @babel/node @babel/preset-env @babel/preset-react`的开发依赖。

# 配置

在当前目录下创建`.eslintrc.js`，内容如下:

```
// frontend/.eslintrc.js
module.exports = {
  parser: "babel-eslint",
  env: {
    browser: true,
    commonjs: true,
    es6: true,
    node: true,
    jest: true,
  },
  parserOptions: {
    ecmaVersion: 2020,
    ecmaFeatures: {
      impliedStrict: true,
      jsx: true,
    },
    sourceType: "module",
  },
  plugins: ["react", "react-hooks"],
  extends: [
    "eslint:recommended",
    "plugin:react/recommended",
    "plugin:react-hooks/recommended",
  ],
  settings: {
    react: {
      version: "detect",
    },
  },
  rules: {
    "no-debugger": "off",
    "no-console": "off",
    "no-unused-vars": "warn",
    "react/prop-types": "warn",
  },
};
```

Babel 配置存储在`babel.config.js`中:

```
// frontend/babel.config.js
module.exports = {
  presets: ["[@babel/preset-env](http://twitter.com/babel/preset-env)", "[@babel/preset-react](http://twitter.com/babel/preset-react)"],
};
```

存储在`webpack.config.dev.js`中的开发环境的 Webpack 配置:

```
// frontend/webpack.config.dev.js
const webpack = require("webpack");
const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");process.env.NODE_ENV = "development";module.exports = {
  mode: "development",
  target: "web",
  devtool: "cheap-module-source-map",
  entry: ["babel-polyfill", "./src/index"],
  output: {
    path: path.resolve(__dirname),
    publicPath: "/",
    filename: "bundle.js",
  },
  devServer: {
    stats: "minimal",
    overlay: true,
    historyApiFallback: true,
    disableHostCheck: true,
    headers: { "Access-Control-Allow-Origin": "*" },
    https: false,
  },
  plugins: [
    new webpack.DefinePlugin({
      "process.env.API_URL": JSON.stringify("http://localhost:8000/api/"),
    }),
    new HtmlWebpackPlugin({
      template: "./src/index.html",
      favicon: "./src/favicon.ico",
    }),
  ],
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: [
          {
            loader: "babel-loader",
          },
          "eslint-loader",
        ],
      },
      {
        test: /(\.css)$/,
        use: ["style-loader", "css-loader"],
      },
    ],
  },
};
```

并编辑`package.json` `scripts`部分，使其看起来像这样:

```
// frontend/package.json
{
  "name": "djact",
  "version": "1.0.0",
  "description": "Django + React",
  "scripts": {
    "start:dev": "webpack-dev-server --config webpack.config.dev.js --port 3000",
    "clean:build": "rimraf ../static && mkdir ../static",
    "prebuild": "run-p clean:build",
    "build": "webpack --config webpack.config.prod.js",
    "postbuild": "rimraf ../static/index.html"
  },
  "main": "index.js",
  "author": "Constantine",
  "license": "MIT",
  "dependencies": {
    ...
  },
  "devDependencies": {
    ...
  }
}
```

现在让我们为前端源代码添加一个目录:`mkdir -p src/components`。还要为 React — `touch src/index.js`创建入口点，内容如下:

```
// frontend/src/index.js
import React from "react";
import { render } from "react-dom";
import { BrowserRouter as Router } from "react-router-dom";
import App from "./components/App";render(
  <Router>
    <App />
  </Router>,
  document.getElementById("app")
);
```

创建`html`模板— `touch src/index.html`:

```
<!-- frontend/src/index.html -->
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Djact</title>
  </head><body>
    <div id="app"></div>
  </body>
</html>
```

如果你喜欢，你可以在`src`目录中添加一个图标。

然后创建`App`组件— `touch src/components/App.js`。让它返回一些简单的东西:

```
// frontend/src/components/App.js
import React from "react";function App() {
  return <h1>Hello from React!</h1>;
}export default App;
```

我们现在可以测试我们的应用程序与`yarn start:dev`一起工作。在导航到 http://localhost:3000 之后，我们应该会看到一个“来自 React 的 Hello！”问候！

这里是一部作品`webpack.config.prod.js`:

```
// frontend/webpack.config.prod.js
const webpack = require("webpack");
const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const MiniCssExtractPlugin = require("mini-css-extract-plugin");process.env.NODE_ENV = "production";module.exports = {
  mode: "production",
  target: "web",
  devtool: "source-map",
  entry: {
    vendor: ["react", "react-dom", "prop-types"],
    bundle: ["babel-polyfill", "./src/index"],
  },
  output: {
    path: path.resolve(__dirname, "../static"),
    publicPath: "/",
    filename: "[name].[contenthash].js",
  },
  plugins: [
    new MiniCssExtractPlugin({
      filename: "[name].[contenthash].css",
    }),
    new webpack.DefinePlugin({
      // This global makes sure React is built in prod mode.
      "process.env.NODE_ENV": JSON.stringify(process.env.NODE_ENV),
      "process.env.API_URL": JSON.stringify("[http://localhost:8000/api/](http://localhost:8000/api/)"),
    }),
    new HtmlWebpackPlugin({
      template: "src/index.html",
      favicon: "./src/favicon.ico",
      minify: {
        // see [https://github.com/kangax/html-minifier#options-quick-reference](https://github.com/kangax/html-minifier#options-quick-reference)
        removeComments: true,
        collapseWhitespace: true,
        removeRedundantAttributes: true,
        useShortDoctype: true,
        removeEmptyAttributes: true,
        removeStyleLinkTypeAttributes: true,
        keepClosingSlash: true,
        minifyJS: true,
        minifyCSS: true,
        minifyURLs: true,
      },
    }),
  ],
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: [
          {
            loader: "babel-loader",
          },
          "eslint-loader",
        ],
      },
      {
        test: /(\.css)$/,
        use: [
          MiniCssExtractPlugin.loader,
          {
            loader: "css-loader",
            options: {
              sourceMap: true,
            },
          },
          {
            loader: "postcss-loader",
            options: {
              plugins: () => [require("cssnano")],
              sourceMap: true,
            },
          },
        ],
      },
    ],
  },
};
```

现在我们可以`yarn build`并在`static`目录中看到我们的捆绑文件。如果我们通过`python manage.py runserver 0.0.0.0:8000`启动 Django 应用程序，我们会看到完全相同的东西，但是是在生产模式下运行。

我们的项目目录应该如下所示:

```
.
├── djact
│   ├── apps
│   │   ├── authentication
│   │   │   ├── admin.py
│   │   │   ├── apps.py
│   │   │   ├── __init__.py
│   │   │   ├── migrations
│   │   │   │   ├── 0001_initial.py
│   │   │   │   └── __init__.py
│   │   │   ├── models.py
│   │   │   ├── serializers.py
│   │   │   ├── urls.py
│   │   │   └── views.py
│   │   └── core
│   │       ├── admin.py
│   │       ├── apps.py
│   │       ├── __init__.py
│   │       ├── migrations
│   │       │   └── __init__.py
│   │       ├── templates
│   │       │   └── index.html
│   │       ├── templatetags
│   │       │   ├── __init__.py
│   │       │   └── load_react.py
│   │       └── urls.py
│   ├── asgi.py
│   ├── db.sqlite3
│   ├── __init__.py
│   ├── settings
│   │   ├── dev.py
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── urls.py
│   └── wsgi.py
├── .env
├── .env.example
├── frontend
│   ├── babel.config.js
│   ├── package.json
│   ├── src
│   │   ├── components
│   │   │   └── App.js
│   │   ├── favicon.ico
│   │   ├── index.html
│   │   └── index.js
│   ├── webpack.config.dev.js
│   ├── webpack.config.prod.js
│   └── yarn.lock
├── log
│   └── django.log
├── manage.py
├── public
│   ├── media
│   └── static
├── requirements.txt
└── static
    ├── bundle.76ba356d74f1017eda2f.js
    ├── bundle.76ba356d74f1017eda2f.js.map
    ├── favicon.ico
    ├── vendor.9245c714f84f4bbf6bdc.js
    └── vendor.9245c714f84f4bbf6bdc.js.map
```

# API 服务

在`components`目录内创建`axiosApi.js`:

```
// frontend/src/components/api/axiosApi.js
import axios from "axios";const baseURL = process.env.API_URL;
const accessToken = localStorage.getItem("access_token");const axiosAPI = axios.create({
  baseURL: baseURL,
  timeout: 5000,
  headers: {
    Authorization: accessToken ? "JWT " + accessToken : null,
    "Content-Type": "application/json",
    accept: "application/json",
  },
});axiosAPI.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;// Prevent infinite loops
    if (
      error.response.status === 401 &&
      originalRequest.url === baseURL + "token/refresh/"
    ) {
      window.location.href = "/login/";
      return Promise.reject(error);
    }if (
      error.response.status === 401 &&
      error.response.statusText === "Unauthorized"
    ) {
      const refresh = localStorage.getItem("refresh_token");if (refresh) {
        const tokenParts = JSON.parse(atob(refresh.split(".")[1]));// exp date in token is expressed in seconds, while now() returns milliseconds:
        const now = Math.ceil(Date.now() / 1000);if (tokenParts.exp > now) {
          try {
            const response = await axiosAPI.post("/token/refresh/", {
              refresh,
            });
            setNewHeaders(response);
            originalRequest.headers["Authorization"] =
              "JWT " + response.data.access;
            return axiosAPI(originalRequest);
          } catch (error) {
            console.log(error);
          }
        } else {
          console.log("Refresh token is expired", tokenParts.exp, now);
          window.location.href = "/login/";
        }
      } else {
        console.log("Refresh token not available.");
        window.location.href = "/login/";
      }
    }// specific error handling done elsewhere
    return Promise.reject(error);
  }
);export function setNewHeaders(response) {
  axiosAPI.defaults.headers["Authorization"] = "JWT " + response.data.access;
  localStorage.setItem("access_token", response.data.access);
  localStorage.setItem("refresh_token", response.data.refresh);
}export default axiosAPI;
```

和`authenticationApi.js`:

```
// frontend/src/components/api/authenticationApi.js
import axiosAPI, { setNewHeaders } from "./axiosApi";export async function signUp(email, username, password) {
  const response = await axiosAPI.post("users/create/", {
    email,
    username,
    password,
  });
  localStorage.setItem("user", response.data);
  return response;
}export async function obtainToken(username, password) {
  const response = await axiosAPI.post("token/obtain/", {
    username,
    password,
  });
  setNewHeaders(response);
  return response;
}export async function refreshToken(refresh) {
  const response = await axiosAPI.post("token/refresh/", {
    refresh,
  });
  setNewHeaders(response);
  return response;
}// eslint-disable-next-line
export async function logout(accessToken) {
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
  // TODO: invalidate token on backend
}export const isAuthenticated = () => {
  const token = localStorage.getItem("access_token");
  return !!token;
};
```

# Redux

首先在`djact/frontend/src/`下创建`redux`目录，放入以下文件:

```
// frontend/src/redux/configureStore.dev.js
import { createStore, applyMiddleware, compose } from "redux";
import rootReducer from "./reducers";
import reduxImmutableStateInvariant from "redux-immutable-state-invariant";
import thunk from "redux-thunk";export default function configureStore(initialState) {
  const composeEnhancers =
    window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose; // add support for Redux dev toolsreturn createStore(
    rootReducer,
    initialState,
    composeEnhancers(applyMiddleware(thunk, reduxImmutableStateInvariant()))
  );
}
```

```
// frontend/src/redux/configureStore.prod.js
import { createStore, applyMiddleware } from "redux";
import rootReducer from "./reducers";
import thunk from "redux-thunk";export default function configureStore(initialState) {
  return createStore(rootReducer, initialState, applyMiddleware(thunk));
}
```

```
// frontend/src/redux/configureStore.js
// Use CommonJS require below so we can dynamically import during build-time.
if (process.env.NODE_ENV === "production") {
  module.exports = require("./configureStore.prod");
} else {
  module.exports = require("./configureStore.dev");
}
```

商店已配置，现在开始行动！用以下文件在`redux`中创建`actions`目录:

```
// frontend/src/redux/actions/types.js
export const LOGIN_USER_SUCCESS = "LOGIN_USER_SUCCESS";
export const LOGOUT_USER = "LOGOUT_USER";
```

```
// frontend/src/redux/actions/auth.js
import { LOGIN_USER_SUCCESS, LOGOUT_USER } from "./types";
import { obtainToken, logout } from "../../components/api/authenticationApi";export function loginUserSuccess(token) {
  return { type: LOGIN_USER_SUCCESS, token };
}export function loginUser(username, password) {
  return async function (dispatch) {
    try {
      const response = await obtainToken(username, password);
      dispatch(loginUserSuccess(response.data.access));
    } catch (error) {
      console.log("Error obtaining token. " + error);
    }
  };
}export function logoutUserSuccess() {
  return { type: LOGOUT_USER };
}export function logoutUser() {
  return async function (dispatch) {
    await logout();
    dispatch(logoutUserSuccess());
  };
}
```

redux 的最后一步是 reducers 本身，在`frontend/src/redux/reducers`目录中。

```
// frontend/src/redux/reducers/initialState.js
export default {
  accessToken: localStorage.getItem("access_token"),
};
```

```
// frontend/src/redux/reducers/auth.js
import * as types from "../actions/types";
import initialState from "./initialState";export default function authReducer(state = initialState.accessToken, action) {
  switch (action.type) {
    case types.LOGIN_USER_SUCCESS:
      return action.token;
    case types.LOGOUT_USER:
      return "";
    default:
      return state;
  }
}
```

```
// frontend/src/redux/reducers/index.js
import { combineReducers } from "redux";
import auth from "./auth";const rootReducer = combineReducers({
  auth,
});export default rootReducer;
```

现在我们需要在`index.js`中注册所有内容:

```
import React from "react";
import {render} from "react-dom";
import {BrowserRouter as Router} from "react-router-dom";
import {Provider as ReduxProvider} from "react-redux";
import App from "./components/App";
import configureStore from "./redux/configureStore";*const* store = configureStore();render(
  <*ReduxProvider* store={store}>
    <*Router*>
      <*App*/>
    </*Router*>
  </*ReduxProvider*>,
  document.getElementById("app")
);
```

# 成分

## 证明

我们已经准备好了减速器，现在我们需要把它们投入使用。所以让我们在`frontend/src/components`中创建`authentication`目录，并将接下来的三个文件放在那里。

这将是我们专用路由的包装:

```
// frontend/src/components/authentication/PrivateRoute.js
import React from "react";
import { Redirect, Route } from "react-router-dom";
import PropTypes from "prop-types";
import { isAuthenticated } from "../api/authenticationApi";const PrivateRoute = ({ component: Component, ...rest }) => (
  <Route
    {...rest}
    render={(props) =>
      isAuthenticated() ? (
        <Component {...props} />
      ) : (
        <Redirect
          to={{ pathname: "/login", state: { from: props.location } }}
        />
      )
    }
  />
);PrivateRoute.propTypes = {
  component: PropTypes.func.isRequired,
  location: PropTypes.object,
};export default PrivateRoute;
```

```
// frontend/src/components/authentication/LoginPage.js
import React, { useState } from "react";
import { connect } from "react-redux";
import PropTypes from "prop-types";
import { loginUser } from "../../redux/actions/auth";const LoginPage = ({ loginUser, history }) => {
  const [state, setState] = useState({
    username: "",
    password: "",
  });const handleChange = (event) => {
    const { name, value } = event.target;
    setState({ ...state, [name]: value });
  };const login = async (event) => {
    event.preventDefault();
    const { username, password } = state;await loginUser(username, password);
    history.push("/");
  };return (
    <div>
      <h1>Login page</h1>
      <form onSubmit={login}>
        <label>
          Username:
          <input
            name="username"
            type="text"
            value={state.username}
            onChange={handleChange}
          />
        </label>
        <label>
          Password:
          <input
            name="password"
            type="password"
            value={state.password}
            onChange={handleChange}
          />
        </label>
        <input type="submit" value="Submit" />
      </form>
    </div>
  );
};LoginPage.propTypes = {
  loginUser: PropTypes.func.isRequired,
  history: PropTypes.object.isRequired,
};const mapDispatchToProps = {
  loginUser,
};export default connect(null, mapDispatchToProps)(LoginPage);
```

注册组件很简单，因为我不想实现它，但是应该很简单:

```
// frontend/src/components/authentication/SignUpPage.js
import React from "react";
import { useHistory } from "react-router-dom";const SignUpPage = () => {
  const history = useHistory();const handleClick = () => {
    history.push("/");
  };return (
    <div>
      <h1>Sign Up page</h1>
      <button onClick={handleClick}>sign up</button>
    </div>
  );
};export default SignUpPage;
```

## 普通的

通用组件将只包含标题。但是从理论上讲，这里可以生活任何东西..你是知道的..常见。

```
// frontend/src/components/common/Header.js
import React from "react";
import { connect } from "react-redux";
import PropTypes from "prop-types";
import { NavLink, useHistory } from "react-router-dom";
import { logoutUser } from "../../redux/actions/auth";const Header = ({ accessToken, logoutUser }) => {
  const history = useHistory();const handleLogout = async () => {
    await logoutUser();
    history.push("login/");
  };return (
    <nav>
      {accessToken ? (
        <>
          <NavLink to="/">Profile</NavLink>
          {" | "}
          <NavLink to="/logout" onClick={handleLogout}>
            Logout
          </NavLink>
        </>
      ) : (
        <>
          <NavLink to="/login">Login</NavLink>
          {" | "}
          <NavLink to="/sign-up">SignUp</NavLink>
        </>
      )}
    </nav>
  );
};Header.propTypes = {
  accessToken: PropTypes.string,
  logoutUser: PropTypes.func.isRequired,
};function mapStateToProps(state) {
  return {
    accessToken: state.auth,
  };
}const mapDispatchToProps = {
  logoutUser,
};export default connect(mapStateToProps, mapDispatchToProps)(Header);
```

## 核心

最后一部分是具有应用程序逻辑的核心组件。这里我们将有我们的受保护页面:

```
// frontend/src/components/core/ProfilePage.js
import React from "react";
import axiosAPI from "../api/axiosApi";const ProfilePage = () => {
  const handleClick = async () => {
    const response = await axiosAPI.get("protected/");
    alert(JSON.stringify(response.data));
  };
  return (
    <div>
      <h1>Profile page</h1>
      <p>Only logged in users should see this</p>
      <button onClick={handleClick}>GET protected</button>
    </div>
  );
};export default ProfilePage;
```

最后要做的是更新我们的`App.js`:

```
// frontend/src/components/App.js
import React from "react";
import {Route, Switch} from "react-router-dom";import PageNotFound from "./PageNotFound";
import Header from "./common/Header";
import ProfilePage from "./core/ProfilePage";
import PrivateRoute from "./authentication/PrivateRoute";
import LoginPage from "./authentication/LoginPage";
import SignUpPage from "./authentication/SignUpPage";function App() {
  return (
    <>
      <Header/>
      <Switch>
        <PrivateRoute exact path="/" component={ProfilePage}/>
        <Route path="/login" component={LoginPage}/>
        <Route path="/sign-up" component={SignUpPage}/>
        <Route component={PageNotFound}/>
      </Switch>
    </>
  );
}export default App;
```

我们最终的项目结构应该是这样的:

```
.
├── blogpost.md
├── djact
│   ├── apps
│   │   ├── authentication
│   │   │   ├── admin.py
│   │   │   ├── apps.py
│   │   │   ├── __init__.py
│   │   │   ├── migrations
│   │   │   │   ├── 0001_initial.py
│   │   │   │   └── __init__.py
│   │   │   ├── models.py
│   │   │   ├── serializers.py
│   │   │   ├── urls.py
│   │   │   └── views.py
│   │   └── core
│   │       ├── admin.py
│   │       ├── apps.py
│   │       ├── __init__.py
│   │       ├── migrations
│   │       │   └── __init__.py
│   │       ├── templates
│   │       │   └── index.html
│   │       ├── templatetags
│   │       │   ├── __init__.py
│   │       │   └── load_react.py
│   │       └── urls.py
│   ├── asgi.py
│   ├── db.sqlite3
│   ├── __init__.py
│   ├── settings
│   │   ├── dev.py
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── urls.py
│   └── wsgi.py
├── .env
├── .env.example
├── frontend
│   ├── babel.config.js
│   ├── package.json
│   ├── src
│   │   ├── components
│   │   │   ├── api
│   │   │   │   ├── authenticationApi.js
│   │   │   │   └── axiosApi.js
│   │   │   ├── App.js
│   │   │   ├── authentication
│   │   │   │   ├── LoginPage.js
│   │   │   │   ├── PrivateRoute.js
│   │   │   │   └── SignUpPage.js
│   │   │   ├── common
│   │   │   │   └── Header.js
│   │   │   ├── core
│   │   │   │   └── ProfilePage.js
│   │   │   └── PageNotFound.js
│   │   ├── favicon.ico
│   │   ├── index.html
│   │   ├── index.js
│   │   └── redux
│   │       ├── actions
│   │       │   ├── auth.js
│   │       │   └── types.js
│   │       ├── configureStore.dev.js
│   │       ├── configureStore.js
│   │       ├── configureStore.prod.js
│   │       └── reducers
│   │           ├── auth.js
│   │           ├── index.js
│   │           └── initialState.js
│   ├── webpack.config.dev.js
│   ├── webpack.config.prod.js
│   ├── yarn-error.log
│   └── yarn.lock
├── log
│   └── django.log
├── manage.py
├── public
│   ├── media
│   └── static
├── requirements.txt
└── static
    ├── bundle.c86ace9a42dd5bd70a59.js
    ├── bundle.c86ace9a42dd5bd70a59.js.map
    ├── favicon.ico
    ├── vendor.0d40e04c29796a70dc89.js
    └── vendor.0d40e04c29796a70dc89.js.map
```

# 运转

现在，设置环境变量`export $(cat .env | xargs)`。构建前端部分`cd frontend && yarn:build`。用`cd ../ && python manage.py createsuperuser`创建超级用户进行测试，并按照说明操作。运行 Django app `python manage.py runserver`，导航到 [http://localhost:8000](http://localhost:8000) 。我们应该会看到我们的登录页面。输入您在创建超级用户时提供的凭证，我们将进入受保护的配置文件页面。如果我们点击一个`GET protected`按钮，我们会看到来自服务器的响应警告。

就是这样！如果你大老远来到这里..哇！如果你真的实现了所有这些..哇！！干得好，我的朋友！希望你学到了新东西或者解决了你的一个问题🚀

谢谢大家，编码快乐！

# 资源

正如我在本文开始时所承诺的，这里列出了帮助我构建这一切的所有资源:

[PluralSight](https://pluralsight.com/) 课程:

*   [Cory House](https://app.pluralsight.com/library/courses/react-redux-react-router-es6/table-of-contents)[使用 React 和 Redux](https://app.pluralsight.com/profile/author/cory-house) 构建应用程序
*   [由](https://app.pluralsight.com/library/courses/react-auth0-authentication-security/table-of-contents) [Cory House](https://app.pluralsight.com/profile/author/cory-house) 使用 Auth0 保护 React 应用
*   [高级 React.js](https://app.pluralsight.com/library/courses/reactjs-advanced/table-of-contents) 作者[萨梅尔布纳](https://app.pluralsight.com/profile/author/samer-buna)

文章:

*   [110%完成与 Django 的 JWT 认证& React — 2020](https://hackernoon.com/110percent-complete-jwt-authentication-with-django-and-react-2020-iejq34ta) 作者[斯图尔特·莱奇](https://hackernoon.com/u/Toruitas)
    - [React + Redux — JWT 认证教程&示例](https://jasonwatmore.com/post/2017/12/07/react-redux-jwt-authentication-tutorial-example)作者[杰森·瓦特莫尔](https://jasonwatmore.com/)
    - [在 React + Redux 应用中使用 JWT 进行授权](https://levelup.gitconnected.com/using-jwt-in-your-react-redux-app-for-authorization-d31be51a50d2)作者[莱兹尔·萨马诺](https://levelup.gitconnected.com/@leizl.samano)