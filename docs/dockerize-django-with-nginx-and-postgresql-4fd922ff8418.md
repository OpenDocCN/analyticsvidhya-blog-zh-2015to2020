# 用 Nginx 和 PostgreSQL 实现 Django

> 原文：<https://medium.com/analytics-vidhya/dockerize-django-with-nginx-and-postgresql-4fd922ff8418?source=collection_archive---------0----------------------->

“docker 是 Linux 的 Inception”，这是我在解释 Docker 时喜欢用的说法。因为 docker 就是这样，一个 Linux 在一个共享主机内核的 Linux 里面。对于任何有同情心的 Linux 用户来说，可以在几天内学会 docker。在本文中，我将讨论用 docker-compose 在生产环境(Nginx、PostgreSQL 和 Gunicorn)中对一个完整的 Django 应用程序进行 dockerizing。所以，让我们开始-

![](img/b38e1fae9f73c6ef8189f5ab978f1bd0.png)

燕尾服和码头工人作者— [伊藤京平](https://www.flickr.com/photos/134416355@N07/)

**满足先决条件**

*   安装 docker
*   安装 docker-compose

**获取代码**

[](https://github.com/by-sabbir/django-gunicorn-nginx-docker) [## by-sab Bir/django-guni corn-nginx-docker

### 当涉及到部署和 CI/CD 时，Docker 会让您的生活稍微轻松一些。这种方法可用于部署大多数…

github.com](https://github.com/by-sabbir/django-gunicorn-nginx-docker) 

在根目录(包含 docker-compose.yml 的同一个目录)创建一个. env 文件，并按如下所示进行编辑

```
SECRET_KEY="YOUR SECRET KEY"
DB_HOST="db"
DB_PASSWORD="hello"   #from docker-compose.yml
DB_USER="postgres"    #from docker-compose.yml
DB_NAME="postgres"    #from docker-compose.yml
DB_PORT="5432"        #from docker-compose.yml
DB_ENGINE="django.db.backends.postgresql"
```

使用以下两个命令`docker-compose build`，系统将启动并运行

这将从 docker hub 下载所有的依赖项和 docker 映像

`docker-compose up -d`

最后，我们必须将静态文件复制到我们指定的 settings.py 目录下，我们将这样做，

`docker-compose exec web python manage.py collectstatic --no-input`

希望现在您的项目已经启动并运行了，您可以在 [localhost:8008](http://localhost:8008/) 访问它。如果没有，使用`docker-compose logs -f` 进行调试。

**设计/思考过程**

基本上，问题是在生产环境中运行 docker 内的 Django 应用程序，意味着使用 Nginx 作为 HTTP 服务器，Postgresql 作为 DB 服务器。这是我的计划——我将在不同的容器中运行应用程序、HTTP 服务器和 DB 服务器，使其更易于管理和健壮。该应用程序将在 Gunicorn 上运行，稍后将使用 Nginx 的代理通行证。我们需要一个用于 HTTP 服务器的共享卷，一个用于静态和媒体文件托管的应用程序，以及一个用于 DB 的持久文件系统。所有的秘密都将储存在一个完全独立的。环境文件。

让我们来看看 web 服务:

```
web:
    build: .
    container_name: test_deploy_web
    command: gunicorn app.wsgi:application --bind 0.0.0.0:8000
    volumes:
      - ./app:/app/
      - staticfiles:/app/static/
    expose:
      - 8000
    env_file:
      - ./.env
    depends_on:
      - db
```

第二行是“构建”意味着它将从根 docker 文件创建一个 docker 映像，让我们来看看这个文件，

```
FROM python:latest
  ENV PYTHONDONTWRITEBYTECODE 1
  ENV PYTHONUNBUFFERED 1
  RUN mkdir /app
  WORKDIR /app
  RUN pip install --upgrade pip
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY ./app /app
```

法利简单易懂。让我们进入下一个阶段，数据库

```
db:
    image: postgres
    container_name: test_deploy_db
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    environment:
      - POSTGRES_DB=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=hello
```

我们将只使用官方的 Postgres docker 映像，postgres_data 是 docker 中的持久数据量。应该够了。

现在 Nginx。你可以用它做任何你想做的事。Nginx 真正带给你的是极致的力量。

```
nginx:
    build: ./nginx
    container_name: test_deploy_nginx
    volumes:
      - staticfiles:/app/static/
    ports:
      - 8008:80
    depends_on:
      - web
```

因此，让我们深入了解一下这个组合配置。第二行表示它将编译‘nginx’文件夹中的任何 docker 文件。让我们跟着线索走-

```
FROM nginx:1.19.0-alpine
RUN mkdir /app
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/
WORKDIR /app
```

哇，那很容易。简单地说，我们只是删除了默认的配置文件，并用我们自己的替换它。问题是我们的配置文件中有什么，让我们看一个简单的版本-

```
upstream app {
    server web:8000;
}server {
    listen 80;
    location / {
        proxy_pass [http://app;](http://app;)
    }
    location /static/ {
        alias /app/static/;
    }
}
```

Nginx 的上游配置代理请求服务器。Gunicorn 是我们 Django 应用程序的 Python WSGI 服务器。在第一个位置派生中，我们将“/”根位置从应用程序代理传递到 HTTP 服务器。二阶位置导数只是静态文件的别名。现在，我们可以用 Nginx 服务器来发挥创造力。

我想就是这里了。如果我错过了什么，请告诉我。[自述文件](https://github.com/by-sabbir/django-gunicorn-nginx-docker#dockerize-django-along-with-nginx-and-postgresql)有更详细的命令。你可以随意修改代码，我会检查 Github 的问题，到时见。