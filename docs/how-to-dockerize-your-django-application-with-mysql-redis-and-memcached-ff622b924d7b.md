# 如何用 mysql，redis，memcached 将你的 django 应用 dockerize？

> 原文：<https://medium.com/analytics-vidhya/how-to-dockerize-your-django-application-with-mysql-redis-and-memcached-ff622b924d7b?source=collection_archive---------7----------------------->

在这个故事中，我将尝试解释如何对一个 enter django 应用程序进行 dockerize。首先，安装 docker 和 docker compose。

现在，我们将为应用程序维护一个 docker 文件，为控制 docker 实例(mysql、redis、memcached)维护另一个 docker-compose.yml 文件。我们的目录结构将是:

```
base_directory
 - Dockerfile
 - docker-compose.yml
 - app
 - requirements.txt
```

**Dockerfile:**

这个文件将用于安装 django 应用程序的不同依赖项，将本地应用程序目录复制到 docker 实例。

```
**FROM** python:3.5
**ENV** PYTHONUNBUFFERED 1
**RUN** mkdir /base_directory
**WORKDIR** /base_directory
**ADD** . /base_directory/
**RUN** apt-get update
**RUN** apt-get install -y git
**RUN** git init
**RUN** apt-get install -y gcc python3-dev
**RUN** apt-get install -y libxml2-dev libxslt1-dev build-essential python3-lxml zlib1g-dev
**RUN** apt-get install -y default-mysql-client default-libmysqlclient-dev
**RUN** wget https://bootstrap.pypa.io/get-pip.py
**RUN**  python3 get-pip.py
**RUN** rm get-pip.py
**RUN** pip install -r requirements.txt
```

**docker-compose.yml:**

该文件将用于下载图像，下载后将创建一些容器并运行这些容器。

```
**version**: **'2'

services**:
  **db**:
    **container_name**: **'db'
    image**: mysql:5.7
    **ports**:
      - **'3307:3306'
    environment**:
       **MYSQL_DATABASE**: **'appdb'
       MYSQL_PASSWORD**: **'root'
       MYSQL_ROOT_PASSWORD**: **'root'
    volumes**:
      - **'db:/var/lib/mysql'
    command**: [**'mysqld'**, **'--character-set-server=utf8mb4'**, **'--collation-server=utf8mb4_unicode_ci'**] **redis**:
    **container_name**: **'redis'
    image**: **'redis:3.2.0'
    ports**:
      - **'6378:6379'
    volumes**:
      - **'redisdata:/data'** **memcached**:
    **container_name**: **'memcached'
    image**: **'memcached:latest'
    ports**:
      - **"11212:11211"** **web**:
    **build**: .
    **command**: python3 app/manage.py runserver 0.0.0.0:8001 
    **volumes**:
      - **'.:/base_directory'
    ports**:
      - **"8001:8001"
    depends_on**:
      - db
      - redis
      - memcached

**volumes**:
  **redisdata**:
  **db**:
  **.**:
```

Dockerfile 正在通过 web 容器下的 build 命令从此 docker-compose 运行。在我们的设置文件中，我们的数据库设置部分将是:

```
DATABASES = {
    **'default'**: {
        **'ENGINE'**: **'django.db.backends.mysql'**,
        **'NAME'**: **'appdb'**,
        **'HOST'**: **'db'**,
        **'PORT'**: 3306,
        **'USER'**: **'root'**,
        **'PASSWORD'**: **'pass'**,
        **'OPTIONS'**: {**'sql_mode'**: **'STRICT_ALL_TABLES'**, **'charset'**: **'utf8mb4'**,},
    }
}
```

我们将使用以下方法来包装容器:

```
docker-compose up
```

我们可以在后台运行它们:

```
docker-compose up -d 
```

我们可以通过以下方式制作集装箱:

```
docker-compose down
```

我们可以从 docker 卷中删除所有内容:

```
docker-compose down -v
```

我们可以通过以下方式进入任何集装箱的码头:

```
docker exec -it container_name/id /bin/bash
```

我们可以通过以下方式导入 mysql 数据库:

```
docker exec -i container mysql -uuser -ppass db <db.sql
```

如果我们想先运行数据库，然后等待它完全准备好，在这种情况下，将首先运行数据库:

```
docker-compose up db
```

然后其他人也上来了:

```
docker-compose up
```

最后，我们的应用程序将在 8001 端口上启动:

```
[http://localhost:8001/](http://localhost:8001/)
```