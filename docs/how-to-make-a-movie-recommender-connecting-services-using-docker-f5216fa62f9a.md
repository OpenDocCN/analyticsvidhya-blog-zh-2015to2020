# 如何制作电影推荐器:使用 Docker 连接服务

> 原文：<https://medium.com/analytics-vidhya/how-to-make-a-movie-recommender-connecting-services-using-docker-f5216fa62f9a?source=collection_archive---------11----------------------->

Docker 是一个很好的工具，可以在保持整洁的同时进行部署和测试。这个项目的代码在这里是[，本教程的代码在同一个存储库中。](https://github.com/jdortuzar5/movie-recommender)

# 你好码头工人

那么 Docker 是什么？Docker 本身是一个平台，但是我们关心 Docker 容器。所以 Docker 容器是一个轻量级的执行环境，用于在“密封的”环境中运行代码。这意味着，它是一台假计算机，存在于另一台运行代码的计算机中，而不必共享配置。

这意味着我们可以像容器一样运行我们的后端、数据库、模型和前端，而不会有配置或依赖性的问题。因为每个服务都可以在独立的容器中运行。

要在你的电脑上安装 Docker，如果你使用的是 Linux 发行版，我建议你遵循这个教程。如果你用的是 Windows，我会推荐这个[教程](https://docs.docker.com/docker-for-windows/install-windows-home/)，对于 Mac，推荐这个[教程](https://docs.docker.com/docker-for-mac/install/)

# 如何为我们的服务制作 Docker 容器

要创建 Docker 容器，首先必须创建 Docker 映像，这是我们容器的配置。Docker 的伟大之处在于，你可以使用其他人或公司创建的图像来简化事情。

让我们从创建后端容器开始。在我们后端代码所在的文件夹中，我们必须创建一个名为`Dockerfile`的文件，这是图像的默认名称。该图像的代码如下:

```
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7COPY . .RUN pip install -r requirements.txtEXPOSE 8000CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

第一行是`base image`，我们容器所基于的图像。在这种情况下，我们使用官方的 FastAPI 映像。然后我们将文件从后端目录复制到容器目录，这是两个文件我们的`[main.py](<http://main.py>)`和`requirements.txt`。现在我们使用`pip`在我们的容器中安装来自我们需求的依赖项。第四行将 8000 端口暴露给网络系统，这将允许我们的容器与系统的其余部分进行通信。最后，我们在后台运行代码。

现在我们必须创建我们的容器，在终端中我们的`Dockerfile`所在的文件夹内，运行以下命令:

```
docker build -t tutorial-backend .
```

(`tutorial-backend`是容器的名称，你可以放任何你想要的东西)

要运行容器，您可以使用以下命令:

```
docker run tutorial-backend
```

让我们看看我们的前端容器，因为它是一个苗条的应用程序，它不同于我们的后端，但在看完代码后，你会看到 Docker 是多么容易。

```
FROM node:12-alpineWORKDIR /usr/src/appCOPY package*.json ./RUN npm installCOPY . .EXPOSE 5000ENV HOST=0.0.0.0RUN npm run buildCMD [ "npm", "start" ]
```

# 连通容器

我们不必担心在一台机器上配置我们所有的服务，但它们必须相互通信，这很好。就像编程中的几乎所有事情一样，有一个工具可以做到这一点，这个工具叫做`Docker compose`。如果你用的是 Linux 机器，你需要安装 Docker compose(你可以用这个[教程](https://www.digitalocean.com/community/tutorial_collections/how-to-install-docker-compose)，其他人都已经安装好了。

Docker compose 允许我们运行多个可以相互对话的容器。我们所要做的就是描述我们的容器，它们如何通信以及其他变量。首先必须创建一个名为`docker-compose.yml`的文件。下面是我们一起运行所有东西所需的所有代码:

```
version: "3"services:
  tensorflow-servings:
    image: tensorflow/serving:latest
    ports:
      - 8501:8501
    environment: 
      - MODEL_NAME=movie_model
    volumes: 
      - ./ai-model/model:/models/movie_model
    depends_on: [mongo] mongo:
    image: "mongo"
    container_name: "movieDB"
    environment: 
      - MONGO_INITDB_DATABASE=movieRecommenderDB
    volumes:
      - ./mongo-volume:/data/db
    ports: 
      - 27017:27017

  backend:
    build:
      context: backend/
      dockerfile: Dockerfile
    image: movie-backend
    ports:
      - 8000:8000
    depends_on: ["mongo"]
    environment: 
      - MONGOHOST=mongo
      - TF_SERVING_HOST=tensorflow-servings frontend:
    build:
      context: frontend/
      dockerfile: Dockerfile
    image: movie-frontend
    ports:
      - 5000:5000
    depends_on: ["backend"]
```

第一行定义了 Docker compose 的版本，然后我们看看服务密钥里面有什么。第一个服务是`tensorflow-serving`服务，我们首先给我们的服务起一个名字，在这里是`tensorflow-serving`。在给出服务使用的图像之后，我们使用官方的 Tensorflow 服务图像。然后，在定义环境变量之后，我们公开端口。`volume`键用于定义共享文件夹，在这种情况下，我们希望共享我们的训练模型使用 Tensorflow 服务的文件夹。最后，`depend`的值告诉 Docker 在运行这个特定的容器之前组合所有需要运行的容器。

最后，要运行所有东西，我们需要在`docker-compose.yml`所在的文件夹中使用下面的命令:

```
docker-compose up
```

现在，如果您打开网络浏览器并进入`[<http://localhost:5000>](<http://localhost:5000>)`，您将能够使用您的应用程序并享受观看电影的乐趣。