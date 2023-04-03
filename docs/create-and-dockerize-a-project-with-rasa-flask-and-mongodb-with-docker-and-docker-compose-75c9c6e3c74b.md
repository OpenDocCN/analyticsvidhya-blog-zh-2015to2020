# 用 RASA，Flask 和 Mongodb 创建一个项目，并用 Docker 和 Docker-compose 编写

> 原文：<https://medium.com/analytics-vidhya/create-and-dockerize-a-project-with-rasa-flask-and-mongodb-with-docker-and-docker-compose-75c9c6e3c74b?source=collection_archive---------3----------------------->

我在寻找建立意图和实体识别模型的教程。经过一些搜索，我能够找到 rasa，rasa_core，rasa_nlu 等 docker 容器。因为我只是想使用我自己的架构，我是用虚拟环境开发的，所以我想在我的 web_app 容器中添加 RASA nlp 引擎。这篇博客将让你了解如何为由 Docker 部署的 Flask 支持的意图和实体部署 NLP 引擎。