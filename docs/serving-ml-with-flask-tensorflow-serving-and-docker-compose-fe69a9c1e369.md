# 用烧瓶，TensorFlow 服务和 Docker 组成的服务毫升

> 原文：<https://medium.com/analytics-vidhya/serving-ml-with-flask-tensorflow-serving-and-docker-compose-fe69a9c1e369?source=collection_archive---------4----------------------->

*关于如何使用 Flask、Docker-Compose 和 Tensorflow 服务于生产中的深度学习模型的简短指南*

![](img/d166f40c7dc06328496d1366ff331f6f.png)

# **简介**

在[的第一部分](/@osas.usen/image-quality-classification-from-training-to-production-part-1-92439e4342d9)，我们使用迁移学习构建了一个神经网络分类器来预测给定图像的质量是“好”还是“差”。在本教程中，我们将把这个模型作为…