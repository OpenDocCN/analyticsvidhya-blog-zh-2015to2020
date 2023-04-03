# 将电影评论情感分析模型部署到 Windows EC2 实例

> 原文：<https://medium.com/analytics-vidhya/deploying-sentiment-analysis-model-to-ec2-instance-c3e8ad900e98?source=collection_archive---------21----------------------->

工作流程可以分为以下几个基本步骤:

1.  在 colab 上训练机器学习模型。
2.  将推理逻辑包装到 flask 应用程序中。
3.  在 AWS ec2 实例上托管 flask 应用程序并使用 web 服务。

# 准备模型

我们将用于分类的算法是 DistilBERT。DistilBERT 是一个小型，快速，廉价和轻型变压器模型，由…