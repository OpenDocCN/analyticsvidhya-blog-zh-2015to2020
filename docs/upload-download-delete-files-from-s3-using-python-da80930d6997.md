# 使用 Python 从 S3 上传|下载|删除文件

> 原文：<https://medium.com/analytics-vidhya/upload-download-delete-files-from-s3-using-python-da80930d6997?source=collection_archive---------4----------------------->

## 如何上传，下载和删除文件从 S3 使用 Python 使用 Boto3

在我们的大部分开发中，我们目前使用 amazon s3 作为数据存储。在我作为一名人工智能工程师的情况下，我使用亚马逊 s3 主要存储深度学习模型，这是一个很大的尺寸。在许多其他大数据使用案例中，我们也需要不断地从 S3 上传、下载和删除文件。这个故事将告诉你如何使用 Python 作为编程语言，借助一个名为 Boto3 的库来实现这一点。