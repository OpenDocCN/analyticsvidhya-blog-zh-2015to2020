# 如何在 Google 云存储桶中写并获取一个 JSON 文件？

> 原文：<https://medium.com/analytics-vidhya/how-to-write-and-get-a-json-file-in-google-cloud-storage-when-deploying-flask-api-in-google-app-9121fa936d85?source=collection_archive---------0----------------------->

![](img/271aea701e2e3e61a2c1dc0c3022eaf3.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Daniel Páscoa](https://unsplash.com/@dpascoa?utm_source=medium&utm_medium=referral) 拍照

我们将创建两个函数，它们将上传并在云存储桶中创建一个 JSON 文件，然后您可以在 flask、fastAPI 等中使用这些函数。

```
# libraries used
google-cloud-storage==1.40.0
```

**功能 1 :** 在这个功能中我们将创建 json 对象并上传到云存储桶。

在 google 云存储桶中创建 json 对象

**函数 2 :** 在这个函数中，我们将从云存储桶中下载给定的 json 对象。

从 google 云存储桶下载 json 对象

我已经为你的理解注释了代码，如果你不能理解一些行，写下你对文章的回应，我会回来找你。