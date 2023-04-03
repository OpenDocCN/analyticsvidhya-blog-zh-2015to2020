# 在 SQL 中随机化数据

> 原文：<https://medium.com/analytics-vidhya/randomizing-data-in-sql-66554ba2761?source=collection_archive---------8----------------------->

![](img/e57cc9c4a3d6d7cd659e99575f93c37a.png)

由[大卫·梅尼德雷](https://unsplash.com/@cazault?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/planet?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

## 数据工程

## 使用 NASA JPL 数据集在 [SQL](https://linktr.ee/kovid) 中生成随机测试/训练数据集

可以使用 equel 从给定的数据集中获取随机记录。人们可以很容易地使用`rand()`函数来生成 0 到 1 之间的随机浮点值。我最近写了关于[如何只用 SQL](https://towardsdatascience.com/generating-test-data-using-sql-2a1162f5ef16?source=your_stories_page---------------------------) 生成测试数据的文章。有了这个功能，很多…