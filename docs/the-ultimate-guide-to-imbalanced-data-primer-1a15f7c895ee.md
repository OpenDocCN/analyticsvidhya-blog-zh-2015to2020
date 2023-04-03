# 不平衡数据的研究

> 原文：<https://medium.com/analytics-vidhya/the-ultimate-guide-to-imbalanced-data-primer-1a15f7c895ee?source=collection_archive---------5----------------------->

![](img/7fed674878f8d236d9c73e03deb80de7.png)

来源:https://engineering.fb.com/

# 介绍

不平衡数据意味着所有类别中至少有一个类别超过了其他类别(假设:标签 1 的比例= 98% &标签 2 的比例= 2%)。这在现实世界中很常见，比如众所周知的欺诈检测或 Kaggle 的 Porto Seguro 安全驾驶预测竞赛。

目前，解决不平衡数据的两种主要方法是:1)基于数据的方法和基于数据库的方法