# Python 中基于 K 均值聚类和主成分分析的图像压缩

> 原文：<https://medium.com/analytics-vidhya/image-compression-using-k-means-clustering-and-principal-component-analysis-in-python-893fd37e8e15?source=collection_archive---------10----------------------->

读者们，欢迎来到我的第一篇媒体博客！让我们尝试使用 *sklearn* 软件包实现并比较 K-Means 聚类算法和主成分分析(PCA)在图像压缩上的结果。对压缩图像的评估是基于内存大小的减少以及它能在多大程度上概括或解释原始图像的差异。图像压缩的目的是在保持与图像相似性的同时，尽可能减小存储空间