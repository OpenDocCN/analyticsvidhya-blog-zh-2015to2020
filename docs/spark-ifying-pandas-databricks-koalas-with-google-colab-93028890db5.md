# 火花四射的熊猫:Databrick 的考拉与谷歌 Colab

> 原文：<https://medium.com/analytics-vidhya/spark-ifying-pandas-databricks-koalas-with-google-colab-93028890db5?source=collection_archive---------4----------------------->

![](img/2c1c5a6234f78ace47698958fcf9b8e4.png)

大数据是数据科学家需要更频繁处理的新规范。Apache Spark 是广泛使用的大数据框架之一。对于数据科学家来说，也许 Pandas 是最受欢迎的 python 库，数据科学家工作的每一分钟都在使用它。虽然 pandas 在大多数情况下足以执行典型的数据科学家分析，但据观察，它在处理大数据时性能会下降。

Modin 是一个声称能让熊猫更快的框架，但是它并没有解决大数据的处理。Databrick 的考拉是在 spark 基础设施上运行时使用熊猫式架构的另一种选择。

> 在 Google Colab 上运行考拉

在这篇文章中，你可以学习如何开始使用谷歌 colab 的考拉。