# 图的拉普拉斯和傅立叶基的特征向量和特征值

> 原文：<https://medium.com/analytics-vidhya/eigen-vectors-and-eigen-values-of-graph-laplacian-and-fourier-basis-b1517cc3250?source=collection_archive---------11----------------------->

TL；DR —给定一个图及其相关的拉普拉斯算子(在图卷积的情况下)，主特征值直观地给出图的结构，如连通分量和特征向量，捕捉图中与零交叉相关的各种频谱。

先决条件—[https://medium . com/analytics-vid hya/Graph-Convolution-intuition-9416 c0f 51167](/analytics-vidhya/graph-convolution-intuition-9416c0f51167)关于图形卷积

在这篇文章中，我们将把图形卷积作为傅立叶域中的“乘法”问题来处理(正如在…