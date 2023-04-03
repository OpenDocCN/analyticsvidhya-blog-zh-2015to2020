# 用 python 实现 PCA、LDA 和 PLS 第 1 部分:主成分分析

> 原文：<https://medium.com/analytics-vidhya/pca-lda-and-pls-exposed-part-1-principal-component-analysis-c8f5b826caa6?source=collection_archive---------2----------------------->

在这篇文章中，我想考虑 PCA(主成分分析)、LDA(线性判别分析)和 PLS(偏最小二乘法)算法之间的主要区别，以及它们在分类/回归的典型问题中的应用。我将在 sklearn 中使用 python 及其实现的包。本教程的第二部分将重点介绍 LDA 和 PLS。

[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html?highlight=pca#sklearn.decomposition.PCA) 是一种降维技术，现在作为无监督学习广泛应用于机器学习中。它广泛应用于…领域