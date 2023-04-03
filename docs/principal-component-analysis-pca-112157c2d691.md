# 主成分分析

> 原文：<https://medium.com/analytics-vidhya/principal-component-analysis-pca-112157c2d691?source=collection_archive---------10----------------------->

## 在 numpy 和 sklearn 中逐步实现概念的深入探讨。

***“在任何一种数据丰富的环境中寻找模式都很容易…关键在于确定模式代表的是噪声还是信号。”***

**——内特·西尔弗**

# 介绍

## **偏差-方差权衡**

学生在拟合模型时遇到的一个典型问题是平衡模型的偏差和方差，称为**偏差-方差** …