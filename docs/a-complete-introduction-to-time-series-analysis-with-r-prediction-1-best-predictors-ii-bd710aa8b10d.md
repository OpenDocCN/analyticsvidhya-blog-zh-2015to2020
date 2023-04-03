# 时间序列分析完全介绍(带 R)::预测 1 →最佳预测器 II

> 原文：<https://medium.com/analytics-vidhya/a-complete-introduction-to-time-series-analysis-with-r-prediction-1-best-predictors-ii-bd710aa8b10d?source=collection_archive---------16----------------------->

在[上一篇文章](/@hair.parra/a-complete-introduction-to-time-series-analysis-with-r-tests-for-stationarity-prediction-1-a78c1cf16676)中，我们看到了如何在给定函数 X_{n} **的情况下，获得 X_{n+h}的 [**最佳线性预测值(BLP)。本周，我们将看到，通过一系列操作，这个函数最合适的模型具有上面给出的形式。我们会遵循和之前一样的思路:最小化某个目标函数！**](/@hair.parra/a-complete-introduction-to-time-series-analysis-with-r-tests-for-stationarity-prediction-1-a78c1cf16676)**

## X_{n+h}的最佳线性预测值