# 如何用代码建立对模型预测的信任

> 原文：<https://medium.com/analytics-vidhya/building-trust-in-models-prediction-with-code-c55c0d18ce5a?source=collection_archive---------25----------------------->

使用 LIME 解释机器学习模型预测的分步指南

![](img/862888b138addc50d435b641cd738c93.png)

DJ Johnson 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

这个博客是关于解释我们的模型预测。即使我们的模型达到了接近 100%的准确率，我们的脑海中总会有一个问题。我们应该相信它吗？

考虑一下在某个诊所的情况，如果电脑只显示…