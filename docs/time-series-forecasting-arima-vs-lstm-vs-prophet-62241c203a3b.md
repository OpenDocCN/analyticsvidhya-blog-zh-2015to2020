# 时间序列预测:ARIMA vs LSTM vs 预言家

> 原文：<https://medium.com/analytics-vidhya/time-series-forecasting-arima-vs-lstm-vs-prophet-62241c203a3b?source=collection_archive---------0----------------------->

## 基于机器学习和 Python 的时间序列预测

## 摘要

本文的目的是寻找预测的最佳算法，竞争者有 [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) 过程、 [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) 神经网络、[脸书预言家](https://facebook.github.io/prophet/)模型。我将带着注释遍历每一行代码，以便您可以轻松地复制这个示例(下面是完整代码的链接)。

我们将使用来自 Kaggle 竞赛"**预测未来销售**(链接如下)的数据集，您在其中…