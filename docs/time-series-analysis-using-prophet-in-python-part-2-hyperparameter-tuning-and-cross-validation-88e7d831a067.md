# 使用 Python 中的 Prophet 进行时间序列分析—第 2 部分:超参数调整和交叉验证

> 原文：<https://medium.com/analytics-vidhya/time-series-analysis-using-prophet-in-python-part-2-hyperparameter-tuning-and-cross-validation-88e7d831a067?source=collection_archive---------3----------------------->

[](/@msdata/time-series-analysis-using-prophet-in-python-part-1-math-explained-5936509c175c) [## 使用 Python 中的 Prophet 进行时间序列分析—第 1 部分:数学解释

### Prophet 将时间序列建模为一个广义加性模型(GAM ),它结合了趋势函数、季节性函数…

medium.com](/@msdata/time-series-analysis-using-prophet-in-python-part-1-math-explained-5936509c175c) 

在上一篇文章中，我们解释了 Prophet 背后的所有数学原理。在这篇文章中，我将向您展示如何在实践中使用 Prophet 以及如何进行超参数调整。

# 资料组