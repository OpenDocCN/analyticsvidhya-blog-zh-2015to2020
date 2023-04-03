# 正确检测和处理异常值

> 原文：<https://medium.com/analytics-vidhya/detect-and-handling-outliers-53723d8ec17a?source=collection_archive---------6----------------------->

## 使用 Python 上的 IQR 进行检测

![](img/e72d4ccd08d92c593683cab63c6de0d4.png)

杰西卡·鲁斯切洛在 [Unsplash](https://unsplash.com/s/photos/unique?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

O utlier 检测是分析和清理数据之一。在某些情况下，这些异常值会像在回归模型中一样干扰我们的模型。如果我们处理不好，它会使我们的模型不会表现得更好。所以，在这里我想告诉你如何检测异常值，并妥善处理。通过适当的处理，它使我们的模型将执行…