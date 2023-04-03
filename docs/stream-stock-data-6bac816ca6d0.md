# 流式股票数据

> 原文：<https://medium.com/analytics-vidhya/stream-stock-data-6bac816ca6d0?source=collection_archive---------9----------------------->

![](img/c70a5ed781b579466c65ec5f054a6dc5.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Austin Distel](https://unsplash.com/@austindistel?utm_source=medium&utm_medium=referral) 拍摄的照片

如果你想建立一些机器学习模型来预测股票价格，还有什么比通过流式传输实时股票数据更好的方法呢？使用历史数据通常是一个很好的开始方式，但对于试图建立最稳健模型的更“铁杆”的数据科学家/机器学习工程师来说，获得正确的数据是关键。在本教程中，我们将使用 Python 库 [robin_stocks](http://www.robin-stocks.com/en/latest/index.html) 来传输股票价格。因为很少得到股票价格就足以…