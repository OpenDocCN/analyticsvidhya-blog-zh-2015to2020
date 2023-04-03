# 为什么不应该使用 ARIMA 来预测需求

> 原文：<https://medium.com/analytics-vidhya/why-you-should-not-use-arima-to-forecast-demand-196cc8b8df3d?source=collection_archive---------5----------------------->

您是否使用 ARIMA(X)来预测供应链需求？
我看**五个理由**你为什么不应该。

1.  💾ARIMA 需要长期的历史视野，尤其是季节性产品。使用三年的历史需求可能**而不是**就足够了。**生命周期短的产品。**生命周期短的产品不会从这么多数据中受益。在更高层次上预测需求可能会有所帮助。但是它会带来其他的挑战(协调，失去准确性)。
2.  💻运行 ARIMA 在一个广泛的…