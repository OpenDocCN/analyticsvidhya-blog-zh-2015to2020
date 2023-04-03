# 滑动窗口价格预测

> 原文：<https://medium.com/analytics-vidhya/sliding-window-price-predictions-20fea38bae6d?source=collection_archive---------8----------------------->

*这里使用的代码可在其原来的* [*资源库*](https://github.com/manyshapes/Produce-Soup/blob/master/Web%20scraping%20fruit%20index.ipynb) *中找到。ipynb 格式。你可以下载它&在你自己的设备上用 Jupyter 笔记本摆弄它。*

# 介绍

今天我们将看到如何利用历史农产品价格来预测未来二十年的价格。这将在 **Python** 中使用简单的线性回归模型来完成。**美丽的汤 4** 帮助解析来自在线来源的观察结果。然后将从 **Pandas** 数据帧中访问&这些数据。培训将在滑动窗口上进行；这和模型拟合…