# 用 Plotly 分析美国出口

> 原文：<https://medium.com/analytics-vidhya/analyzing-u-s-exports-with-plotly-aa09fcaa3a8f?source=collection_archive---------15----------------------->

## 使用可视化工具理解数据

在我之前的[文章](https://towardsdatascience.com/data-visualization-with-plotly-71af5dd220b5)中，我已经介绍了一些在 [Plotly](https://plot.ly/) 中可用的有用的图形工具，这是一个开源库，可以在 Python 和 r 中使用

在这里，我将更多地使用 Plotly 的功能，使用 2011 年美国出口的一些数据作为输入。因此，让我们导入并研究我们的数据:

```
import pandas as pd df= pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv') 
```