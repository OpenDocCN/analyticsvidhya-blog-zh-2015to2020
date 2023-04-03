# 意大利冠状病毒 ODE 模型及参数优化 python 预测

> 原文：<https://medium.com/analytics-vidhya/coronavirus-in-italy-ode-model-an-parameter-optimization-forecast-with-python-c1769cf7a511?source=collection_archive---------4----------------------->

本文的目的是介绍冠状病毒在意大利传播的现有数据，并用一个模型对其进行分析。然后，我们将使用 scipy toolkit 优化模型的参数，并绘制所获得的预测。我们还将从 statsmodels 预测 ARIMA 下周的趋势。

首先要做的是导入 python 中的常用包和数据源:

```
import numpy as np
import random
import pandas as pd
from scipy.integrate import odeint…
```