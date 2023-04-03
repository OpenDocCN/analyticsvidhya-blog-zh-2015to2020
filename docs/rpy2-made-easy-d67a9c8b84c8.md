# Rpy2 变得简单

> 原文：<https://medium.com/analytics-vidhya/rpy2-made-easy-d67a9c8b84c8?source=collection_archive---------3----------------------->

## 初学者使用 Rpy2 的综合指南

> 两名数据科学家走进一家酒吧。他们都想使用 Python 中的 R。瞧吧— [Rpy2](https://rpy2.github.io/doc/latest/html/index.html) 。

![](img/51fd2be0fdff103e9a72dcfd157b6208.png)

Rpy2(嵌入 python 中的 R)使用起来相当吓人，这完全是因为它不像它的各个部分那样有血有肉。还因为单独使用 R 和 python 要容易得多，但是某些软件架构可能需要两者的结合。Python 直观的数据结构、可视化库和优秀的 IDE 与 R 的可信包相结合，为数据科学家的开发提供了坚实的资源。

# 如何设置

1.  如果你还没有的话，下载 Python3+和 R。
2.  从[这里](https://pypi.org/project/rpy2/)安装 Rpy2

# 使用 Python 中的 R 包和函数

下面的例子展示了如何使用适当的包从 python 调用现有的 R 函数。将要进口的货物如下:

```
#Import necessary packages
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
#Must be activated
pandas2ri.activate()
```

下面的代码片段展示了如何将 R 对象和包导入 python。示例中使用了 R 中的时序对象和预测包。人们需要确保这些功能/包在 R 中独立工作。

```
time_series=robjects.r('ts')
forecast_package=importr('forecast')
```

使用 python 中预测包中函数的时间:

```
#converting the training data into an R time series object
r_times_series_data=time_series(df["Actuals"].values,frequency=12)#fit the Time series into a model
#Using auto_arima
fit_arima=forecast.auto_arima(r_times_series_data,seasonal=False)#getting the forecast value
forecasted_arima=forecast_package.forecast(fit_arima,h=10,level=(95.0))
```

变量“forecasted _ arima”是一个列表向量，一个 R 数据类型。用户可以打印列名来查找所需的列。以下几行可用于查找预测结果:

```
#The 3rd index has the forecast value for Arima.
#NOTE: THE INDEX MAY NOT BE SAME FOR ALL MODELS
arima_output=forecasted_arima[3]
```

或者，用户可以按如下方式使用列名来访问列:

```
#Alternate way to find forecasted result
arima_output=forecasted_arima.rx2("mean")
```

# 从 Python 调用现有的 R 函数

假设有一个 R 代码，如下所示:

```
#Saved in the file Forecast_r_function.r
library(forecast)
Forecast_r_function= function(actuals, freq){
    y <- ts(actuals,frequency = freq)
    fit <- auto.arima(y, seasonal=TRUE)
    forecasted = forecast(fit, h=5)
    return (forecasted)
    }
```

从 python 调用上述预测函数，需要进行以下导入:

```
#Import the [SignatureTranslatedAnonymousPackage](https://rpy2.github.io/doc/v3.0.x/html/robjects_rpackages.html#rpy2.robjects.packages.SignatureTranslatedAnonymousPackage) 
from rpy2.robjects.packages import STAP#Read the file with the R code snippetwith open('Forecast_r_function.r', 'r') as f:
    string = f.read()#Parse using STAP
forecast_func_in_python= STAP(string, "Forecast_r_function")
```

上面的步骤使 python 中的 R 函数可用。可以通过以下方式访问它:

```
#Calling R function
forecasted_arima=forecast_func_in_python.Forecast_r_function(time_series, 10)
#storing result
arima_output=forecasted_arima.rx2("mean")
```

# 结论

在 python 框架中使用 Rpy2 还有许多其他方式，包括用 python 本身编写 R 代码的方式。然而，这些比这里阐述的更容易出错。

数据科学家经常使用 R 和 Python，如果使用得好的话，将一个嵌入另一个的接口确实非常强大。尽管它不是很受欢迎，但 rpy2 可以在设计和集成整体系统时为数据科学家和开发人员提供技术优势。

*请通读 rpy2 文档* [*此处*](https://rpy2.github.io/doc/v3.0.x/html/high-level.html) *。*