# 从 Python 调用 R | rpy 2 的魔力

> 原文：<https://medium.com/analytics-vidhya/calling-r-from-python-magic-of-rpy2-d8cbbf991571?source=collection_archive---------0----------------------->

![](img/e101fc37dd064a3cf74a68797ab93075.png)

数据科学的两个大朋友是 **Python** 和 **R** 。虽然许多人对这两种方法都很满意，但在 Python 中使用 R 功能通常不会出现在我们的日常生活中。因为当我开始我的数据科学之路时，Python 是语言，而 R 是人们开始忘记的东西。像`<-, %>%, variable$attribute`这样的 R 语法已经足够让我说*不*并坚持使用 python。

长话短说，一些写得很好的库强迫我“动手”。看一看 R，它并不全是坏的，事实上时间序列分析和某些数据处理确实比 python 快。这工作得很好，直到我的项目中既有 R 代码又有 Python 代码。这太疯狂了，出于很多原因，你也可能以这种情况告终。于是我开始问这个问题，**如何从 Python 调用 R 脚本？**

如果你以前问过谷歌这个问题，首先出现的话题显然是令人惊叹的 python 库 [rpy2](https://pypi.org/project/rpy2/) 。现在，这不会是一个关于 rpy2 的详细教程，但是我将解释如何从 Python 中调用用 R 编写的函数。Rpy2 提供了很多功能来使用 python 本身的 R 库和函数，但是修复数据类型不一致可能是一个真正的麻烦，所以这是我能想到的最好的快速修复方法。如果我正在读这篇文章，我不会走这么远，所以恭喜你！让我们直接进入代码，看看我们在这里处理的是什么。

预处理。稀有

```
filter_country <- function(df, country){
  #' Preprocessing df to filter country
  #'
  #' This function returns a subset of the df
  #' if the value of the country column contains 
  #' the country we are passing
  #'
  #' [@param](http://twitter.com/param) df The dataframe containing the data 
  #' [@param](http://twitter.com/param) country The country we want to filter
  #
  df = subset(df, df$Country == country)
  return(df)
}
```

这是一个 R 脚本**预处理。R** 包含一个函数 **filter_country** ，这个函数基本上过滤你所经过的国家的数据帧。这纯粹是为了演示，你可以用 R 的高级函数库让这个函数尽可能的复杂。现在让我们看看调用 R 脚本的 python 脚本。

预处理程序. py

```
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri# Defining the R script and loading the instance in Python
r = robjects.r
r['source']('preprocess.R')# Loading the function we have defined in R.
filter_country_function_r = robjects.globalenv['filter_country']# Reading and processing data
df = pd.read_csv("Country-Sales.csv")#converting it into r object for passing into r function
df_r = pandas2ri.ri2py(df)
#Invoking the R function and getting the result
df_result_r = filter_country_function_r(df_r, 'USA')
#Converting it back to a pandas dataframe.
df_result = pandas2ri.py2ri(df_result_r)
```

一开始这可能有点让人不知所措，但我会一步一步解释。

1.  我们正在定义我们想要从中获取函数的源文件。为此，我们通过传入 R 脚本的路径在`robjects.r[‘source’]()`中指定 source。
2.  然后，我们从`robjects.globalenv`中加载我们想要的函数，方法是将键作为我们在 r 中定义的函数的名称传入。您可以在这个脚本中定义多个函数，并通过它们各自的名称引用它们。我们将检索到的函数存储在一个 python 变量中。
3.  现在，在我们将一个数据帧传递给这个函数之前，我们必须注意到一个**熊猫数据帧**和一个 **R 数据帧**是不同的。幸运的是 *pandas2ri* 提供了在数据帧类型之间来回转换的函数。
4.  将熊猫数据帧转换为 R 数据帧。我们对它进行转换，并将其传递给我们从 r 对象创建的 python 函数。
5.  最后的结果是一个 R 数据帧，所以我们需要把它转换回熊猫数据帧。我们使用`pandas2ri.ri2py(df_result_r)`将其转换回熊猫数据帧。

瞧，我们已经用 Python 完成了 R 函数的接口。因此，通过维护一个基本的 dataframe 输入输出操作，我们将能够轻松地在 Python 内部接口 R 函数！:)

也请注意，在一些论坛中，有人建议 pandas version==0.23.x 是最理想的，我亲自测试了它，发现了一些与其他更高版本集成的错误。