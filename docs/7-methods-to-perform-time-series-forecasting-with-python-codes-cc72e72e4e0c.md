# 执行时间序列预测的 7 种方法(使用 Python 代码)

> 原文：<https://medium.com/analytics-vidhya/7-methods-to-perform-time-series-forecasting-with-python-codes-cc72e72e4e0c?source=collection_archive---------1----------------------->

我们大多数人都听说过市场上的新热点，即加密货币。我们中的许多人也会投资他们的硬币。但是，将资金投资于如此波动的货币安全吗？我们如何确保现在投资这些硬币在未来一定会产生健康的利润？我们不能确定，但我们可以根据之前的价格得出一个大概的值。时间序列模型是预测它们的一种方法。

![](img/a85b6078a59e6a0a67128d58c5953f63.png)

*来源:比特币*

除了加密货币之外，时间序列预测还有许多重要的应用领域，例如:销售预测、呼叫中心的呼叫量、太阳活动、海洋潮汐、股票市场行为等等。

假设一家酒店的经理想要预测他预计明年会有多少游客，从而相应地调整酒店的库存，并对酒店的收入做出合理的猜测。根据前几年/月/日的数据，他(她)可以使用时间序列预测，并获得一个访客的近似值。预测的游客价值将有助于酒店管理资源和相应的计划。

在本文中，我们将学习多种预测技术，并通过在[数据集](https://datahack.analyticsvidhya.com/contest/practice-problem-time-series-2/)上实现来比较它们。我们将通过不同的技术，看看如何使用这些方法来提高分数。

我们开始吧！

# 目录

*   理解问题陈述和数据集
*   安装库(statsmodels)
*   方法 1——从简单的方法开始
*   方法 2 —简单平均
*   方法 3 —移动平均
*   方法 4 —单一指数平滑
*   方法 5 —霍尔特线性趋势法
*   方法 6 —霍尔特冬季季节法
*   方法 7 — ARIMA

# 理解问题陈述和数据集

我们遇到了一个时间序列问题，涉及到独角兽投资者的新高速铁路服务 JetRail 的通勤人数预测。我们获得了 2 年的数据(2012 年 8 月至 2014 年 9 月),利用这些数据，我们必须预测未来 7 个月的通勤人数。

让我们开始处理从上面的链接下载的数据集。在本文中，我只处理训练数据集。

```
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
#Importing data
df = pd.read_csv('train.csv')
#Printing head
df.head()
```

![](img/95710ee994300a2e0326e79c9ec3a934.png)

```
#Printing tail
df.tail()
```

![](img/955b3afb80676e72191b412843a380d3.png)

从上面的打印报表中可以看出，我们得到了 2 年(2012-2014)每小时*的数据，以及通勤人数，我们需要估计未来的通勤人数。*

*在本文中，我每天都在子集化和聚合数据集，以解释不同的方法。*

*   *数据集子集(2012 年 8 月-2013 年 12 月)*
*   *为建模创建训练和测试文件。前 14 个月(2012 年 8 月—2013 年 10 月)用作训练数据，后 2 个月(2013 年 11 月—2013 年 12 月)用作测试数据。*
*   *每天汇总数据集*

```
*#Subsetting the dataset
#Index 11856 marks the end of year 2013
df = pd.read_csv('train.csv', nrows = 11856)

#Creating train and test set 
#Index 10392 marks the end of October 2013 
train=df[0:10392] 
test=df[10392:]

#Aggregating the dataset at daily level
df.Timestamp = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 
df.index = df.Timestamp 
df = df.resample('D').mean()
train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
train.index = train.Timestamp 
train = train.resample('D').mean() 
test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
test.index = test.Timestamp 
test = test.resample('D').mean()*
```

*让我们将数据可视化(一起训练和测试)，以了解它在一段时间内是如何变化的。*

```
*#Plotting data
train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
test.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
plt.show()*
```

*![](img/f0d420d233e6f1427762353e05856ed1.png)*

# *安装库(statsmodels)*

*我用来进行时间序列预测的库是 statsmodels。您需要在应用一些给定的方法之前安装它。statsmodels 可能已经安装在您的 python 环境中，但它不支持预测方法。我们将从他们的存储库中克隆它，并使用源代码进行安装。请遵循以下步骤:-*

1.  *使用 pip freeze 检查它是否已经安装在您的环境中。*
2.  *如果已经存在，使用“conda 移除 statsmodels”将其移除*
3.  *使用“git clone git://github . com/statsmodels/stats models . git”克隆 stats models 存储库。在克隆之前，使用“git init”初始化 Git。*
4.  *使用“cd statsmodels”将目录更改为 statsmodels*
5.  *使用“python setup.py build”构建安装文件*
6.  *使用“python setup.py install”安装它*
7.  *退出 bash/终端*
8.  *在您的环境中重启 bash/terminal，打开 python 并执行“from stats models . TSA . API import exponential smoothing”进行验证。*

# *方法 1:从简单的方法开始*

*考虑下面给出的图表。假设 y 轴表示硬币的价格，x 轴表示时间(天数)。*

*![](img/bc6fb87da7f3379e9a926ff1c1d70605.png)*

*我们可以从图表中推断出硬币的价格从一开始就是稳定的。很多时候，我们会得到一个数据集，这个数据集在整个时间段内都是稳定的。如果我们想预测第二天的价格，我们可以简单地用最后一天的价格来估计第二天的价格。这种假设下一个期望点等于上一个观察点的预测技术被称为**朴素方法。***

*![](img/295cc0e6068d8f370700fb4571926723.png)*

*现在我们将实现 Naive 方法来预测测试数据的价格。*

```
*dd= np.asarray(train.Count)
y_hat = test.copy()
y_hat['naive'] = dd[len(dd)-1]
plt.figure(figsize=(12,8))
plt.plot(train.index, train['Count'], label='Train')
plt.plot(test.index,test['Count'], label='Test')
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()*
```

*![](img/a8d602898e4db1bbe9a337e8e5b6273d.png)*

*我们现在将计算 RMSE，以检查我们的模型在测试数据集上的准确性。*

```
*from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(test.Count, y_hat.naive))
print(rms)

RMSE = 43.9164061439*
```

*我们可以从 RMSE 值和上图中推断出，朴素方法不适合可变性很高的数据集。它最适合稳定的数据集。我们仍然可以通过采用不同的技术来提高分数。现在我们将看看另一种技术，并试图提高我们的分数。*

# *方法 2: —简单平均*

*考虑下面给出的图表。假设 y 轴表示硬币的价格，x 轴表示时间(天数)。*

*![](img/2c9c44bc32a3411c76df4e3b54a0921b.png)*

*我们可以从图表中推断出，硬币的价格以很小的幅度随机上升和下降，因此平均值保持不变。很多时候，我们会得到一个数据集，尽管在其整个时间段内会有小幅度的变化，但每个时间段的平均值保持不变。在这种情况下，我们可以预测第二天的价格与过去几天的平均价格相近。*

*这种预测期望值等于所有先前观察点的平均值的预测技术称为简单平均技术。*

*![](img/6398354ec13bf7192fdca26e4703ebbd.png)*

*我们取所有先前已知的值，计算平均值，并将其作为下一个值。当然不会很精确，但也差不多。作为一种预测方法，实际上在某些情况下这种技术效果最好。*

```
*y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['Count'].mean()
plt.figure(figsize=(12,8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
plt.legend(loc='best')
plt.show()*
```

*![](img/d93e58428fd90003392d2a81d019eb11.png)*

*我们现在将计算 RMSE，以检查我们的模型的准确性。*

```
*rms = sqrt(mean_squared_error(test.Count, y_hat_avg.avg_forecast))
print(rms)

RMSE = 109.545990803*
```

*我们可以看到这个模型并没有提高我们的分数。因此，我们可以从分数中推断出，当每个时间段的平均值保持不变时，这种方法效果最好。虽然朴素方法的得分优于平均方法，但这并不意味着朴素方法在所有数据集上都优于平均方法。我们应该一步一步地走向每个模型，并确认它是否改进了我们的模型。*

# *方法 3 —移动平均*

*考虑下面给出的图表。假设 y 轴表示硬币的价格，x 轴表示时间(天数)。*

*![](img/89b1bd395103691e3eae66c6ac7001ee.png)*

*我们可以从图表中推断出，一段时间以前硬币的价格大幅上涨，但现在价格稳定了。很多时候，我们被提供一个数据集，其中对象的价格/销售额在一段时间之前急剧上升/下降。为了使用以前的平均方法，我们必须使用所有以前数据的平均值，但使用所有以前的数据听起来不对。*

*使用初始期间的价格会极大地影响下一期间的预测。因此，作为对简单平均值的改进，我们将只取最近几个时间段的价格平均值。显然，这里的想法是，只有最近的价值是重要的。这种使用时间段窗口计算平均值的预测技术称为移动平均技术。移动平均的计算包括有时被称为大小为 n 的“滑动窗口”*

*使用简单的移动平均模型，我们根据固定的有限数量“p”个先前值的平均值来预测时间序列中的下一个值。因此，对于所有的 i > p*

*![](img/e875403b998c1b198a9df34bb18812a5.png)*

*均线实际上是非常有效的，特别是如果你选择了正确的 p 值。*

```
*y_hat_avg = test.copy()
y_hat_avg['moving_avg_forecast'] = train['Count'].rolling(60).mean().iloc[-1]
plt.figure(figsize=(16,8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()*
```

*![](img/42dbe4c56a397d66d9e970e4690321ac.png)*

*我们只选择了最近 2 个月的数据。我们现在将计算 RMSE，以检查我们的模型的准确性。*

```
*rms = sqrt(mean_squared_error(test.Count, y_hat_avg.moving_avg_forecast))
print(rms)RMSE = 46.7284072511*
```

*我们可以看到，对于这个数据集，朴素方法优于平均方法和移动平均方法。现在我们来看看简单的指数平滑法，看看它的表现如何。*

*移动平均法的一个进步是**加权移动平均**法。在上面看到的移动平均法中，我们同样对过去的“n”个观察值进行加权。但是我们可能会遇到这样的情况，即过去的每一次观测都会以不同的方式影响预测。这种对过去的观测值进行不同加权的技术称为加权移动平均技术。*

*加权移动平均是一种移动平均，其中滑动窗口内的值被赋予不同的权重，通常是为了使最近的点更重要**。***

***![](img/2d42c239c0479f9b9c422dc80b43fb26.png)***

***它不需要选择窗口大小，而是需要一个权重列表(加起来应该是 1)。例如，如果我们选择[0.40，0.25，0.20，0.15]作为权重，我们将分别为最后 4 个点赋予 40%，25%，20%和 15%的权重。***

# ***方法 4 —简单指数平滑***

***在我们理解了上述方法之后，我们可以注意到，简单平均线和加权移动平均线都位于完全相反的两端。我们需要介于这两种极端方法之间的方法，在对数据点进行不同加权的同时考虑所有数据。例如，可能明智的做法是，对最近的观测结果给予比遥远过去的观测结果更大的权重。基于这一原理的技术被称为简单指数平滑。使用加权平均值计算预测，其中权重随着观测值来自更远的过去而呈指数下降，最小的权重与最早的观测值相关联:***

***![](img/4d71a485416e127f0136c4a5709220f0.png)***

***其中 0≤ α ≤1 为**平滑**参数。***

***时间 T+1 的一步预测是系列 y1，…，yT 中所有观测值的加权平均值。权重降低的速率由参数α控制。***

***如果你盯着它足够长的时间，你会看到期望值 ŷx 是两个乘积之和:α⋅yt 和(1−α)⋅ŷ t-1。***

***因此，它也可以写成:***

***![](img/ac672c1db7c089518cc321c6b6706615.png)***

***本质上，我们得到了一个有两个权重的加权移动平均线:α和 1α。***

***正如我们所见，1α乘以之前的期望值 ŷx1，这使得表达式是递归的。这就是为什么这种方法被称为**指数**。时间 t+1 的预测值等于最近的观测值 yt 和最近的预测值 ŷt | t1 之间的加权平均值。***

```
***from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['Count'])).fit(smoothing_level=0.6,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()***
```

***![](img/c59c7f8689189fd7990df317c4bf6fab.png)***

***我们现在将计算 RMSE，以检查我们的模型的准确性。***

```
***rms = sqrt(mean_squared_error(test.Count, y_hat_avg.SES))
print(rms)

RMSE = 43.3576252252***
```

***我们可以看到，实现 alpha 为 0.6 的简单指数模型产生了迄今为止更好的模型。我们可以使用验证集来调整参数，以生成更好的简单指数模型。***

# ***方法 5 —霍尔特线性趋势法***

***我们现在已经学习了几种预测方法，但我们可以看到，这些模型对变化较大的数据效果不佳。考虑到比特币的价格正在上涨。***

***![](img/c1d6b5e449eec38338092000bd0e46dd.png)***

***如果我们使用上述任何一种方法，都不会考虑到这种趋势。趋势是我们在一段时间内观察到的价格的一般模式。在这种情况下，我们可以看到有增加的趋势。***

***尽管这些方法中的每一种都可以应用于趋势。例如，简单的方法会假设最后两点之间的趋势保持不变，或者我们可以对所有点之间的所有斜率进行平均以获得平均趋势，使用移动趋势平均值或应用指数平滑。***

***但是，我们需要一种方法，能够在没有任何假设的情况下准确地绘制趋势图。这种考虑数据集趋势的方法称为霍尔特线性趋势法。***

***每个时间序列数据集可以分解成趋势、季节性和残差三个分量。任何遵循趋势的数据集都可以使用霍尔特的线性趋势方法进行预测。***

```
***import statsmodels.api as sm
sm.tsa.seasonal_decompose(train.Count).plot()
result = sm.tsa.stattools.adfuller(train.Count)
plt.show()***
```

***![](img/da5d2d15cbe1b574a2353d0a967788f0.png)***

***我们可以从获得的图表中看到，该数据集呈增长趋势。因此，我们可以使用霍尔特的线性趋势来预测未来的价格。***

***Holt 扩展了简单的指数平滑法，允许用趋势预测数据。它只不过是应用于水平(系列中的平均值)和趋势的指数平滑。为了用数学符号来表达这一点，我们现在需要三个等式:一个用于水平，一个用于趋势，一个用于结合水平和趋势来获得预期的预测 ŷ***

***![](img/121faced869e4f8d5d99ff5478fcfbb2.png)***

***我们在上述算法中预测的值称为 Level。在上述三个等式中，您可以注意到我们添加了级别和趋势来生成预测等式。***

***与简单的指数平滑一样，此处的水平方程显示它是观测值的加权平均值，样本内一步预测。趋势方程显示它是基于ℓ(t)−ℓ(t−1 和 b(t1)的时间 t 的估计趋势的加权平均值，b(t1)是趋势的先前估计值。***

***我们将添加这些方程以生成预测方程。我们也可以通过将趋势和水平相乘而不是相加来生成乘法预测方程。当趋势线性增加或减少时，使用加法方程，而当趋势指数增加或减少时，使用乘法方程。实践表明，乘法是一种更稳定的预测方法，而加法更容易理解。***

***![](img/95986a7ed8e0d679ed0b48f87abcb9f8.png)***

```
***y_hat_avg = test.copy()

fit1 = Holt(np.asarray(train['Count'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))

plt.figure(figsize=(16,8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()***
```

***![](img/57ca08e049e821eca40818b126431bf0.png)***

***我们现在将计算 RMSE，以检查我们的模型的准确性。***

```
***rms = sqrt(mean_squared_error(test.Count, y_hat_avg.Holt_linear))
print(rms)RMSE = 43.0562596115***
```

***我们可以看到，这种方法准确地映射了趋势，因此与上述模型相比，提供了更好的解决方案。我们仍然可以调整参数以得到更好的模型。***

# ***方法 6 —霍尔特-温特斯法***

***所以让我们引入一个新的术语，它将在这个算法中使用。考虑一个位于山上的旅馆。它在夏季有很高的访问量，而在一年中的其他时间游客相对较少。因此，业主在夏季获得的利润将远远好于任何其他季节。这种模式每年都会重复。这种重复被称为季节性。在一段时间的固定间隔后显示一组相似模式的数据集受到季节性的影响。***

***![](img/4d9cdf6aa4ef275fa8496d2930919914.png)***

***上述模型在预测时没有考虑数据集的季节性。因此，我们需要一种既考虑趋势又考虑季节性的方法来预测未来价格。我们可以在这种情况下使用的一种算法是 Holt 的 Winter 方法。三重指数平滑(霍尔特的冬天)背后的想法是，除了水平和趋势之外，还将指数平滑应用于季节成分。***

***由于季节性因素，使用霍尔特的冬季方法将是其余模型中的最佳选择。霍尔特-温特斯季节性方法包括预测方程和三个平滑方程——一个用于ℓt 水平，一个用于趋势 bt，一个用于用 st 表示的季节性成分，平滑参数为α、β和γ。***

***![](img/0ae16fa6a38cab5a53f53707e85f7213.png)***

***其中 s 是季节循环的长度，0 ≤ α ≤ 1，0 ≤ β ≤ 1，0 ≤ γ ≤ 1。***

***水平方程显示了时间 t 的季节性调整观测值和非季节性预测值之间的加权平均值。趋势方程与霍尔特的线性方法相同。季节等式显示了当前季节指数和去年同一季节(即 s 个时间段之前)的季节指数之间的加权平均值。***

***在这种方法中，我们也可以实现加法和乘法技术。当季节变化在整个系列中大致不变时，最好使用加法方法，而当季节变化与系列水平成比例变化时，最好使用乘法方法。***

```
***y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot( train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()***
```

***![](img/d41a8045439b168e7c156f1645fd15a0.png)***

***我们现在将计算 RMSE，以检查我们的模型的准确性。***

```
***rms = sqrt(mean_squared_error(test.Count, y_hat_avg.Holt_Winter))
print(rms)

RMSE = 23.9614925662***
```

***从图表中我们可以看到，绘制正确的趋势和季节性提供了一个更好的解决方案。我们选择季节性周期= 7，因为数据每周重复一次。其他参数可以根据数据集进行调整。在建立这个模型时，我使用了默认参数。您可以调整参数以获得更好的模型。***

# ***方法 7 — ARIMA***

***另一个在数据科学家中非常流行的常见时间序列模型是 ARIMA。代表**自回归综合移动平均**。指数平滑模型基于对数据趋势和季节性的描述，而 ARIMA 模型旨在描述数据之间的相关性。ARIMA 的一个改进是季节性 ARIMA。它考虑了数据集的季节性，就像霍尔特的温特方法一样。您可以从这些文章[(1)](https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/)[(2)](https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/)中了解更多关于 ARIMA 和季节性 ARIMA 模特及其预处理的信息。***

```
***y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)
plt.figure(figsize=(16,8))
plt.plot( train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()***
```

***![](img/0544f601ac2913b9af9ed0e44d0d66c9.png)***

***我们现在将计算 RMSE，以检查我们的模型的准确性。***

```
***rms = sqrt(mean_squared_error(test.Count, y_hat_avg.SARIMA))
print(rms)

RMSE = 26.035582877***
```

***我们可以看到，使用季节性 ARIMA 产生了类似的解决方案，如霍尔特的冬天。我们根据 ACF 和 PACF 图选择参数。你可以从上面提供的链接中了解更多。如果您在寻找 ARIMA 模型的参数时遇到任何困难，您可以使用用 R 语言实现的 **auto.arima** 。Python 中 auto.arima 的一个替代可以在这里[查看](https://github.com/tgsmith61591/pyramid)。***

***我们可以根据这些模型的 RMSE 分数来比较它们。***

***![](img/8bb5185c07c85ec3b38a3de075bc6a92.png)***

# ***结束注释***

***我希望这篇文章对你有所帮助，现在你可以轻松地解决类似的时间序列问题了。我建议你采取不同种类的问题陈述，用上面提到的技巧慢慢解决。尝试这些模型，并找出哪种模型最适合哪种时间序列数据。***

***从这些步骤中学到的一个教训是，这些模型中的每一个都可以在特定的数据集上胜过其他模型。因此，这并不意味着在一种类型的数据集上表现最好的一个模型在所有其他数据集上也会表现相同。***

***您还可以探索用 R 语言为时间序列建模构建的**预测**包。你也可以从预测包中探索双季节性模型。在这个数据集上使用双季节性模型将产生更好的模型，从而得到更好的分数。***

***你觉得这篇文章有帮助吗？请在下面的评论区分享你的观点/想法。***

****原载于 2018 年 2 月 8 日*[*【www.analyticsvidhya.com*](https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/)*。****