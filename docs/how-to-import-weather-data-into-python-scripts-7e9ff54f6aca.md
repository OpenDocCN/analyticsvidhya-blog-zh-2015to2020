# 如何将天气数据导入 Python

> 原文：<https://medium.com/analytics-vidhya/how-to-import-weather-data-into-python-scripts-7e9ff54f6aca?source=collection_archive---------1----------------------->

![](img/841cde185625745e9f122dee27f45c32.png)

在本文中，我们将使用基于云的天气 API 将天气数据导入到一个 [Python](https://www.python.org/) 应用程序中。这些技术可用于扩展许多需要包含天气预报数据或历史天气观测数据的 Python 脚本和应用程序。

# 为什么包括天气数据？

天气数据成为 Python 应用程序的一个重要特性的原因有很多。

我们看到的天气预报数据最常见的应用之一是帮助规划事件或业务流程。例如，在零售业中，天气预报数据可用于计划受天气影响的产品的库存，如温暖或寒冷天气服装或在恶劣天气情况下与天气紧急情况相关的物品。

保险公司将使用天气预报数据来为即将到来的天气事件制定计划。建筑公司可以通过了解天气预报来计划温度和天气关键活动，如混凝土浇筑。

历史天气观测数据有助于企业和个人了解过去特定日子里发生的事情。这可用于关联业务指标，以回答诸如“下雨会影响我的销售吗”之类的问题。用 Python 对天气数据进行统计分析是非常有价值的。

历史天气摘要描述了一年中某个时间某个地点通常经历的总体天气状况。除了“正常”天气条件，这些总结也描述了极端天气。例如，您可以了解一月份某个特定地点的正常最高和最低温度。你也可以很容易地确定最高和最低温度。

# 为什么要用 Python？

Python 使得以编程方式使用天气数据变得很容易，因此您可以快速、重复、跨多个位置和时间段执行上述分析。

# 创建一个简单的 python 脚本来导入天气数据

在 Python 中，从天气 API 导入数据非常简单。在这个例子中，我们将使用围绕他们的[天气数据](https://www.visualcrossing.com/weather-data)产品构建的[可视交叉天气 API](https://www.visualcrossing.com/weather-api) 。其他天气 API 已经存在，请选择最符合您需求的一个。

**第一步——为你选择的天气 API 注册一个账户**

大多数天气 API 都需要一个帐户。如果你跟随我们的样本，你可以在这里创建一个账户[。](https://www.visualcrossing.com/weather-api)

**步骤 2 —构建一个天气数据查询。**

我们使用的 API 使用 restful APIs 来返回逗号分隔值(CSV)或基于 JSON 的结构。为了简单起见，我们将使用 CSV 格式。CSV 是一种非常简单的结构，使用逗号作为值分隔符来描述数据表。

API 有许多针对不同天气类型的端点。

对于天气预报，一个示例 URL 是:

```
[https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/](http://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/)forecast?location=Herndon,VA&aggregateHours=24&unitGroup=us&key=APIKEY
```

对于历史天气数据，简单的 API 结构是:

```
[https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history?goal=history&aggregateHours=24&startDateTime=2019-12-04T00:00:00&endDateTime=2019-12-11T00:00:00&contentType=csv&unitGroup=us&locations=Herndon,VA](https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history?goal=history&aggregateHours=24&startDateTime=2019-12-04T00%3A00%3A00&endDateTime=2019-12-11T00%3A00%3A00&collectStationContributions=false&maxStations=-1&maxDistance=-1&includeNormals=false&shortColumnNames=false&sendAsDatasource=false&allowAsynch=false&contentType=csv&unitGroup=us&key=X2KW264AEYIX7DF30FB18RTBL&locations=Herndon%2CVA)&key=APIKEY
```

这两个函数都返回 CSV，我们可以在 python 脚本中直接导入和处理它。

您可以将它们直接提交到浏览器窗口中来查看输出(记得用您自己的值替换‘API key’参数！)

**步骤 3——编写一个 python 脚本**

我们将使用几个库来帮助下载和处理 Python 中的天气数据

```
import csv //to process the CSV data
import codecs //to download and decode the information
import urllib.request //to request data from a URL
import sys
```

脚本的下一部分基于查询类型和位置请求构造天气 API 查询。天气 API 支持更多可以添加的参数，但我们会保持简单。

首先我们构造定义 URL 的根:

```
BaseURL = ‘http://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/'
```

然后我们添加公共参数，包括位置和 API 键。注意，我们使用 Python 脚本的输入参数来允许用户指定位置和键。这些值包含在 sys.argv 数组中。

```
DateParam = sys.argv[2].upper() # Set up the location parameter for our query
QueryLocation = ‘&location=’ + urllib.parse.quote(sys.argv[1]) # Set up the key parameter for our query
QueryKey = ‘&key=’ + sys.argv[3]
```

查询的下一部分特定于类型—天气预报和历史天气数据。对于天气预报，我们只需要预报的时间段。aggregationHours=24 相当于一天—我们可以通过将该值更改为 1 来检索每小时的预测。

```
# Set up the specific parameters based on the type of queryif DateParam == ‘FORECAST’: 
 print(‘ — Fetching forecast data’) 
 QueryTypeParams = ‘forecast
        &aggregateHours=24&unitGroup=us&shortColumnNames=false’
else: 
 print(‘ — Fetching history for date: ‘, DateParam)  # History requests require a date. We use the same date for start
   and end since we only want to query a single date in this example
 QueryDate = ‘&startDateTime=’ + DateParam +
        ‘T00:00:00&endDateTime=’  + sys.argv[2] + ‘T00:00:00’  QueryTypeParams = ‘history
        &aggregateHours=24&unitGroup=us&dayStartTime=0:0:00
        &dayEndTime=0:0:00’ + QueryDate
```

现在我们从所有部分构建完整的查询:

```
# Build the entire query
URL = BaseURL + QueryTypeParams + QueryLocation + QueryKey
```

我们现在准备提交查询。这可以通过我们在开始时导入的三行库来实现:

```
# Parse the results as CSV
CSVBytes = urllib.request.urlopen(URL)
CSVText = csv.reader(codecs.iterdecode(CSVBytes, ‘utf-8’))
```

这会请求数据，然后将结果解析为 CSV。CSVText 实例帮助我们轻松提取数据。

最后，我们可以使用数据。在我们的简单示例中，我们将输出显示为一个简单的表输出:

```
for Row in CSVText:
  if RowIndex == 0:
    FirstRow = Row
  else:
    print(‘Weather in ‘, Row[0], ‘ on ‘, Row[1])
    ColIndex = 0
    for Col in Row:
      if ColIndex >= 4:
        print(‘ ‘, FirstRow[ColIndex], ‘ = ‘, Row[ColIndex])
      ColIndex += 1
    RowIndex += 1
```

你可以在这里找到《T2》的完整代码。

# 后续步骤

上面的简单代码示例显示了从天气 API 中检索天气并将其集成到 Python 应用程序中是多么容易。从这一点来看，将天气预报集成到规划应用程序中或者使用历史数据进行统计分析是很简单的。

## 有问题吗？请在下面告诉我！