# 超市数据集的 EDA 自动化

> 原文：<https://medium.com/analytics-vidhya/automation-of-eda-for-superstore-dataset-ee382fa26410?source=collection_archive---------5----------------------->

![](img/403a5014c8b88864cc63957be71ef87f.png)

威廉·艾文在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

探索性数据分析(EDA)是一种数据分析和数据探索的方法，它对我们正在处理的数据采用各种技术(主要是图形表示)。

EDA 帮助寻找

*   揭示变量之间的不同关系
*   提取重要变量
*   检测异常值
*   最大限度地深入了解数据集
*   测试基本假设
*   开发简约的模型
*   确定最佳因子设置

根据《福布斯》的一项调查，数据科学家花费 **80%** 的时间在**数据准备上。**

但是，如果我告诉你 python 可以在一些库的帮助下自动化 EDA 的过程呢？不会让你的工作很舒服吗？那么让我们开始学习自动化 EDA 吧。

因此，为了最大限度地减少时间，我们将使用一个开源的 python 模块，只需几行代码就可以自动化整个 EDA 过程。

此外，假设这不足以说服我们使用这个工具。在这种情况下，它还生成交互式 web 格式的报告，可以呈现给任何不了解编程语言的人。

一些流行的自动化 python 库是:-

*   压型
*   Sweetviz
*   Autoviz

数据集可以在这里找到 [**。**](https://github.com/kashish-Rastogi-2000/Medium-Automation-of-EDA-for-superstore-dataset)

**1。** **熊猫画像**

我们主要使用 df.describe()函数进行探索性的数据分析，但是对于严肃的 EDA，我们需要使用 pandas_profiling，用 df.profile_report()扩展 pandas 数据帧，以便进行快速数据分析。

该数据集包含 9994 行和 13 列，因为手动分析将耗费大量精力和时间。我们不必担心，因为我们可以使用 Pandas Profiling 库来处理大型数据集，因为它速度很快，只需几秒钟就可以创建。

使用熊猫烧香的一个优点是它在一开始就显示警告信息。

## 概要分析概述:

*   有多少排
*   检测数据帧中的列类型
*   查找具有重复行、缺失值、唯一值的数据集
*   显示不同技术的最高相关变量
*   它做了一个非常有帮助的描述性分析
*   显示最高的基数，它也做文本分析

# **安装熊猫档案**

使用 pip 或 Github 安装

```
#Installing using pip
pip install pandas-profiling[notebook]#Installing using Github
pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
```

我们可以使用以下代码生成 HTML 报告

```
profile = ProfileReport(df, title="Pandas Profiling Report")
```

**保存报告**

如果我们想生成一个 HTML 报告文件，将`ProfileReport`保存到一个对象并使用`to_file()`函数:

```
profile.to_file("your_report.html")
```

或者，我们也可以获取 json 文件中的数据:

```
#As a string
json_data = profile.to_json()

# As a file
profile.to_file("my_report.json")
```

让我们将其应用到超级商店数据集来创建报告

## 理解报告

*   **概述:**显示数据集的简单概述
*   **变量属性:**显示数据集中的所有变量及其属性，如均值、中值等。
*   **变量的相互作用:**不同的类别变量和数值变量之间的相互作用。
*   **变量的相关性:**生成的报告包含数据集所有属性的不同类型的相关性，如 Pearson、Spearman、Kendall、Phik、Cramer 相关性。
*   **缺少** **值**:除此之外，报告还显示哪些属性缺少值。

现在让我们看看几行代码的输出:

显示超级商店数据集的输出

# 2.Sweetviz

Sweetviz 是一个 python 库，主要致力于在高密度和易于理解的可视化的帮助下探索和分析数据。它不仅自动化了 EDA 过程，还用于比较数据集并从中得出推论。

**使用 pip 安装**

```
pip install sweetviz
```

我们可以使用以下代码生成 HTML 报告

```
# Importing 
import sweetviz as sv# Analyzing & Display the
store_report = sv.analyze(df)
store_report.show_html('store.html')
```

**理解报告**

在 Sweetviz 中，我们可以清楚地看到数据集及其属性的不同属性。

现在让我们看看几行代码的输出:

Sweetviz 输出

Sweetviz 还允许您比较两个不同的数据集或同一数据集中的数据，方法是将其转换为测试和训练数据集。

```
df1 = sv.compare(df[4997:], df[:4997])
df1.show_html('Compare.html')
```

让我们看看它是怎么做的。

# 3.Autoviz

Autoviz 是一个开源 python 库，主要致力于深度可视化数据关系。它是仅用几行代码就能实现的最具冲击力的功能和情节创意可视化。

使用 pip 安装

```
#Installing using pip
pip install autoviz
```

我们可以使用以下代码生成 HTML 报告

```
# Importing 
from autoviz.AutoViz_Class import AutoViz_Class# Analyzing & Display the report
AV = AutoViz_Class()
df = AV.AutoViz('SampleSuperstore.csv')
```

## **了解报告**

上述命令将创建一个包含以下属性的报告:

*   **所有连续变量的两两散点图**
*   **箱线图&距离图**
*   **所有连续变量的直方图(KDE 图)**
*   **所有连续变量的小提琴图**
*   **连续变量热图**

现在让我们看看几行代码的输出:

# 就是这样！

## **参考文献**

[](/datadriveninvestor/10-python-automatic-eda-libraries-which-makes-data-scientist-life-easier-825d0a928570) [## 10 个 Python 自动 EDA 库，让数据科学家的生活更加轻松

### 如果你愿意倾听，数据会说话——吉姆·贝吉森

medium.com](/datadriveninvestor/10-python-automatic-eda-libraries-which-makes-data-scientist-life-easier-825d0a928570) [](https://analyticsindiamag.com/tips-for-automating-eda-using-pandas-profiling-sweetviz-and-autoviz-in-python/) [## 使用 Python 中的 Pandas Profiling、Sweetviz 和 Autoviz 实现 EDA 自动化

### 下载我们的移动应用探索性数据分析(EDA)用于探索我们正在处理的数据的不同方面…

analyticsindiamag.com](https://analyticsindiamag.com/tips-for-automating-eda-using-pandas-profiling-sweetviz-and-autoviz-in-python/) 

# 我希望你喜欢这个内容，支持我的工作！