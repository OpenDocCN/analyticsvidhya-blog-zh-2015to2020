# 用 Python 分析公共数据

> 原文：<https://medium.com/analytics-vidhya/analyze-covid-19-cases-data-with-python-74d6c43f9e1c?source=collection_archive---------19----------------------->

![](img/732726bca0bceefb1e25acd32c741651.png)

本文讨论了如何使用 Python 和 Pandas 库来分析官方的新冠肺炎案例数据。您将看到如何从实际数据集收集见解，发现乍看之下可能不太明显的信息。特别是，文章中提供的例子说明了如何获得有关疾病在不同国家的传播速度的信息。

# 准备您的工作环境

为了跟进，您需要在 Python 环境中安装 Pandas 库。如果您还没有它，您可以使用 pip 命令安装它:

```
pip install pandas
```

然后，您需要挑选一个实际的数据集来使用。对于本文提供的例子，我需要一个数据集，其中包括按国家和日期分列的新冠肺炎确诊病例总数的信息。这样的数据集可以从[https://data . hum data . org/dataset/novel-coronavirus-2019-ncov-cases](https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases)下载为 CSV 文件:time _ series _ covid 19 _ confirmed _ global _ narrow . CSV

# 加载数据并为分析做准备

在将下载的 CSV 文件读入熊猫数据帧之前，我手动删除了不必要的第二行:

```
#adm1+name,#country+name,#geo+lat,#geo+lon,#date,#affected+infected+value+num
```

然后我把它读入一个熊猫数据帧:

```
>>> import pandas as pd
>>> df= pd.read_csv("/home/usr/dataset/time_series_covid19_confirmed_global_narrow.csv")
```

现在让我们仔细看看文件结构。最简单的方法是使用 dataframe 对象的 head 方法:

```
>>> df.head()
 Province/State Country/Region Lat Long Date Value
0 NaN Afghanistan 33.0 65.0 2020–04–01 237
1 NaN Afghanistan 33.0 65.0 2020–03–31 174
2 NaN Afghanistan 33.0 65.0 2020–03–30 170
3 NaN Afghanistan 33.0 65.0 2020–03–29 120
4 NaN Afghanistan 33.0 65.0 2020–03–28 110
```

由于我们不会执行复杂的分析来考虑受影响国家在地理上的相互距离，因此我们可以安全地从数据集中移除地理纬度和地理经度列。这可以通过以下方式完成:

```
>>> df.drop("Lat", axis=1, inplace=True)
>>> df.drop("Long", axis=1, inplace=True)
```

我们现在看到的应该是这样的:

```
>>> df.head()
 Province/State Country/Region Date Value
0 NaN Afghanistan 2020–04–01 237
1 NaN Afghanistan 2020–03–31 174
2 NaN Afghanistan 2020–03–30 170
3 NaN Afghanistan 2020–03–29 120
4 NaN Afghanistan 2020–03–28 110
```

在我们开始删除不必要的行之前，了解数据集中有多少行也是很有趣的:

```
>>> df.count
…[18176 rows x 4 columns]>
```

# 压缩数据集

浏览数据集中的行，您可能会注意到有些国家的信息是按地区详细列出的，例如中国。但你需要的是全国的综合数据。要完成此合并步骤，您可以对数据集应用 groupby 操作，如下所示:

```
>>> df = df.groupby(['Country/Region','Date']).sum().reset_index()
```

该操作应该减少数据集中的行数，并消除省/州列:

```
>>> df.count
...[12780 rows x 3 columns]
```

# 执行分析

假设你需要在初始阶段确定疾病在不同国家的传播速度。比方说，您想知道从报告至少 100 个病例的那一天起，该疾病达到 1500 个病例需要多少天。

首先，你需要筛选出那些没有受到太大影响、确诊病例人数没有达到大量的国家。这可以通过以下方式完成:

```
>>> df = df.groupby(['Country/Region'])
>>> df = df.filter(lambda x: x['Value'].mean() > 1000)
```

然后，您可以只检索满足指定条件的那些行:

```
>>> df = df.loc[(df['Value'] > 100) & (df['Value'] < 1500)]
```

完成这些操作后，行数应该会显著减少。

```
>>> df.count
… Country/Region Date Value
685 Austria 2020–03–08 104
686 Austria 2020–03–09 131
687 Austria 2020–03–10 182
688 Austria 2020–03–11 246
689 Austria 2020–03–12 302
… … … …
12261 United Kingdom 2020–03–11 459
12262 United Kingdom 2020–03–12 459
12263 United Kingdom 2020–03–13 802
12264 United Kingdom 2020–03–14 1144
12265 United Kingdom 2020–03–15 1145[118 rows x 3 columns]
```

此时，您可能想要查看整个数据集。这可以通过下面一行代码来完成:

```
>>> print(df.to_string())Country/Region Date Value
685 Austria 2020–03–08 104
686 Austria 2020–03–09 131
687 Austria 2020–03–10 182
688 Austria 2020–03–11 246
689 Austria 2020–03–12 302
690 Austria 2020–03–13 504
691 Austria 2020–03–14 655
692 Austria 2020–03–15 860
693 Austria 2020–03–16 1018
694 Austria 2020–03–17 1332
1180 Belgium 2020–03–06 109
1181 Belgium 2020–03–07 169…
```

剩下的工作就是计算每个国家的行数。

```
>>> df.groupby(['Country/Region']).size()
>>> print(df.to_string())Country/Region
Austria        10
Belgium        13
China          4
France         9
Germany        10
Iran           5
Italy          7
Korea, South   7
Netherlands    11
Spain          8
Switzerland    10
Turkey         4
US             9
United Kingdom 11
```

上述列表回答了这样一个问题，即在某个国家，从报告至少 100 例病例的那一天起，该疾病需要多少天才能达到大约 1500 例确诊病例。

免责声明:重要的是要意识到官方确诊病例的比率可能与实际情况有所偏差。问题是，上述分析中使用的数据集忽略了延迟。