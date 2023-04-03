# 熊猫简介

> 原文：<https://medium.com/analytics-vidhya/introduction-to-pandas-826b203986b5?source=collection_archive---------17----------------------->

## Python 最流行的数据分析库的完整指南。

近年来，python 编程语言获得了很大的普及，并且由于它的特性，已经成为大多数开发人员的首选编码。Python 有非常多的包可用于几乎所有的任务。python 的众多库之一是 *Pandas* ，在 python 中广泛用于数据分析。在这个博客中，我们将会介绍熊猫的基本知识。

![](img/2834c7faa2a792e2f353547765680e7c.png)

*Pandas* 是一个开源 python 库，它提供了高性能、易于使用、灵活且富有表现力的数据结构，旨在轻松直观地处理结构化(表格、多维、潜在异构)和时间序列数据。它旨在成为用 python 进行实际的、真实世界的数据分析的基础高级构建块。 *Pandas* 构建于 *Numpy* 之上，这也是一个开源的 python 库。

Pandas 非常适合不同类型的数据，例如:

*   表格数据，如 SQL 表或 Excel 电子表格。
*   有序和无序时间序列数据。
*   同质或异质的任意矩阵数据。
*   任何其他形式的统计数据。

对于数据科学家来说，处理数据包括执行任务，如清理数据、建模数据、解释结果以及以合适的格式组织数据以供进一步可视化。熊猫是完成所有这些任务的完美工具。在开始实际操作之前，让我们首先将这个包安装到我们的系统中。

## 装置

在 python 代码中使用 pandas 之前，我们需要先安装它，因为它不会预装在 python 中，除非您使用 anaconda，在这种情况下它会预装。您可以通过在命令提示符下执行以下代码来使用 pip 安装 pandas。这篇博客中的所有代码都将适用于 windows。对于 Mac 或 Linux，你需要谷歌一下。我会尽可能提供链接。

```
pip install pandas
```

现在让我们通过运行一行代码来检查软件包是否安装成功。打开 python 代码编辑器，然后输入以下命令。

```
import pandas as pd
```

这段代码应该没有任何错误。如果出现任何错误，这意味着安装不成功。

Pandas 因为其强大的数据结构而在数据分析方面如此受欢迎。pandas 中有两种主要的数据结构:

*   *系列(一维)。*
*   *数据帧(二维)。*

这两种数据结构处理了工业中绝大多数的典型用例，如金融、统计、社会科学和许多工程领域。

## **系列**

Series 是带有轴标签的一维 numpy 数组，能够处理任何类型的数据(整数、字符串、浮点甚至 python 对象)。轴标签称为*索引。*简单来说，series 只不过是 excel 电子表格中的一列。

可以使用 series()构造函数创建 pandas 系列，该构造函数有许多参数，但下面描述了其中最重要的参数。

*   _data_:它采用各种形式的数据，如数组、列表、字典、常量等。
*   _index_:将用作索引的值。如果没有提供，默认情况下 np.arrange(n)将被指定为 index，其中 n 是数据中的总行数。
*   _dtype_:用于数据类型。如果没有，将推断数据。
*   _copy_:用于复制数据，默认为 false。

**创建系列**

```
#creating lists
data = np.array([15000, 250000, 500000, 1000000])#creating lists of index values
index_val = ['2011', '2012', '2013', '2014']#creating dataframe 
ser = pd.Series(data, index=index_val)
ser
```

## 数据帧

Dataframe 是一个二维大小可变的、潜在异构的表格数据结构，带有标记轴(行和列)。数据在 dataframe 中以表格格式排列。Dataframes 由三部分组成:行、列和数据。它也可以被认为是多个熊猫系列的组合，因为 dataframe 中的每一列都是一个系列。

可以使用 dataframe()构造函数创建 pandas 数据帧，该构造函数有许多参数，但最重要的参数将在下面描述。

*   _data_:它采用各种形式的数据，如数组、系列、地图、列表、字典、常数和其他数据帧等。
*   _index_:将用作行索引的值。如果没有提供，默认情况下 np.arrange(n)将被指定为 index，其中 n 是数据中的总行数。
*   _columns_:对于列标签，如果未提供，则默认采用 np.arrange(n)。
*   _dtype_:用于每列的数据类型。如果没有，将推断数据。
*   _copy_:用于复制数据，默认为 false。

**创建数据帧**

```
#creating lists of lists
data = [['1','Thor','Mjolnir'],['2', 'Cap America', 'Shield'],['3', 'Iron Man', 'Armour'],['4', 'Black Widow', 'Combat'], ['5', 'Hawk Eye', 'Arrow']]#creating lists of column names
columns = ['Avenger number', 'Name', 'Weapon']#creating dataframe 
df = pd.DataFrame(data, columns)
df
```

Pandas 提供了很多东西，像 series 和 dataframe 这样的数据结构太强大了，无法在一篇博客中解释它们的实现。这对于这个博客来说已经足够了。我将介绍系列和数据框架的实际实现，也许将来会使用 pandas 进行实际的数据分析，直到那时继续学习。

感谢您的阅读。享受 python。