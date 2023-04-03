# Python 中的“大熊猫”简介

> 原文：<https://medium.com/analytics-vidhya/an-introduction-to-giant-pandas-in-python-e015f01bbe66?source=collection_archive---------16----------------------->

![](img/06dd4d2e4ff12f49e09ca3291d142655.png)

萨法尔·萨法罗夫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

如 [上所述【https://pandas.pydata.org/](https://pandas.pydata.org/)*pandas*是一个 BSD 许可的库，为 [Python](https://www.python.org/) 编程语言提供高性能、易于使用的数据结构和数据分析工具。It 有助于关注业务问题，而不是编程。它简单易学，易于使用，易于维护。

Pandas 数据结构:它有两个主要的数据存储结构。
1。系列
2。数据帧

**Series:**Series 非常类似于 NumPy 数组(构建在 NumPy 数组对象之上)，能够保存任何类型的数据(整数、字符串、浮点、python 对象等)。).轴标签统称为*索引*。

让我们从几个例子开始

![](img/6f2900e3d90a204de039eb09c69ca3c9.png)

进口熊猫和熊猫

*使用列表创建系列*

![](img/22ba194691b1be1074f93aadb6fa17cb.png)

序列使用列表

*使用 numpy 数组创建序列*

![](img/2886b8cfaf89899a840a5fc892e9982f.png)

使用 numpy 数组的系列

*使用字典创建系列*

![](img/961585c62bc24712cdeabace6f7f4094.png)

使用字典的系列

数据帧是熊猫的工作母机，直接受到 R 编程语言的启发。DataFrame 可以被认为是 NumPy 数组的一般化，或者是 Pyhton 字典的专门化。多个系列对象的集合也称为 DataFrame。我们会试着用几个例子来理解它。

让我们从导入数据开始

![](img/53a224299c3a65ba3826c8fb0aec622d.png)

正在导入。csv 数据转换成名为 sal_df 的数据帧

**数据的属性**

获取数据帧的索引

![](img/a52e04069e8d57aa5d12c60ccff53d13.png)

数据框架.索引

获取数据帧的列标签

![](img/1dd8324ccd9aadfa19821fff74526c50.png)

数据框架.列

从数据帧中获取元素总数

![](img/9ddadf7dcc05e2410632274835678823.png)

DataFrame.size

为了得到数据帧的维数

![](img/5c6896e0cc4d32cbae392f13f03189d5.png)

数据框架.形状

# 索引和数据选择

函数 *head* 返回数据帧的前 n 行。默认情况下， *head()* 返回前 5 行

![](img/9714809208d7689d23acfc19800a6d24.png)

DataFrame.head()

函数 *tail* 根据位置返回对象的最后 n 行。默认情况下， *tail()* 返回前 5 行

![](img/53487aa55082b0294e3fb8d512e2a3d6.png)

DataFrame.tail()

函数 sample 从 dataframe 中返回任意*随机* n 行，默认情况下， *sample()* 返回 1 行。

![](img/38a96803cd3911788964a0b041a8263c.png)

DataFrame.sample()

要访问标量值，最快的方法是使用 at 和 iat 方法

![](img/8908e1717515e8338d8bd27b4f232734.png)

DataFrame.at[]

![](img/480a2c369b469d57050699af96979c58.png)

DataFrame.iat[]

通过标签访问一组行和列。可以使用 loc[ ]

![](img/9144d2cd907005addbd0f470ce939a56.png)

DataFrame.loc[]

按列名选择

![](img/3756dea363822df4e5ec7e2df5f03887.png)

DataFrame['列名']

按列名列表选择

![](img/e1d62bf07fcca21b84434d9096a61972.png)

DataFrame[['列名 1 '，'列名 2']]

我个人最喜欢的是用点(。)批注但不被专家建议

![](img/f656194502ed0995f4d21694a8ca3d59.png)

DataFrame.column 名称

# 数据帧的简明摘要

info()返回数据帧的简明摘要

![](img/16f28964de2ba80cb74345e9d4d75d5f.png)

DataFrame.info()

我们还可以使用 dtypes 获取每一列的数据类型，dtypes 返回一系列数据类型。

![](img/e1f827e1d811ff4f8ccb8e2aa230922d.png)

DataFrame.dtypes

## 从列中获取唯一值

1.  使用 *unique()，*在建系列中

![](img/6178a644839594ea1d1f98f74462479f.png)

DataFrame['列名']。唯一()

2.使用 numpy *unique()*

![](img/860e41b600b31ae82a738881f4786137.png)

numpy . unique(data frame[' column _ name '])

如果我们检查输出的*类型*，两者都是 ndarray，唯一的区别是 *np.unique()* 排序，而 *unique()* 未排序。

![](img/bd8b42a7c598eda4985706f740e53227.png)

类型()

计算每列中唯一值的数量

![](img/be70b3079c4cf92c4f70a0a182eaeead.png)

DataFrame['列名']。努尼克

## 数据的条件选择

![](img/a3fb0f13a99a868e80c42886559345d1.png)

数据的条件选择

对于两种情况，我们可以使用带括号的|和&:

![](img/efe67b4aa0d508813a19494e0aa2352d.png)

多重条件

## 检测缺失值

使用 Dataframe.isnull.sum() 检查每列中缺失值的计数

![](img/b4509c8d1854ab558de57544c380b632.png)

DataFrame.isnull()。总和()

> 这是对 Python 中熊猫库的介绍，我会在下一个故事中介绍更高级的内容。感谢阅读！