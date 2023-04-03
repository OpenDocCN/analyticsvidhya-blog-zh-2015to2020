# 使用 PYTHON 熊猫面板

> 原文：<https://medium.com/analytics-vidhya/using-python-pandas-panel-9aef3e9259a5?source=collection_archive---------11----------------------->

![](img/7d45b41122219c0022ccbbb10d982601.png)

面板是一个 3D 数据容器。面板数据来源于计量经济学，是熊猫这个名字的部分原因。
有一些语义值来描述涉及面板数据的操作。

*   **条目:**0 轴对应每个条目中的一个数据帧。
*   **major_axis:** 轴 1 是每行数据帧。
*   **minor_axis:** 轴 2，每列为 DataFrame。

使用以下结构创建面板:

上述结构的参数:

*   **数据:**数据采用多种形式，如数组、序列、映射、系列、字典、常数和其他数据帧。
*   **项:**轴= 0
*   **主轴:**轴= 1
*   **次轴:**轴= 2
*   **数据类型:**各列的数据类型
*   **复制:**复制数据。默认值为 false。

# 创建一个熊猫面板

可以使用多条路径创建面板。其中两项如下:

*   使用 ndarrays 创建
*   用数据帧字典创建

**第一路**

```
<class 'pandas.core.panel.Panel'>
Dimensions: 2 (items) x 4 (major_axis) x 5 (minor_axis)
Items axis: 0 to 1
Major_axis axis: 0 to 3
Minor_axis axis: 0 to 4
```

**第二路**

```
<class 'pandas.core.panel.Panel'>
Dimensions: 2 (items) x 5 (major_axis) x 3 (minor_axis)
Items axis: Madde1 to Madde2
Major_axis axis: 0 to 4
Minor_axis axis: 0 to 2
```

# 创建一个有熊猫的空面板

使用面板生成器，可以创建一个空面板，如下所示。

# 从熊猫小组获得数据

以下值用于从面板中提取数据。

*   项目
*   主轴
*   短轴

## 从带有项目的面板中捕获数据

## 用 major_axis 从面板捕获数据

## 用 minor_axis 从面板中捕获数据