# D-Tale 图书馆简介

> 原文：<https://medium.com/analytics-vidhya/introduction-to-d-tale-library-fabb369c5d91?source=collection_archive---------18----------------------->

D-Tale 是可视化熊猫数据结构的 python 库。D-Tale 是基于 Flask 和 React 的交互式图形用户界面工具。D-Tale 是可视化和分析 pandas 数据结构的最简单的方法之一。

D-tale 库的安装:

Pip 安装数据表

#康达环境

康达安装有限公司

正在导入 dtale 库。使用 seaborn 库加载数据集。在这里，我使用虹膜数据集。

![](img/acaa880d8a626882e9470522517e67e4.png)![](img/706588db0dd18fe1e1fecef1d61a1909.png)![](img/8d473225178e27c0bdb1500f0f353e87.png)

dtale.show(df)将在 d-Tale 窗口中显示数据帧。

![](img/8e88d3aee743e15811bb3687f317dccc.png)

左上角箭头弹出如图菜单。这个菜单有多个功能，如数据集的总结，找出异常值和找出重复。

![](img/ebbe1c808283cd0196e95da0e5e36fe0.png)

该图显示了具有特征“sepal_length”的数据集描述。左侧是特征列表，右侧是特征值直方图。

![](img/471ef10f2cd6526b24431eb90d8b9f58.png)

**关联:**热图用于显示数据集的关联矩阵。

![](img/fdaecfcc441ca4e34af7968f370a10c2.png)

**重复:**在数据集中查找重复的实例。

![](img/e1c14cf0ff2e18f1a70309b328b51272.png)

**单个特征分析:**点击如图所示的特征名称，对单个特征进行操作。

![](img/1ecbf94d9bede70e9eb069d5d110a724.png)

**导出修改后的数据集:**以 CSV 或 TSV 格式导出修改后的数据集。

![](img/0411bdc3fc2a033923b9a97b13a346ba.png)

总之，我们已经看到了如何使用 D-tale 库以图形交互方式对 pandas 数据帧进行 EDA 分析。