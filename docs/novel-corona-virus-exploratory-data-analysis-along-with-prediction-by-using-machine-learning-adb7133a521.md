# 新型冠状病毒:探索性数据分析以及使用机器学习算法的预测。

> 原文：<https://medium.com/analytics-vidhya/novel-corona-virus-exploratory-data-analysis-along-with-prediction-by-using-machine-learning-adb7133a521?source=collection_archive---------5----------------------->

> 一种新的基于 SARS 冠状病毒的感染确实是致命的，它起源于中国武汉，并传播到世界其他地方。

**背景**

2019-nCoV 是一种β冠状病毒，与 MERS 和 SARs 一样，都起源于蝙蝠。

该数据集包含 2019 年新型冠状病毒的受影响病例数、死亡数和恢复数的日常信息。

数据从 2020 年 1 月 22 日开始。

**数据**

来自世卫组织——2019 年 12 月 31 日，世卫组织接到警报，中国湖北省武汉市出现几例肺炎病例。这种病毒与其他任何已知的病毒都不匹配。这引起了关注，因为当一种病毒是新的时，我们不知道它如何影响人。

因此，当更广泛的数据科学社区可以获得关于受影响人群的日常信息时，这些信息可以提供一些有趣的见解。

该数据具有关于 2019 年新型冠状病毒的受影响病例数、死亡数和恢复数的每日水平信息。

数据从 2020 年 1 月 22 日开始。

约翰·霍普斯金仪表盘为数据来源:[https://docs . Google . com/spreadsheets/d/1 yzv 9 w9 zrkwrgtar-yzmamaqmefw 4 wmlaxocejdxzats 6 w/html view？usp=sharing & sle=true#](https://docs.google.com/spreadsheets/d/1yZv9w9zRKwrGTaR-YzmAqMefw4wMlaXocejdxZaTs6w/htmlview?usp=sharing&sle=true#)

**读取数据**

![](img/d4bab41effb85560d8c9f72defb910bd.png)

最后更新、确认、死亡和恢复是重要的列

**可视化**

![](img/44de92071218b404191931dfa0cd3063.png)![](img/06ec6c5c6f19309507fec8e81befe2ed.png)![](img/e299d3c3a76a3d530f7391053b3851c0.png)![](img/0584f1faec3c0cfb0f627b59afeaeead.png)

中国的死亡人数似乎是 100%

![](img/2763060e5bf0be1b1e149a5fc7064e43.png)

有一些病毒病例的国家→病例最多的中国

![](img/cb8c10b3eccaee327ee8a0d260581c5b.png)

中国确诊病例

![](img/ae558b0c688c5ee50b89706ca40e35f2.png)

中国各州病死率

![](img/038af43a1910989242ce0453b66f3e0b.png)

计算并绘制局部异常因子以检查数据集中是否存在异常

![](img/c0251727ee27167032d7c471e39e9267.png)

LOF 分数

![](img/1e0679840461da0edacdc833483bb765.png)

k 均值聚类

![](img/6dae641924a172039835b6d210855f28.png)

簇状构造

**将各种模型应用于数据集**

![](img/50941800e472ba618d8a82834d33f9a8.png)

线性回归

![](img/3d9edd382949536ff6a6abffebe18513.png)

实际值和预测值

![](img/4b4f83f89674389e8c79bf81f3efe5da.png)

简单的可视化

![](img/3dddd15f2ee6caaf302c32742007e66d.png)

利用随机梯度下降优化器应用神经网络模型

![](img/1850ee79c8a50efd3a4274c383a1a875.png)

模型准确性似乎是 80%

![](img/e8703f87109d666d53361985dc5b2755.png)

使用“adam”优化器和“sigmoid”激活功能尝试另一个模型

![](img/1eea061922497445735a16266d3b2e49.png)

模型精度仍然是 80%,优化器没有变化。

**结论** :-利用更多数据集，即实时冠状病毒可以在受影响区域之前根据传播情况进行预测。

**参考资料:-**

1.  [https://www.cdc.gov/coronavirus/2019-ncov/summary.html](https://www.cdc.gov/coronavirus/2019-ncov/summary.html)
2.  优化者:-[https://keras.io/optimizers/](https://keras.io/optimizers/)
3.  [关于冠状病毒的信息](https://hub.jhu.edu/novel-coronavirus-information/):-【https://hub.jhu.edu/novel-coronavirus-information/ 