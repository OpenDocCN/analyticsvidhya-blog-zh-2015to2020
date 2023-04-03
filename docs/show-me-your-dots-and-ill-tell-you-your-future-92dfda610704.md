# 给我看你的点，我会告诉你你的未来……

> 原文：<https://medium.com/analytics-vidhya/show-me-your-dots-and-ill-tell-you-your-future-92dfda610704?source=collection_archive---------11----------------------->

这个故事是记录我第一个独立数据科学项目进展的系列报道的一部分。找到上一篇 [*这里*](/swlh/richard-panda-wrangler-a92a59c82dad) *和 Jupyter 笔记本* [*这里*](https://github.com/rrpelgrim/springboard_repo/tree/master/capstones/capstone-two/notebooks) *。*

我有一些初步结果要展示！很早的时候，我还在整理我的数据集，但是我已经有了一些地图；众所周知，哪里有地图，哪里就有机会……

## …圆点。很多点。

准确地说，是三十四万零四百六十九个点。蓝点是所有的 GHCN 气象站；橙色的是 UCDP 冲突事件。即使只是目测一下，很明显一些地理区域有非常密集的天气测量覆盖(美国、北欧、澳大利亚)，而在其他区域 GHCN 覆盖要稀疏得多。许多最密集的冲突地区几乎没有气象站覆盖，这并不完全令人惊讶。

![](img/c14b38488b964b48ca47d477ee2d7533.png)

## 争吵策略#1:空间连接！

为了缩小感兴趣的区域，我对两个数据集执行了内部空间连接。由于两个数据集对于每个观测值都有精确的纬度和经度点，直接的内部空间连接将仅显示冲突事件和共享精确位置的气象站。虽然这可能会导致占领气象站的游击战术的有趣发现，但(不幸的是)这不是这个项目的主要目的。

因此，我使用 geopandas 包在冲突事件观察点周围创建了一个 100 公里的缓冲区。这将输出一个面几何，当指定正确的“内部”内部连接时，该几何可用于在第二个数据集中查找位于这些面内的点:

![](img/30f3a42dc2aa3d306b817b2fcfb52946.png)

*技术方面-注意:*为了能够执行。缓冲区方法需要使用将地理数据框架的坐标参考系统(CRS)设置为 EPSG3857(单位为米而非度)。to_crs 方法。我的 Jupyter 笔记本内核在尝试执行该方法时不断死亡，我花了一个多星期(不是开玩笑)才弄清楚发生了什么。你可以在这里和这里追溯我的一些兔子洞步骤[。](https://stackoverflow.com/questions/64663553/kernel-dying-when-executing-to-crs-to-convert-geometry-of-geodataframe)

但是花在论坛和重新安装我的 conda 包和环境上的时间是非常值得的，宝贝！

![](img/a7ad74b169985033f2b644780984d83e.png)

## 争论策略#2:自定义提取功能

如前所述，GHCN 数据集太大，无法完全提取到我的电脑上。因此，我的第二个策略是编写一个函数，它接受冲突事件 ID，查找冲突事件指定半径内的所有气象站，提取。dly 文件将天气数据作为数据帧保存到我的笔记本中，最后将这些数据帧保存到本地。csv 文件。我已经在一个单独的笔记本(“01_Data_Collection”)中完成了这项工作，这样这个耗时的过程只需在我需要访问新的电台文件时运行。在我实际的争论笔记本中，我可以直接打开。本地存储的 csv 文件。

如需完整代码，请在[我的 Github](https://github.com/rrpelgrim/springboard_repo/tree/master/capstones/capstone-two/notebooks) 上查看该项目的 Jupyter 笔记本。