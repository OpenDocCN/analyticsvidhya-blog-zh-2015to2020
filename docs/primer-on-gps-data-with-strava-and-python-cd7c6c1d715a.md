# Strava 和 Python 的 GPS 数据入门

> 原文：<https://medium.com/analytics-vidhya/primer-on-gps-data-with-strava-and-python-cd7c6c1d715a?source=collection_archive---------6----------------------->

![](img/eb97719ab900a985ae06c30c78b9564a.png)

马库斯·斯皮斯克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

毫不奇怪，作为一名数据科学书呆子和自行车爱好者，我非常沉迷于 Strava，这是一款面向运动员的社交网络应用程序，也是一款非常棒的数据工具。

Strava 应用程序允许您记录您的骑行，并保存有趣的数据，如您的速度、距离、时间、海拔、功率输出、消耗的能量、心率、天气状况等。

就我个人而言，我最喜欢的数据类型是我旅行地点的 GPS 地图，因为当我查看以前骑行的地图时，我可以回忆起沿途不同地点的景点和记忆。有些人甚至创造了[“斯特拉瓦艺术”。](https://www.strav.art/home)

![](img/4c7cd9de08170402dc5d43b54e35701f.png)

为了创建每次骑行的地图，Strava 必须聚合由电话或自行车电脑以 GPX 格式记录的 GPS(全球定位系统)数据。

GPX 代表 GPS 交换格式，这是一种数据结构，我把它比作一棵有树枝和树叶的树。

![](img/3caed029e9363681493797768e22ba7b.png)

就像一棵树有许多分支，每个分支有许多叶子一样，一个 GPX 文件通常包含许多“轨迹”,这些轨迹包含许多“片段”,这些片段包含许多“点”。点是我们将在本文中看到的信息的基本单位。

当 Strava 创建 GPX 文件时，每个点都包含您设备的纬度、经度、海拔和时间戳*。点数更新的频率取决于您的设备和蜂窝网络强度，但通常足以相当合理地代表您的位置。

在斯特拉瓦的网站上，你可以下载一个活动的 GPX 文件，所以我们要去看看我的一个朋友在得克萨斯州沃思堡的骑行。

![](img/92a40b4ee6cce5142927e31ce5b41851.png)

对于接下来的步骤，我假设您的机器上已经安装了 Python。

首先，我们将导入一个名为 gpxPy 的 Python 库来解析 XML 模式树。为了安装这个库，我在终端上运行了“pip install gpxpy”。[https://pypi.org/project/gpxpy/](https://pypi.org/project/gpxpy/)

打开并解析 GPX 文件后，我们得到一个包含轨道、段和点的 GPX 对象。我们现在只对使用点数据感兴趣，所以在对它进行索引后，我们将得到一个包含 GPS 更新的元组列表(本例中有 7273 个点)。

![](img/b4919eb005c15a85d5af90c373cec205.png)

接下来，我们简单地将元组列表放入 Pandas 数据帧，其中包含经度、纬度和海拔。最后，我们可以用 Matplotlib 绘制经度和纬度，以获得我们对路线的第一次视觉观察。

![](img/cdadca4218f7ff76bc99739429d43b11.png)![](img/e743cd6df5cdca0270cc11c46ac033e5.png)

真正有趣的地方是使用另一个名为 gmplot 的 Python 库在谷歌地图上绘制我们的路线。最简单的终端安装方式是“pip install gmplot”。[https://pypi.org/project/gmplot/](https://pypi.org/project/gmplot/)

使用 gmplot，您需要为地图显示设置中心坐标，实例化一个新的 GoogleMapPlotter 对象，使其适合您的数据，并将地图绘制到 HTML 文件中。

![](img/48133be738c10603aa1dd76385ff6faf.png)

瞧啊。我希望您喜欢这个关于可视化 GPS 数据的简短教程示例。

![](img/99803de31c2cd11dc03939c98c21f692.png)

*注意:我的朋友用 Garmin 自行车电脑而不是手机记录了这条路线。我注意到我自己的 GPX 数据和他的数据之间的两个不同之处是:1)时间戳不包括在他的 GPX 点的 Garmin 中，这可能是因为 2)他的 GPX 每秒钟更新一次(2:01:13 经过的时间为 7273 点(121.2167 分钟* 60 秒/ 7273 点=每个 GPS 点 1.00 秒))。这是有利的相比，GPX 更新从我的手机不是那么精确，因为他们依赖于我的手机设备之间的连接强度&蜂窝网络提供商。因此，评估你骑自行车的实际平均速度的一种方法是使用自行车电脑，而不是手机。*