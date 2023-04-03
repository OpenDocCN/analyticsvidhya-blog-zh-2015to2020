# 表格:层次结构

> 原文：<https://medium.com/analytics-vidhya/tableau-hierarchies-ee9cf482b328?source=collection_archive---------23----------------------->

## 通过创建层次结构来组织数据和简化可视化

![](img/df36a9102dc618844a84983545195ea5.png)

由[Edvard Alexander lvaag](https://unsplash.com/@edvardr?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

Tableau 中通常使用层次结构来组织数据和简化数据可视化过程。

一个典型的例子是当我们处理地理数据时，我们需要创建一个层次结构来定义国家、地区和省份之间的关系。

我们正在使用的数据库由意大利的地区、省份及其各自的人口组成。

创建层次结构的一种方法是**将一个元素拖放到另一个元素中。在我们的例子中，`Provinces`是`Region`的子类。因此，我们点击`Provinces`和**并拖拽**到`Region`。**

![](img/4cbc1485fb27014c25200d97bc059345.png)

选择层次结构的名称后，我们刚刚创建了…

![](img/e76279bcd94142c2ab4e69bb2c195f34.png)

…我们可以在**数据窗格(1)** 和**行字段(2)** 中看到它。

![](img/ebb154682b3ac85f3d68bdb11b620f27.png)

通过点击**行字段**中的(+)或(-)，我们可以在不同层级之间移动。

在我们使用更高级别的 **(+)** 的情况下:

![](img/e35768c73ea65cdf5bbef4e7f031d912.png)

如果我们需要更多的粒度 **(-)** :

![](img/ba793301492bccb0911a00c0d2f2ea12.png)

我们可以用来创建层次结构的另一种方法是使用 **CTRL +单击**突出显示我们想要包含在层次结构中的维度，并选择**层次结构>创建层次结构…**

![](img/531c0187e76cf82cbc1bb99efe0dd517.png)

在构建仪表板时，我一直在使用**层次结构**和**组**批次！如果您想了解更多关于分组的信息，请点击此处。

我希望你喜欢它！！