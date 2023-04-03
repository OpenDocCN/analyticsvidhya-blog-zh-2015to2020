# 机器学习模型中的数据预处理

> 原文：<https://medium.com/analytics-vidhya/data-preprocessing-in-machine-learning-model-3af34d0f3ceb?source=collection_archive---------9----------------------->

![](img/fb257e45c183b1e7c87d2fb9e6d7c043.png)

数据预处理

# 数据预处理

*数据预处理是将原始数据转换成可理解格式的数据。原始数据包含缺失数据、噪声数据和许多错误。利用数据预处理技术可以解决所有的问题。整个数据集被分成训练集和测试集。该训练集用于训练机器学习模型。*

![](img/5025690ffee9309db36ce4b94ee58401.png)

原始数据— —经过处理的数据

# 原始数据

*原始数据是未经加工的计算机数据。这些信息可能存储在文件中，也可能只是存储在计算机硬盘某处的数字和字符的集合。例如，输入数据库的信息通常被称为原始数据。*

# 为什么要进行数据预处理？

*原始数据包含大量缺失数据、噪音数据和错误，因此我们不能在机器学习模型中使用这种格式。预处理数据仅用于机器学习模型。*

更好的数据预处理提高了模型的精度。因此，在机器学习和深度学习模型中，数据预处理是最重要的。

# 数据预处理技术

1.获取数据集。
2。导入库。
3。正在导入数据集。
4。编码分类数据。
5。寻找丢失的数据。
6。将数据集分为训练集和测试集。
7。特征缩放。

## 1.获取数据集

*首先，我们从网站或任何其他地方获取数据集。数据集具有 csv 或 excel 格式。csv 表示逗号分隔值，excel 是正常的 Microsoft excel 格式。titanic 数据集用于这种数据预处理技术。*

[](https://www.kaggle.com/c/titanic/data) [## 泰坦尼克号:机器从灾难中学习

### 从这里开始！预测泰坦尼克号上的生存并熟悉 ML 基础知识

www.kaggle.com](https://www.kaggle.com/c/titanic/data) 

## 2.导入库

*导入数据预处理所需的所有包。pandas 包用于处理数据集。Numpy 用于在数据集上执行的数组操作。matplotlib 用于可视化数据。*

![](img/87676c4219a032946bf8196159335da3.png)

## 3.导入数据集

*使用熊猫将数据集导入到 python 文件中。*

![](img/30f7ffee6df21731526ba67275e0c90e.png)

## 4.编码分类数据

*数据集有一些字符串格式的卷。机器学习模型只允许数值。通过分类值技术将字符串转换成数值。*

*两种技术:
1 .标签编码器
2 .一个热编码器*

## 标签编码器

![](img/9a9bb395c8dcfb02cc10c02965114627.png)

标签编码

*SK learn 提供了一个非常有效的工具，用于将分类特征的级别编码成数值标签编码器使用 0 和 n_classes-1 之间的值对标签进行编码，其中 n 是不同标签的数量。*

![](img/e999f04bf58e612373c8725c5357a40c.png)

*“性别”栏不是数值。所以转换成数值。*

![](img/96105d8a8d1a6211a9b9b525b1a8fa67.png)![](img/66c852552f06062a762aa1a2a3e9c556.png)

## 一个热编码器

![](img/5a1a1ab2d0f281d53e5b62b2bd1d980c.png)

一个热编码器

## 5.查找丢失的数据

缺失数据通过各种方法填充，以下方法用于填充缺失值。
1。平均值/中位数
2。零方法
3。最常见的值

![](img/8e105674e3b57604b37a888ff624539f.png)

缺少值

*前 5 列中缺失值较多。现在，我们可以填充“年龄”和“费用”列中缺少的值。*

## 平均值/中值

在缺失值列中找到平均值，然后将平均值填入缺失值位置。

![](img/6af7695f555b1cccc85fd3880b37b085.png)

## 零点方法

*缺失值用零填充。*

![](img/ec0bf80b7dda526bf7207fd25d3b09d0.png)

## 最常见的值

*缺失值由该特定列中最常用的数据填充。*

![](img/b3ddb80db700535e30fb06c8b0c11a06.png)![](img/f219bf13cb6b06ef527cb8f40fd4ff8a.png)

## 6.将数据集分为训练集和测试集

*数据集分为训练集和测试集。重要特征仅选择用于该模型训练过程，因为通过选择数据集中最重要的特征来增加准确性。*

![](img/b4965fc812abc47c2cd37aba8af59944.png)

## 7.特征缩放

*特征缩放是一种将固定范围内的数据中存在的独立特征标准化的技术。在回归算法中使用特征缩放以获得更高的精度。*

![](img/15aecabc49413c7a7f08af49e55f59b1.png)![](img/5586ae4635db8acd00fdb0d357fb490b.png)![](img/3e2fae000202a505fac8b1e2b2574d06.png)

现在，数据已准备好构建机器学习模型。

如果你想要完整的代码，请访问我的 Github 页面

[](https://github.com/jackdaniel-softsolution/Machine-Learning-Algorithms/tree/master/Machine%20Learning%20Algorithms/03.%20data%20preprocessing) [## jack Daniel-软解决方案/机器学习-算法

### 此时您不能执行该操作。您已使用另一个标签页或窗口登录。您已在另一个选项卡中注销，或者…

github.com](https://github.com/jackdaniel-softsolution/Machine-Learning-Algorithms/tree/master/Machine%20Learning%20Algorithms/03.%20data%20preprocessing)