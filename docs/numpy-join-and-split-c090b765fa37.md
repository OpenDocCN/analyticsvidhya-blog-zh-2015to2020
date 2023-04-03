# 数字连接和分割

> 原文：<https://medium.com/analytics-vidhya/numpy-join-and-split-c090b765fa37?source=collection_archive---------6----------------------->

![](img/02f6bba9a7bdc11af1fd1c3397500b90.png)

# Numpy 连接:

连接意味着将两个或多个数组的内容放在一个数组中。

我们使用 concatenate()和 axis 来实现这一点；如果轴没有被传递；它被视为 0。

![](img/5c6b157d7b8683f00d7af233743466e9.png)

带轴；

![](img/5c6b157d7b8683f00d7af233743466e9.png)

## 使用堆栈连接数组:

堆叠用于沿新轴连接相同维度的数组。

堆栈有三种类型:

水平堆叠

垂直堆叠

高度堆叠

## 水平堆叠:

水平堆叠沿行进行。

![](img/1d9f31bc1b7c63a48229854683952bfc.png)

## 垂直堆叠:

沿着列进行垂直堆叠。

![](img/0274019cb8c137926f00c56cfb38c723.png)

## 高度堆叠:

高度堆叠用于沿高度堆叠。

![](img/8313cb9127041c471d76e2ab412afdf3.png)

# 数字拆分:

Split 函数与 join 操作相反。

Join 将多个数组合并成一个，而 split 将一个数组拆分成多个数组。

我们使用 array_stack()来拆分数组。

![](img/ea7b46cfb0bc13bf5fef386a9e417a28.png)

如果数组的元素比要求的少，它将从末尾进行相应的调整。

![](img/d1b1d95e2a80d4b67e123e8a4c757deb.png)

还有一种方法做拆分；这是使用类似于 array_split()的 split()完成的。

但是当元素比要求的少时，split()不会调整，它会抛出错误。

访问拆分的数组；

![](img/d8c067a4bd577d41f0994b746179170e.png)

类似地，我们可以对二维数组做同样的事情。有 hsplit()，vsplit()和 dsplit()。

![](img/c3763caa8300befb31a6f03ba2f802c3.png)

dsplit 仅适用于 3 维或更多维的数组。

就这样，我们来到了这篇文章的结尾。

快乐编码…😊😊😊