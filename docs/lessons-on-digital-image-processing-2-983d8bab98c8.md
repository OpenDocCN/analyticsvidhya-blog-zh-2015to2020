# Python 中的数字图像处理简介

> 原文：<https://medium.com/analytics-vidhya/lessons-on-digital-image-processing-2-983d8bab98c8?source=collection_archive---------4----------------------->

![](img/234756a9d6a6a7a0a1e51a179a84a0ea.png)

图片提供: [torange.biz](https://torange.biz/fx/blue-computer-internet-technology-background-173390)

如果你还没有阅读这个系列的第一篇文章，请在这里阅读:[数字图像处理课程(#1)](/@yedhukrishnan/lessons-on-digital-image-processing-1-b7a1fa3acfe9)

在第二部分中，让我们编写一些实际的代码。我们将使用 Python 编写所有代码。

Python 使得处理和操作图像变得非常容易。Python 可用的 NumPy 和 SciPy 包帮助我们执行科学计算，主要是对矩阵的运算，这是我们感兴趣的。我们已经看到图像只是电脑中的矩阵。

在 Linux 机器上安装 SciPy 和 NumPy 最简单的方法是使用 Python 包管理器`pip`。

```
sudo apt-get install python3-pip  
sudo pip3 install numpy scipy
```

让我们再安装两个包，这是在 Python 中加载和显示图像所必需的。

```
sudo pip3 install imageio matplotlib
```

现在，要加载和显示图像，您只需要几行代码。确保在代码所在的同一目录中有一个图像文件。

仅此而已。代码是不言自明的。我添加了必要的注释来帮助您理解。

我们有一幅彩色图像。我们如何从它创建一个灰度图像？

我们已经知道彩色图像在每个像素位置有三个值，分别代表红色、绿色和蓝色强度值。在灰度图像中，只有一个值。我们需要做的就是将这三个值转换成一个值。

我们如何做到这一点？

取所有三个值的平均值就行了！

```
grey pixel value = (red + green + blue) / 3
```

让我们用 Python 来实现它。我们使用 NumPy 数组在 Python 中存储图像。NumPy 提供了一个方便的函数来取任意轴的平均值。在 RGB 图像的情况下，轴= 0 是行方向，轴= 1 是列方向，轴= 2 是通道方向。我们需要一个通道式总和。

```
gray_image = image.mean(axis = 2)
```

这一行代码将彩色图像转换为灰度图像。它的作用与下面几行代码相同:

上面的代码比之前的版本慢很多。然而，如果你是新手，理解这一点是很重要的。

我们刚刚创建了图像的灰色版本。现在，我们可以把它转换成二进制图像。一幅灰色图像的像素值范围从 0 到 255。在二进制中，它只能接受两个值。在某些表示中，我们使用 0 和 1。这里，我们将使用 0(黑色)和 255(白色)。

我们如何将灰度图像转换成二值图像？通常的方法是设置一个阈值`T`。如果像素值高于阈值，我们将把它设置为 255。如果它在下面，我们将把它设置为 0。这里我们可以把阈值取为灰度中值，也就是 128。

下面是代码的扩展版本。

简短的版本只有两行:

```
binary_image = grey_image.copy()
binary_image[binary_image <= 128] = 0
binary_image[binary_image > 128]  = 255
```

就是这样！NumPy 负责将所有像素值更新为 0 或 255，这取决于它们是高于还是低于阈值。

现在我们知道了彩色、灰度和二进制图像，我们知道如何将彩色图像转换成灰度图像，然后再转换成二进制图像。我们将在接下来的文章中看到更多的图像处理技术和 python 代码。

现在，请分享您对这些帖子的反馈和意见。