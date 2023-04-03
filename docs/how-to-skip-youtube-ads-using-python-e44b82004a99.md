# 如何用 python 跳过 Youtube 广告？

> 原文：<https://medium.com/analytics-vidhya/how-to-skip-youtube-ads-using-python-e44b82004a99?source=collection_archive---------3----------------------->

![](img/af6b915dbb0055a2298ba5882c4d3865.png)

照片由 [YTCount](https://unsplash.com/@ytcount?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

Youtube 视频已经成为我们日常生活的一部分。不像电视，它帮助我们观看我们感兴趣的内容。

我还广泛使用 youtube 进行娱乐和教育。

由于广告是 youtube.com 的主要收入来源，因此，他们前所未有地使用它。

我遇到的一个问题是，我有边吃饭边看电视的习惯。现在在 youtube 上，当我看一个节目/视频时，每当有广告出现时，我都必须点击跳过广告按钮。

此外，我习惯在聆听灵性导师的教导时冥想。每次广告来的时候，我的沉思都会被打扰。

因此，我决定创建一个机器学习模型，它可以识别屏幕上是否有广告，并点击跳过广告按钮。

我开始使用 Windows snipping 工具收集数据集。由于神经网络需要大量的数据来训练，我收集了大约 600 张图像。几乎 300 张图片包含跳过广告按钮，另外 300 张图片属于其他类别标签。

我意识到用手动方式收集图像是非常困难的。我知道我可以使用市场上的任何软件录制屏幕。因此，我在互联网上搜索分离视频的帧，以便我可以将每一帧作为单独的图像。

OpenCV 可以选择这样做。所以，我了解了这一点，并把每一帧分开。数据收集部分变得非常快。现在我有大约 1800 张图片来训练一个模型。

我意识到大尺寸的图像即使在谷歌实验室也很难训练。我缩小了图像的尺寸，尽管它占用了太多的空间，以至于每次我训练一个模型时，Google Colab 笔记本都会崩溃。

## 特征工程

我尝试了另一种技术，其中我只选择了图像中有跳过按钮的部分。我使用 numpy 索引切片只选择图像的一部分。

正如预期的那样，训练进行得很顺利，模型的训练准确率达到了 99%左右。

我使用 pyautogui 库点击屏幕上的特定点。

很快我意识到有一个问题，不同的屏幕可以有不同的分辨率，跳过按钮可以在不同的位置，这取决于缩放或全屏模式或影院视图。

## 策略的改变

当我学习 OpenCV 时，我开始了解 OpenCV 中的 SIFT 特性和模板匹配特性。

我发现 OpenCV 中的模板匹配是我试图解决的问题的准确解决方案。

我根据屏幕的不同分辨率和缩放级别创建了不同的模板。我在两台不同的笔记本电脑上以不同的分辨率彻底测试了这个脚本。

该应用程序在跨平台上运行良好。

令人惊讶的是，它在另一个名为 voot.com 的视频流媒体应用程序中也显示出了良好的效果，一些印度电视节目就是在这个应用程序中播放的。

## 使用的各种库

```
import cv2
import numpy as np
import pyautogui
import time
```

Numpy 用于将 python 图像对象转换成 numpy 数组，以便在 OpenCV 的模板匹配方法中使用。

## 阅读模板

```
# reading the templates
template3 = cv2.imread('template3.png', 0)
template4 = cv2.imread('template4.png', 0)
template5 = cv2.imread('template5.png', 0)
template6 = cv2.imread('template6.png', 0)
```

模板以灰度格式读取，如标志“0”所示。经过全面测试后，我根据不同的缩放级别创建了四个模板。

这些模板涵盖了所有情况(分辨率、缩放和全屏)。

## 阈值处理

```
# setting the threshold for confidence in template matching
threshold = 0.7
```

我将阈值设置为 0.7，这样当图像中某个区域与模板匹配的概率很高时，才考虑该点。

## 停止标准

```
# alert box for stopping criteria
pyautogui.alert(text = 'Keep the mouse pointer on the top left/ corner of screen to stop the program', title= 'Stopping Criteria')
```

在程序开始时，会弹出一个警告框，提示如何停止脚本。

我没有使用任何 GUI 来启动和停止脚本。更确切地说，我使用了一种黑客技术，在这种技术中，我检查特定的条件以跳出程序。

## 连续循环

While 循环检查模板匹配并单击 skip 按钮

这个循环将一个接一个地检查一个模板，一旦找到匹配，它将根据找到它的像素位置点击它。

```
#     Stopping criteria    
if pyautogui.position() == (0,0):
     pyautogui.alert(text = 'Adskipper is Closed', title =/    'Adskipper Closed')
     break
```

在检查完所有的模板后，它检查鼠标的位置是否在(0，0)处，如果是，那么在显示一个警告框，提示程序关闭后，它退出循环。

## 进一步的改进

我们可以使用基于深度学习的图像分割技术，如 U-NET、SegNET 等。这些是像素级图像分割技术。通过这种方式，我们不必创建不同的模板，也没有机会遗漏任何死角，因为方法是完全不同的。

我们也可以把它扩展到手机上。

我们还可以为用户创建一个 GUI，以便他/她可以轻松地打开或关闭它。

完整的源代码可以在我的 Github 档案中找到:[https://github.com/1993jayant/youtube_adskipper](https://github.com/1993jayant/youtube_adskipper)