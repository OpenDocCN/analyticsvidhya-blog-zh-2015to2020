# 通过检测微笑自动捕捉自拍

> 原文：<https://medium.com/analytics-vidhya/auto-capture-selfie-by-detecting-smile-1633218101ba?source=collection_archive---------12----------------------->

# 微笑时自动捕捉美丽的自拍 Python 项目可自动检测和捕捉自拍。

每个人都喜欢微笑的照片，所以我们将开发一个项目，将捕捉你每次微笑的图像。这是一个简单的初学者机器学习项目，我们将使用 OpenCV 库。

## OpenCV 是什么？

**OpenCV** (开源计算机视觉库)是一个开源的计算机视觉和机器学习软件库。 **OpenCV** 的建立是为了给计算机视觉应用提供一个通用的基础设施，并加速机器感知在商业产品中的使用。

# 项目先决条件

要实施这个项目，我们需要了解以下内容:

1.Python 的基本概念
2。openCV 基础。

要安装该库，您可以从命令行使用 pip 安装程序:

1.  pip 安装 OpenCV-python

# 下载培训代码和 XML 文件

# 开发项目的步骤

实施微笑检测和自拍捕捉项目的步骤

1.  我们首先导入 openCV 库。
2.  现在使用 cv2 的视频捕捉功能启动第二行的网络摄像头。
3.  然后，在 python 文件中包含 haarcascade 文件。
4.  视频只不过是一系列的图像，所以我们将运行一个无限的 while 循环。
5.  然后我们通过 read()从视频中读取图像。
6.  由于灰度图像中的特征识别更准确，我们将使用基本 openCV 函数 cvtColor()和 BGR2GRAY 将图像转换为灰度图像。
7.  现在，我们将使用已经包含的 haarcascade 文件和 detectMultiscale()函数来读取人脸，在该函数中，我们传递灰度图像、比例因子和 minNeighbors。

*   ScaleFactor:指定缩放图像的参数，精度取决于它，所以我们将使它接近 1，但不是非常接近，就像我们取 1.001(非常接近 1)，那么它甚至会检测阴影，所以 1.1 对面部来说足够好了。
*   minNeighbors:指定每个矩形应该有多少个邻居来保留它的参数。

1.  如果它检测到人脸，我们将使用 cv2 的 rectangle()方法绘制人脸的外部边界，该方法包含 5 个参数:图像、初始点(x，y)、主对角线的端点(x +宽度，y +高度)、矩形外围的颜色，最后一个参数是绘制的矩形外围的厚度。
2.  如果检测到面部，我们将同样检测到微笑，如果也检测到微笑，我们将打印保存在 cmd/终端中的图像<cnt>，然后我们必须提供要保存图像的文件夹的位置。</cnt>
3.  为了保存图像，我们将使用 imwrite()，它有两个参数——位置和图像。
4.  为了防止内存溢出，我们将只在一次运行中保存 2 个图像，因此使用 if 语句，如果 cnt>=2，该语句将中断循环。
5.  为了打破无限循环，我们使用了一个 if 语句，当我们按“q”表示“退出”时，该语句变为真。
6.  最后，我们将发布视频。
7.  不要忘记破坏所有的窗户。

# **代码:**

> 导入 cv2
> 视频= cv2。视频捕获(0)
> 
> faceCascade=cv2。cascade classifier(" G:/dataset/Haar scade _ frontal face _ default . XML ")
> smile scade = cv2。cascade classifier(" G:/dataset/Haar cascade _ smile . XML ")
> 
> while True:
> success，img = video . read()
> gray img = cv2 . CVT color(img，cv2。COLOR _ bgr 2 gray)
> faces = face cascade . detect scale(gray img，1.1，4)
> CNT = 1
> key pressed = cv2 . wait key(1)
> for x，y，w，h in faces:
> img = cv2 . rectangle(img，(x，y)，(x+w，y+h)，(0，0，0)，0)
> smiles = smile scade . detect scale(gray img，1.8，15)
> jpg'
> cv2.imwrite(path，img)
> CNT+= 1
> if(CNT>= 2):
> break
> 
> cv2 . im show('视频直播'，img)
> if(key pressed&0x ff = = ord(' q '):
> break
> video . release()
> cv2 . destroyallwindows()