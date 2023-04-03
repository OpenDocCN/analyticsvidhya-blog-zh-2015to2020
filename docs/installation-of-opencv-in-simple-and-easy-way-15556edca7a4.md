# 安装 OPENCV 的 5 个简单易行的步骤

> 原文：<https://medium.com/analytics-vidhya/installation-of-opencv-in-simple-and-easy-way-15556edca7a4?source=collection_archive---------0----------------------->

![](img/b5148a50a98eec88810c3671e03ab447.png)

**让我们……………OpenCV**

## 介绍

设置系统有时看起来有点复杂。出于同样的原因，我决定分享一个关于 OpenCV 安装的小教程。在本教程中，我们将学习在 Ubuntu 系统中设置 OpenCV-Python。以下步骤针对 Ubuntu 16.04 (64 位)和 Ubuntu 14.04 (32 位)进行了测试。我们开始吧！

OpenCV——Python 可以通过两种方式安装在 Ubuntu 中:

> ***1。从源代码编译。***
> 
> ***2。从 Ubuntu 存储库中预建的二进制文件安装。***

# ***1。从 Sou*rce 构建 OpenCV**

## 所需的生成依赖项

首先我们需要 **CMake** 来配置安装， **GCC** 来编译， **Python-devel** 和 **Numpy** 来构建 Python 绑定等等。以下是您需要运行的命令。

```
1\. sudo apt-get install cmake2\. sudo apt-get install python-devel numpy3\. sudo apt-get install gcc gcc-c++
```

接下来我们需要 **GTK** 对 GUI 特性的支持，相机支持(libv4l)，媒体支持(ffmpeg，gstreamer)等等。

```
1.sudo apt-get install gtk2-devel2\. sudo apt-get install libv4l-devel3\. sudo apt-get install ffmpeg-devel4\. sudo apt-get install gstreamer-plugins-base-devel
```

# 2.可选依赖项

以上依赖项足以在你的 Ubuntu 机器上安装 **OpenCV** 。但是根据您的需求，您可能需要一些额外的依赖项。下面给出了此类可选依赖项的列表。你要么留下它要么安装它，你的电话:)

OpenCV 附带图像格式的支持文件，如 **PNG、JPEG、JPEG2000、TIFF、WebP** 等。但是可能有点老了。如果你想获得最新的库，你可以为这些格式的系统库安装开发文件。

```
1\. sudo apt-get install libpng-devel2\. sudo apt-get install libjpeg-turbo-devel3\. sudo apt-get install jasper-devel4\. sudo apt-get install openexr-devel5\. sudo apt-get install libtiff-devel6\. sudo apt-get install libwebp-devel
```

# 3.下载 OpenCV

从 OpenCV 的 [GitHub 库](https://github.com/opencv/opencv)下载最新的源代码。

```
1\. sudo apt-get install git2\. git clone [https://github.com/opencv/opencv.git](https://github.com/opencv/opencv.git)
```

它将在当前目录下创建一个文件夹 **"opencv"** 。

然后现在打开一个终端窗口，导航到下载的**“opencv”**文件夹。创建一个新的**“build”**文件夹，并导航至该文件夹。

```
1\. cd ~/opencv2\. mkdir build3\. cd build
```

# 4.配置和安装

现在我们有了所有需要的依赖项，让我们安装 OpenCV。

```
1\. cmake ../
```

您应该在 CMake 输出中看到这些行(它们表示 Python 已被正确找到):

```
— Python 2:— Interpreter: /usr/bin/python2.7 (ver 2.7.6)— Libraries: /usr/lib/x86_64-linux-gnu/libpython2.7.so (ver 2.7.6)— numpy: /usr/lib/python2.7/dist-packages/numpy/core/include (ver 1.8.2)— packages path: lib/python2.7/dist-packages—— Python 3:— Interpreter: /usr/bin/python3.4 (ver 3.4.3)— Libraries: /usr/lib/x86_64-linux-gnu/libpython3.4m.so (ver 3.4.3)— numpy: /usr/lib/python3/dist-packages/numpy/core/include (ver 1.8.2)— packages path: lib/python3.4/dist-packages
```

现在，您使用**“make”**命令构建文件，并使用**“make install”**命令进行安装。

```
1\. make2\. sudo make install
```

# **5。安装结束**

所有文件都安装在 **"/usr/local/"** 文件夹中。打开一个终端，尝试导入**“cv2”。**

```
1\. import cv2 as cv2\. print(cv.__version__)
```

感谢您的阅读。更多轻松有趣的文章，请鼓掌关注。

[](https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html) [## OpenCV:在 Ubuntu 中安装 OpenCV-Python](https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html) [](https://docs.opencv.org/3.4.1/d7/d9f/tutorial_linux_install.html) [## OpenCV:在 Linux 中安装](https://docs.opencv.org/3.4.1/d7/d9f/tutorial_linux_install.html) [](/@ankitgupta_974/what-and-why-opencv-3b807ade73a0) [## 什么和为什么是 OpenCV？

### OpenCV 是用于实时计算机视觉的编程函数库。它是由英特尔开发的…

medium.com](/@ankitgupta_974/what-and-why-opencv-3b807ade73a0) [](https://github.com/ankitAMD) [## ankitAMD -概述

### 机器学习爱好者/计算机视觉。在 GitHub 上关注他们的代码。

github.com](https://github.com/ankitAMD)