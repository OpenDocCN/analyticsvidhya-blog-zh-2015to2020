# 在 ArchLinux 上用 Python3.8 和 OpenCV4.2.0 编译 CAFFE

> 原文：<https://medium.com/analytics-vidhya/compiling-caffe-with-python3-8-and-opencv4-2-0-on-archlinux-db2c90370554?source=collection_archive---------3----------------------->

![](img/b5d394d53084b75bd95245c20f5748b9.png)

由[克里斯·巴巴利斯](https://unsplash.com/@cbarbalis?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片。

这篇博客是关于在 ArchLinux 上安装 CAFFE v1.0 的。 [GitHub](https://github.com/BVLC/caffe) 上的默认 CAFFE 发行版无法与 OpenCV 4 . 2 . 0 版和 Python 3.8 编译。OpenCV 特别成问题。我找不到任何一个帖子或类似的东西，可以带我一步一步地经历这些变化，而这正是这篇帖子所做的。

你当然可以安装旧版本的 OpenCV，但是有新版本有什么意义呢？此外，这些变化应该独立于 Linux 发行版，因此可以在 Ubuntu 或任何其他发行版中尝试。但是我只在 ArchLinux 上测试过。

***免责声明:这只是基于我在我的机器上的经验，并不保证在任何其他系统中都能工作。这只是一个指南，并且是在 ArchLinux 上测试的，但是可能适用于其他 Linux 发行版。***

## Python 文件的更改

![](img/2ecc87ba665616aa350058d86aa6f5a2.png)

[沙哈达特·拉赫曼](https://unsplash.com/@hishahadat?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片。

但是用新版本编译它并不困难，在编译它之前需要对不同的文件做一些修改。蟒蛇的很简单。在 Python 2.7 中，打印语句如下所示:

```
print "My name"
```

然而，在 Python 3 的任何版本中，如 3.6、3.7 或 3.8，上面的打印命令看起来像这样:

```
print("My name")
```

因此，关于 print 语句的任何错误都需要以同样的方式进行更正。Python 提供了一个名为 2to3 的脚本，它安装在系统路径上，可以进行打印校正。与`-w` 开关一起使用。

## 对 Makefile.config 的更改

![](img/e4d02c24f45ad188f4f70003efd5335a.png)

互联网上提供标准 OpenCV 图像。

对于 OpenCV 的问题，我从下面 GitHub 上的补丁引了一个:[https://GitHub . com/AlexandrParkhomenko/aur/blob/master/caffe/readme fix！！！。补丁](https://github.com/AlexandrParkhomenko/aur/blob/master/caffe/readmeFix!!!.patch))。第一个更改是在 Makefile.config 文件中。可以在 Makefile.config 或 Makefile.config.example 上进行更改，然后执行以下命令。

```
cp Makefile.config.example Makefile.config
```

下面提到的行号指的是主 GitHub 存储库中的原始文件，在您编辑文件后可能会有所改变。最好在进行更改之前备份原始文件。这些变化是:

1.  第 23 行，修改 OpenCV 的版本

```
23: OPENCV_VERSION := 4
```

2.如果您在 ArchLinux 上构建 CUDA 版本，那么 CUDA 安装在一个不同于默认目录的目录中，通常是/opt/cuda，所以在第 30 行做如下修改。

```
30: CUDA_DIR := /opt/cuda
```

对于其他 Linux 发行版，请检查 cuda 的安装位置。如果您的处理器支持 [CPU 节流或动态频率调节](http://CPU throttling or dynamic frequency scaling)，请安装 OpenBlas 库，而不是安装 Atlas 库。Makefile.config 提供了一个使用 Python 3 的选项，只要确保它是 Python3.8 就行了，因为目标机器上可以安装多个版本的 Python。此外，对于 ArchLinux，Python 的 Numpy 核心安装在/usr/lib/python3.8/ **站点** -packages/numpy…而不是/usr/lib/Python 3.8/**dist**-packages/Numpy…

```
80: # Uncomment to use Python 3 (default is Python 2)
81: PYTHON_LIBRARIES := boost_python3 python3.8
82: PYTHON_INCLUDE := /usr/include/python3.8 \
                 /usr/lib/python3.8/site-packages/numpy/core/include
```

需要在包含目录中显式包含 OpenCV 4 路径，以便构建过程找到 OpenCV。需要进行以下更改:

```
96: # Whatever else you find you need goes here.
97: INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/opencv4 #explicity included the opencv path
```

## 对 Makefile 的更改

Makefile 本身不需要太多的修改。在第 203 行，需要提到 OpenCV 的版本。从同一行开始，用以下代码替换原始代码:

```
203: ifeq ($(OPENCV_VERSION), $(filter $(OPENCV_VERSION), 3 4))
  LIBRARIES += opencv_imgcodecs
204: endif
205: ifeq ($(OPENCV_VERSION), 4)
206:  ifeq ($(USE_PKG_CONFIG), 1)
   INCLUDE_DIRS += $(shell pkg-config opencv4 --cflags-only-I | sed 's/-I//g')
207:  else
208:   INCLUDE_DIRS += /usr/include/opencv4 /usr/local/include/opencv4
209:   INCLUDE_DIRS += /usr/include/opencv4/opencv2 /usr/local/include/opencv4/opencv
210:  endif 
211: endif
```

需要在第 208 行更改 Python 库的名称，如下所示:

```
208: PYTHON_LIBRARIES ?= boost_python3 python3.8 #changed from boost_python python2.7
```

如果您使用 OpenBlas 库而不是 Atlas 进行高级线性代数运算，请在第 388 行进行以下更改:

```
388: LIBRARIES += blas #instead of openblas
```

替换第 420 行，其中给出了一些用于 C++编译的标志，如下所示:

```
419: # Automatic dependency generation (nvcc is handled separately)
420: COMMON_FLAGS += -D_GLIBCXX_USE_CXX11_ABI=1
421: CXXFLAGS += -MMD -MP -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=1
```

在 Makefile 文件中还需要做一点修改，以便在第 432 行反映 OpenCV 4 的用法:

```
430: USE_PKG_CONFIG ?= 0
431: ifeq ($(USE_PKG_CONFIG), 1)
432:  #PKG_CONFIG := $(shell pkg-config opencv --libs)
433:  ifeq ($(OPENCV_VERSION), 4)
434:   PKG_CONFIG := $(shell pkg-config opencv4 --libs)
435:  else
436:   PKG_CONFIG := $(shell pkg-config opencv --libs)
436:  endif
437: else
438:  PKG_CONFIG :=
439: endif
```

## C++文件中的更改

在一些`C++`文件中需要更多的修改。在 OpenCV4 中，为了读取图像，一些枚举的名称已经改变。下面的文件使用旧的`CV_LOAD_IMAGE_COLOR`和`CV_LOAD_IMAGE_GRAYSCALE`的枚举。必须在以下文件中更改这些枚举

*   `caffe/src/caffe/util/io.cpp`
*   `caffe/src/caffe/layers/window_data_layer.cpp`
*   `caffe/src/caffe/test/test_io.cpp`

例如，在第 77 行的第一个文件中，进行以下更改

```
77: int cv_read_flag = (is_color ? cv::IMREAD_COLOR://CV_LOAD_IMAGE_COLOR :
        cv::IMREAD_GRAYSCALE);//CV_LOAD_IMAGE_GRAYSCALE);
```

同样，其他三个文件也需要修改。

## GitHub 知识库

![](img/67bfecb7bfba876e08664ddcafadabb2.png)

GitHub 的标准 logo。

上面提到的所有改变都可以在我自己的 Caffe 框架中找到，这个框架可以在[www.github.com/asadalam/caffe](https://github.com/asadalam/caffe)上找到。按照[咖啡馆主页上的说明进行安装。](http://caffe.berkeleyvision.org/installation.html)我对上面列出的更改进行了两次验证，它可以在我的系统上工作，但它是在没有任何担保的情况下发布的，并且是“按原样”提供的，我不对任何问题负责。本质上，GitHub 存储库上的许可文件说了什么。