# IoT:ODROID-XU4 上的 OpenCV 4.1

> 原文：<https://medium.com/analytics-vidhya/iot-opencv-4-1-on-odroid-xu4-8a14d395f191?source=collection_archive---------7----------------------->

## 如何在 ODROID-XU4 平台上，在 Ubuntu 18.04.3 LTS(仿生海狸配 Python 3.6.8)上编译安装 OpenCV 4.1.2？

*你可以在我的 Github 库上看到下面的脚本:*【https://github.com/zmacario/OpenCV-4.1-on-ODROID-XU4 

## 1.首先，在 Linux 终端上使用以下命令更新您的系统:

```
**sudo apt update****sudo apt upgrade**
```

## 2.安装这些工具:

```
**sudo apt -y install build-essential cmake gfortran pkg-config unzip software-properties-common doxygen**
```

## 3.使用以下软件更新/安装 Python、Pip 和 Numpy:

```
**sudo apt -y install python-dev python-pip python3-dev python3-pip python3-testresources****sudo apt -y install python-numpy python3-numpy**
```

## 4.安装大量保证或增加 OpenCV 特性的东西:

```
**sudo apt -y install libblas-dev libblas-test liblapack-dev libatlas-base-dev libopenblas-base libopenblas-dev****sudo apt -y install libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev****sudo apt -y install libxvidcore-dev libx264-dev****sudo apt -y install libgtk2.0-dev libgtk-3-dev libcanberra-gtk*****sudo apt -y install libtiff5-dev libeigen3-dev libtheora-dev 
libvorbis-dev sphinx-common libtbb-dev yasm libopencore-amrwb-dev****sudo apt -y install libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libavutil-dev libavfilter-dev****sudo apt -y install libavresample-dev ffmpeg libdc1394-22-dev libwebp-dev****sudo apt -y install libjpeg8-dev libxine2-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libprotobuf-dev****sudo apt -y install protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libhdf5-dev****sudo apt -y install qt5-default v4l-utils****sudo apt -y install libtbb2**
```

*耐心点，这需要很长时间……*

## 5.从 Ubuntu 16.04 安装旧库(这是必要的):

```
**sudo add-apt-repository "deb** [**http://ports.ubuntu.com/ubuntu-ports**](http://ports.ubuntu.com/ubuntu-ports) **xenial-security main"****sudo apt -y update****sudo apt -y install libjasper-dev libjasper**
```

## 6.创建一个文件夹来接收您(将来)编译的 OpenCV 4.1.2 包:

```
**mkdir opencv_package**
```

你可以在任何你想要的地方创建这个文件夹，但是当你开始编译器配置步骤时，你必须记住它的完整路径。

*观察:我在“/home/odroid/Desktop/”中执行的所有后续步骤(包括这一步)。*

## 7.下载官方 OpenCV 4.1.2 压缩源文件:

```
**wget -O opencv.zip** [**https://github.com/opencv/opencv/archive/4.1.2.zip**](https://github.com/opencv/opencv/archive/4.1.2.zip)**wget -O opencv_contrib.zip** [**https://github.com/opencv/opencv_contrib/archive/4.1.2.zip**](https://github.com/opencv/opencv_contrib/archive/4.1.2.zip)
```

## 8.解压缩下载的源文件:

```
**unzip opencv.zip****unzip opencv_contrib.zip**
```

## 9.为了方便起见，重命名解压缩的文件夹:

```
**mv opencv-4.1.2 opencv****mv opencv_contrib-4.1.2 opencv_contrib**
```

## 10.进入解压后的 OpenCV 源文件夹，创建一个编译器应该使用的临时工作文件夹:

```
**cd opencv****mkdir build****cd build**
```

## 11.使用以下命令配置编译器:

```
**cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=/home/odroid/Desktop/opencv_contrib/modules \
-D PYTHON_EXECUTABLE=/usr/bin/python3.6 \
-D PYTHON2_EXECUTABLE=/usr/bin/python2.7 \
-D PYTHON3_EXECUTABLE=/usr/bin/python3.6 \
-D PYTHON_PACKAGES_PATH=/usr/lib/python3/dist-packages \
-D PYTHON_LIBRARY=/usr/lib/python3.6/config-3.6m-arm-linux-gnueabihf/libpython3.6m.so \
-D PYTHON_INCLUDE_DIR=/usr/include/python3.6 \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PYTHON3_INSTALL_PATH=/home/odroid/Desktop/opencv_package \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_DOCS=NO \
-D BUILD_TIFF=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_TBB=ON \
-D BUILD_TBB=ON \
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON \
-D WITH_VTK=OFF \
-D WITH_OPENGL=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D BUILD_TESTS=OFF \
-D BUILD_EXAMPLES=OFF ..**
```

*观察:我决定显式启用我目前所知的所有 OpenCV 特性，并排除其所有文档和示例。*

## 12.开始编译:

```
**make -j4**
```

*观察:由于 ODROID-XU4 有 8 个内核，您将使用其中的一半，并带有“-j4”参数。*

现在，再耐心点！编译需要很长时间…

## 13.最后安装编译好的 OpenCV 4.1.2:

```
**sudo make install****sudo ldconfig****sudo apt update**
```

# 最后的考虑:

*在编译结束时，创建的真正重要的二进制对象将是位于“opencv_package”文件夹中的“cv2 . cpython-36m-arm-Linux-gnueabihf . so”。不要删！*

*Odroid XU4 厂商官方链接供 Ubuntu 下载我用:*[*https://Odroid . in/Ubuntu _ 18.04 lts/XU3 _ XU4 _ MC1 _ HC1 _ HC2/Ubuntu-18 . 04 . 3-4.14-mate-Odroid-XU4-2019 09 29 . img . xz*](https://odroid.in/ubuntu_18.04lts/XU3_XU4_MC1_HC1_HC2/ubuntu-18.04.3-4.14-mate-odroid-xu4-20190929.img.xz)

不要忘记分别删除/移除您在步骤 7 至 10 中下载或创建的临时文件和文件夹。

*使用以下命令清理您的系统:*

```
**sudo apt autoremove****sudo apt clean**
```