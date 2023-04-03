# ODROID-XU4 上的 Bazel 1.1.0

> 原文：<https://medium.com/analytics-vidhya/iot-bazel-1-1-0-on-odroid-xu4-f38e4825c78e?source=collection_archive---------17----------------------->

## 如何从零开始构建 baz El 1 . 1 . 0(bootstrapping)并安装在 ODROID-XU4 + Ubuntu 18.04.3 LTS(仿生海狸)上？

*你可以在我的 Github 库上看到下面的脚本:*【https://github.com/zmacario/Bazel-1.1.0-on-ODROID-XU4 

重要提示:我相信这个食谱也适用于其他版本的 Bazel。我已经测试了 0.26.1、0.27.1 和 0.29.1 版本，在所有情况下都运行良好。

## 1.让我们从在终端窗口中使用以下命令更新 ODROID Linux 开始:

```
sudo apt updatesudo apt upgrade
```

## 2.安装以下软件包:

```
sudo apt -y install build-essential openjdk-8-jdk python zip unzip
```

重要提示:在以下步骤中，您必须确保 OpenJDK v8 是您的系统的缺省值。

## 3.下载官方 Bazel 1.1.0 压缩源文件:

```
sudo wget https://github.com/bazelbuild/bazel/releases/download/1.1.0/bazel-1.1.0-dist.zip
```

## 4.解压缩下载的文件:

```
sudo unzip bazel-1.1.0-dist.zip -d bazel-1.1.0-dist
```

## 5.向一些解压缩的脚本授予执行权限:

```
cd bazel-1.1.0-dist/srcsudo chmod 777 *.shcd ..
```

## 6.在会话中使用以下命令扩展对系统资源的访问限制:

```
ulimit -c unlimited
```

## 7.使用以下命令启动 Bazel 的构建:

```
sudo env BAZEL_JAVAC_OPTS="-J-Xms1g -J-Xmx1g" EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk --discard_analysis_cache --nokeep_state_after_build --notrack_incremental_state" bash ./compile.sh
```

构建过程将需要半个小时。

## 8.验证构建的 Bazel 二进制文件:

```
cd output./bazel version
```

## 9.用一个简单的命令在您的系统上安装 Bazel:

```
sudo cp bazel /usr/bin
```

*嘿！现在您的系统上安装了最新版本的 Bazel！*

其他相关帖子:“[*IoT:ODROID 上的 OpenCV 4.1-XU4*](/@jose.macario/iot-opencv-4-1-on-odroid-xu4-8a14d395f191)*”*

看看我谦虚的 YouTube 频道:[*https://www.youtube.com/user/josemacariofaria/*](https://www.youtube.com/user/josemacariofaria/)