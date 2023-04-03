# Blender、CUDA 和 Linux:皆大欢喜的故事

> 原文：<https://medium.com/analytics-vidhya/blender-cuda-and-linux-the-happy-end-story-31ebe3708402?source=collection_archive---------15----------------------->

嗨，伙计们。它是一个**非常**的小指南，帮助你在 Blender 中使用你的 GPU。

我已经花了大约 5 个小时试图解决这些问题，我希望我能为你节省一些时间来描述我所面临的问题的解决方案。

我们开始吧。

![](img/bf2905efe80c2125fb9abd91afbe55c5.png)

# Blender 看不到我的 GPU，系统标签里也没什么可勾选的

也许你的安装包列表有问题。

我在[https://forums . developer . NVIDIA . com/t/NVIDIA-driver-conflicts-with-cuda-drivers/63009/3](https://forums.developer.nvidia.com/t/nvidia-driver-conflicts-with-cuda-drivers/63009/3)中描述了一种情况，但我不认为这种冲突会阻止 Blender 查看我的 GPU。

我去掉了`xorg-x11-drv-nvidia-cuda-libs`，一切开始正常工作。

# 我有一个不正确的版本的 GCC 的 CUDA

首先，你需要安装正确的，它不会取代你现在的`gcc`。如果你是 Fedora 用户，那么从 Negativo17 repository 安装`cuda-gcc`和`cuda-gcc-c++`包(顺便说一下，安装 CUDA 本身也是这个 repo)。

如果你不是 Fedora 用户，那么你可以自己在谷歌上搜索你自己系统的解决方案。

然后你需要指定 CUDA 应该使用这个`cuda-gcc`而不是普通版本。

```
sudo ln -s /usr/bin/cuda-gcc /usr/local/cuda/bin/gcc
```

您应该指定 CUDA 安装的路径，而不是最后一个路径。

# 我没有 OptiX 的搅拌机 2.83

我是 Fedora 用户，用的是从`dnf`开始安装的 blender。

这是错误的方法。从`dnf`上取下搅拌机，并使用`snap`安装。然后你会有 OptiX。