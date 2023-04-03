# 使用 GPU 快速入门

> 原文：<https://medium.com/analytics-vidhya/getting-started-with-fast-ai-with-gpu-4f159657d8c0?source=collection_archive---------27----------------------->

![](img/879620bf61fefe2db2da7ede3527e207.png)

照片由 [Greg Jeanneau](https://unsplash.com/@gregjeanneau?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

我们都经历过这种情况，有时很难一开始就让一切顺利进行。我将会为如何安装 fastai 库提供一个全面的指导。在 windows 中安装 fastai 库一般有两种方法

*   使用 Python
*   使用蟒蛇

我将用 Python 来覆盖这一部分，因为我总觉得它更容易。

**为了安全起见，你最好卸载已经安装的任何以前的 python 版本。**

1.  从[这里](https://www.python.org/ftp/python/3.6.0/python-3.6.0-amd64.exe)下载并安装 Python 3.x 64 位。**确保您勾选了将 Python 添加到路径复选框。**
2.  从 [**NVIDIA**](https://developer.nvidia.com/cuda-downloads) 下载安装 CUDA 工具包。选择您的架构和操作系统并下载软件包。访问此[站点](https://developer.nvidia.com/cuda-gpus)检查您的 GPU 是否支持 CUDA。如果您没有启用 CUDA 的 Nvidia GPU，请跳过这一步。
3.  fastai 库运行在 PyTorch 之上，所以我们必须从[这里](https://pytorch.org/)安装 PyTorch。
    选择
    PyTorch Build=Stable
    您的操作系统= Windows
    Package = Pip
    Language = Python
    CUDA = 10.1，或者如果您在步骤 2 中安装了不同的 CUDA 版本，请进行更改。
    复制生成的命令并在您的终端上运行。PyTorch 将在 GPU 支持下安装。
4.  现在我们需要安装构建工具，下载[这个包](https://download.microsoft.com/download/5/f/7/5f7acaeb-8363-451f-9425-68a90f98b238/visualcppbuildtools_full.exe)，并在安装过程中选中默认复选框。
5.  完成上述步骤后，在您的终端中执行以下命令来安装 fastai 库。

```
pip install fastai
```

注意，这将只安装 fastai 库，你可能需要其他软件包，如 jupyter notebook、numpy 和 pandas 等，要安装这些副本，请粘贴以下命令。

```
pip install jupyter 
pip install pandas
pip install seaborn
pip install matplotlib
pip install numpy
```

这些只是必需的基本包。