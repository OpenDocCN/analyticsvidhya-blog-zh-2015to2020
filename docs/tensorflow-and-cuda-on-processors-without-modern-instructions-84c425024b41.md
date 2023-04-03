# 无现代指令的处理器上的 Tensorflow 和 CUDA

> 原文：<https://medium.com/analytics-vidhya/tensorflow-and-cuda-on-processors-without-modern-instructions-84c425024b41?source=collection_archive---------29----------------------->

虽然在大多数情况下简单的`pip install tensorflow`工作得很好，但某些硬件组合可能与存储库安装的 tensorflow 包不兼容。在这个简短的教程中，我将从源代码构建最新的 tensorflow 2.3.1 python 包。本教程也可能对那些想要在旧 GPU 上更新到最新 tensorflow 版本的人有所帮助，因为自 [2.3.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.3.0) 以来，旧硬件支持已从预编译版本中移除。

# 准备建筑环境

获取以下 docker 容器:

```
docker pull tensorflow/tensorflow:devel-gpu
```

选择一个位置并创建一个将与容器共享的目录。在我的例子中，我将使用`/home/alexandr/temp/tensorflow`。然后进入工作目录并启动 docker 容器

```
cd /home/alexandr/temp/tensorflow
docker run -it -w /tensorflow_src -v $(pwd):/share tensorflow/tensorflow:devel-gpu bash
```

更新容器中的存储库，并选择最新的稳定分支(在撰写本文时是 2.3 版)

```
git pull
git checkout r2.3
```

接下来，升级 pip 并安装一些 python 依赖项

```
/usr/bin/python3 -m pip install --upgrade pip
pip3 install six numpy wheel keras_applications keras_preprocessing
```

# 找出目标机器上的 CPU 限制

您需要告诉编译器在最终的二进制文件中应该避免哪些指令。这不是一个显而易见的步骤，因为这些限制是特定于机器的。您可能想要咨询 internet，甚至使用一些试错法来查看需要什么标志来使二进制文件在您的特定机器上保持稳定。

如果你在目标机器上编译，`-march=native`应该足够了，因为它应该启用你的 CPU 支持的所有指令。如果你交叉编译，就像我在这个例子中一样，那么你需要更深入地挖掘。

一种方法是查看显示 CPU 特性的`grep flags /proc/cpuinfo | head -n 1`。在我的情况下，我有一台笔记本电脑，其中最新的 tensorflow 可以开箱即用，还有一台带 GPU 的台式电脑，其中没有 GPU。比较两台机器的列表，我发现台式电脑缺少以下项目:`'ida', 'bmi2', 'smep', 'rtm', 'bmi1', 'fma', 'f16c', 'hle', 'avx2', 'smx', 'adx', 'avx', 'mpx'`。

[在这里](https://unix.stackexchange.com/questions/43539/what-do-the-flags-in-proc-cpuinfo-mean)我们可以看到`ida`代表英特尔动态加速，是 CPU 热量和电源管理的一部分，因此它不太可能是我的案例中的破坏因素。同样，`smep, rtm, hle, smx,`和`mpx`特性不太可能影响 tensorflow 的执行。我很难找到 tensorflow 二进制文件是否使用了`f16c`和`adx`。

另一方面，`avx`和`avx2`(高级向量扩展)、`bmi1, bmi2`(第一/组位操作扩展)和`fma`(融合乘加)似乎对 tensorflow 相当重要。因此，我将使用下面的标志组合来构建 tensorflow 二进制文件

```
-march=native -mno-avx -mno-avx2 -mno-fma -mno-bmi -mno-bmi2
```

# 配置 tensorflow 的构建链

在 docker 容器中执行

```
python3 configure.py
```

启动配置管理器。对于大多数问题，您可以选择默认答案。

一个重要的问题是关于你的 GPU 的计算能力。由于默认选项可能不包括您的 GPU 类型，因此最好提前检查一下[这里的](https://developer.nvidia.com/cuda-gpus)并输入到提供的字段中。如果您错误地指定了您的 GPU 计算能力，那么在 tensorflow 中尝试使用 CUDA 时，您可能会收到以下错误消息:

```
InternalError: CUDA runtime implicit initialization on GPU:0 failed. Status: device kernel image is invalid
```

当您遇到关于优化标志的问题时，您应该输入我们在上一节中提出的标志。在我的例子中，我还添加了一面`-Wno-sign-compare`旗。完成后，您可以通过运行以下命令开始构建过程

```
bazel build //tensorflow/tools/pip_package:build_pip_package --local_ram_resources=16384
```

如果 bazel 抱怨它的版本，它也可能会为您提供一个更新它的命令行程序。

构建过程需要大量的 RAM，尤其是在具有多个内核的机器上，因此您可能希望通过使用标志`--local_ram_resources=16384`来限制 RAM 的使用。在我的例子中，我将它限制在机器上可用的 24 GiB 中的 16 GiB。限制资源的另一种方式是使用标志`--jobs 4`将线程数量限制到一个较小的数量。

这个建筑要花很长时间。

# 准备并安装 python 包

执行以下命令来组装 python 包

```
bazel-bin/tensorflow/tools/pip_package/build_pip_package /share
```

现在，您应该能够在主机的挂载目录中看到一个`.whl`文件。将该文件复制到目标机器上，然后用`pip`安装。文件名建议您应该在目标机器上使用哪个 python 版本。如果您有不同的版本，可以使用 conda 的 environments 为 tensorflow 创建一个单独的环境，其中包含所需的 python 版本。

```
conda create -n "tensorflow2" python=3.6
conda activate tensorflow2
pip install tensorflow-2.3.1-cp36-cp36m-linux_x86_64.whl
```

现在，您应该能够在这个新环境中导入和使用 tensorflow。

就是这样！

附:这个故事最初出现在我的个人博客中，地址是[https://alexmoskalev . com/tensor flow-and-cuda-on-processors-without-modern-instructions/](https://alexmoskalev.com/tensorflow-and-cuda-on-processors-without-modern-instructions/)