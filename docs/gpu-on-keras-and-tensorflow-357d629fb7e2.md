# Keras 和 Tensorflow 上的 GPU

> 原文：<https://medium.com/analytics-vidhya/gpu-on-keras-and-tensorflow-357d629fb7e2?source=collection_archive---------8----------------------->

好奇的人们，你们好！

![](img/83e7ced0ae4eb8b83d5fbe7d11f28f7d.png)

由[大卫·拉托雷·罗梅罗](https://unsplash.com/@latorware?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

发表这篇关于如何在 Keras 和 Tensorflow 上使用 GPU 的博客。如果你不太喜欢 GPU，我建议你快速检查一下 GPU 的症结所在。

嗯，早期用于游戏的 GPU 现在已经用于机器学习和深度学习。Tensorflow 或 Keras 上的神经网络必须使用 GPU。此外，令人惊讶的是，这些技术在初始化时使用整个 GPU。因此，这可能给多用户环境设置带来问题。完全不用担心！这个博客有它的解决方案。
但在开始之前，让我们先了解一下如何在 Tensorflow 和 Keras 上使用。

# GPU 上的张量流

Tensorflow 在启动时会自动分配整个 GPU。这可能会导致各种问题。

**问题**:我们不会去了解实际的 GPU 使用情况。多用户环境设置有点令人担忧，当多用户同时访问 1 个 GPU 时会出现警报情况。

**解决方案一:**

这适用于多用户环境。无法指定所需的确切 GPU 内存量，因此***allow _ growth****进入画面*。它允许运行时分配内存。将它设置为 true 意味着它开始时分配很少的内存，然后根据进程需求逐渐分配更多的区域。

> config = tf。config proto()
> config . GPU _ options . allow _ growth = True
> sess = TF。会话(配置=配置)

运筹学

> config = tf。config proto()
> TF . config . GPU . set _ per _ process _ memory _ growth = True
> sess = TF。会话(配置=配置)

**方案二:**

当您确定进程的内存消耗时，可以应用这个解决方案。固定的内存分配可以通过使用**set _ per _ process _ memory _ growth**指定所需的内存部分来完成，而不是动态的 GPU 分配。

> config = tf。config proto()
> config . GPU _ options . set _ per _ process _ memory _ growth = 0.4
> sess = TF。会话(配置=配置)

上面分配了每个 GPU 的固定百分比内存。例如，上面的 40%分配在每个 GPU 上。

# **GPU 上的 Keras**

在 Keras 中，对于后端 Tensorflow 或 CNTK，如果检测到任何 GPU，则代码将自动在 GPU 上运行，而后端则需要一个定制的函数。Keras 文档是一个非常棒和读者友好的。有关更多详细信息，请访问以下文档的常见问题。

[https://keras . io/getting-started/FAQ/# how-can-I-run-a-keras-model-on-multi-GPU](https://keras.io/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus)

# 常见问题解答

**问:如何检查每个进程正在使用的所有 GPU 或 GPU 利用率？**

*有 NVIDIA 管理和监控命令行实用工具“nvidia-smi”。访问* [*了解 nvidia-smi 实用程序*](/@shachikaul35/explained-output-of-nvidia-smi-utility-fc4fbee3b124) *的输出详情。*

**问:如何检查你的系统中是否存在 GPU？**

> tf.test.is_gpu_available()

**问:如何列出 tensorflow 可用的所有物理设备？**

> 从 tensorflow.python.client 导入设备库
> print(设备库.列表本地设备())

**问:如何向 tensorflow 列出所有可用的 GPU？**

> all _ GPU = TF . config . experimental . list _ physical _ devices(" GPU ")
> for GPU in all _ GPU:
> print(" Name:"，gpu.name，" Type:"，gpu.device_type)

运筹学

> if TF . test . gpu _ Device _ name():
> print('默认 GPU 设备:{} '。format(tf . test . gpu _ device _ name())
> else:
> print("请安装 TF 的 GPU 版本")

**问:如何验证 Keras 是否有 GPU？**

> 从 keras 导入后端为 K
> K.tensorflow_backend。_ get _ available _ gpus()

谢谢，

快乐阅读！

***可以通过***[***LinkedIn***](https://www.linkedin.com/in/kaul-shachi)***取得联系。***