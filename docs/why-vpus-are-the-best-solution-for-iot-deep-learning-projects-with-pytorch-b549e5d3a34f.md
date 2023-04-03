# 为什么 VPU 是物联网深度学习项目的最佳解决方案(与 Pytorch 合作)

> 原文：<https://medium.com/analytics-vidhya/why-vpus-are-the-best-solution-for-iot-deep-learning-projects-with-pytorch-b549e5d3a34f?source=collection_archive---------11----------------------->

![](img/18339dc7e12f96facd532426e3485dc9.png)

图片来自 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=3041437) 的 [xresch](https://pixabay.com/users/xresch-7410129/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=3041437)

**VPU** (视觉处理单元)是一种在物联网设备上处理视觉数据的新技术。

如今，tech world 提供了一整套工具，为你的物联网设备提供人工智能应用。例如，你会发现自己正在考虑购买英伟达杰特森纳米或珊瑚 TPU

有许多不同的方法和不同的解决方案。

但是，在设计您的项目时，请记住一条简单的建议:使用哪些物联网设备**功耗和效率是您的项目工作的主要要求，散热问题不是次要的。**

这就是为什么 VPU 的解决方案可能会派上用场。

# VPU 是由什么组成的

## 1.超长指令字处理器(旧技术复兴)

> **VLIWs 可以实现更高的性能，以更低的芯片和功耗成本提供更多的 ILP。**

J. A. Fisher 早在 1983 年的论文*“超长指令字体系结构和 ELI-512”*(Proc。10 年。里面的症状。计算机体系结构，1983 年 6 月，第 140-150 页):

> 更正式地说，VLIW 体系结构具有以下特性:
> 
> 每个周期有一个中央控制单元发出单个长指令。每个长指令由许多紧密耦合的独立操作组成。每个操作都需要少量的、静态可预测的周期来执行。操作可以流水线化。

你可以在这里找到费希尔、保罗·法拉博斯基和克里夫·杨最近写的一篇文章:

[https://www . research gate . net/publication/224535028 _ VLIW _ processors _ once _ blue _ sky _ now _ average](https://www.researchgate.net/publication/224535028_VLIW_processors_once_blue_sky_now_commonplace)

进一步解释了 VLIW 编译器如何工作以及为什么 VLIW 被如此命名:

> **VLIW 的编译器提前重新排列程序**，挑选发布什么，什么时候发布，以便在保持正确行为的同时最大化并行执行。**当程序运行时，其他处理器(称为超标量)依靠硬件来完成这项工作**。
> 
> 由于 VLIW 编译器提供了许多要一次发出的操作，通常要求它将它们捆绑成一个很长的指令字，因此称为 VLIW (64 -1024 位)

## 2.同质片上存储器

在 VPU，你的数据存储在芯片上，以尽量减少延迟和芯片外数据传输

> 集中式片内存储器架构允许高达 400 GB/秒的内部带宽

## 3.视觉加速器

视觉加速器是专门的处理器，旨在以超低功耗提供高性能机器视觉，而不会引入额外的计算开销。

换句话说，有处理器完全致力于处理你的视频帧。

## 4.神经计算引擎

运行设备上深度神经网络应用的专用硬件加速器。

# VPUs 相比 GPU 真的效率高吗？

由于缺乏文献和研究，我将参考最好的研究之一**探索视觉处理单元作为推理的协处理器**(【https://core.ac.uk/download/pdf/185526545.pdf】T2

> 使用基于 Szegedy 等人[3]的 GoogLeNet 工作的预训练网络模型，我们观察到，与参考 CPU 和 GPU 实现相比，在单个 VPU 芯片上进行推理时的性能仅慢 4 倍。然而，通过采用多 VPU 配置，我们展示了等效的性能结果。然而，**预期热设计功率(TDP)仍然可以降低 8 倍**。

# VPU 机器学习框架

虽然珊瑚 TPU 由谷歌制造，仅支持 Tensorflow 框架，但英特尔 VPU(神经计算棒 2)可以通过开放式神经网络交换转换支持 TensorFlow、Caffe、ApacheMXNet、开放式神经网络交换、 **PyTorch** 和 PaddlePadle。

如果您想了解更多关于英特尔 VPU 的信息，您可以阅读这篇发表在 2015 年 3 月 IEEE Micro 上的文章:

[https://www . researchgate . net/publication/275365424 _ Always-on _ Vision _ Processing _ Unit _ for _ Mobile _ Applications](https://www.researchgate.net/publication/275365424_Always-on_Vision_Processing_Unit_for_Mobile_Applications)

 [## 英特尔 Movidius 视觉处理单元(vpu)

### 英特尔 m ovidius VPUs 可高效支持要求苛刻的计算机视觉和边缘人工智能工作负载。通过高度耦合…

www.intel.com](https://www.intel.com/content/www/us/en/products/processors/movidius-vpu.html)