# 使用 OpenVINO 的边缘设备上的人工智能模型—简介

> 原文：<https://medium.com/analytics-vidhya/ai-models-on-edge-devices-with-openvino-5a057bc50e07?source=collection_archive---------10----------------------->

![](img/872f6f5c2d76589a1689093086a6d27d.png)

# 什么是真正的边缘计算？

边缘计算意味着本地(或接近本地)处理，而不是云中的任何地方。它可以是一个实际的本地设备，如智能冰箱，或者尽可能靠近源的服务器。边缘计算可用于需要低延迟的地方，或者网络本身不总是可用的地方。

# 为什么选择边缘计算？

它的使用来自于在某些应用中对实时决策的渴望。通常，边缘计算与云一起提供灵活的解决方案，虽然边缘计算是自动驾驶汽车等实时工作负载的理想选择，但云可以为大规模分析提供集中位置。

# 在边缘部署深度学习模型

在边缘计算中，当我们说模型时，它泛指在云中或数据中心创建和训练并部署到边缘设备上的机器学习模型。

有了模型，我们如何将模型部署到边缘端点？在本文中，我将介绍在边缘设备上部署我们的模型。

为了对模型进行推理，我们需要模型的权重。xml)和二进制文件(。bin)，在演示中，我将使用英特尔开放式视觉推理和神经网络优化(OpenVINO)工具包。

# 关于使用 OpenVINO 和利用英特尔预训练模型的简介

我将演示在英特尔酷睿 i7–4510 u CPU @ 2.00 GHz 上部署预训练人脸识别模型。[点击此处](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/pretrained-models.html)查看 OpenVINO 工具包中可用的预训练模型及其应用。

首先，使用[这个资源](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html)下载 OpenVINO 工具包。
为了运行我们的人脸识别模型，我们将创建 2 个 python 文件。

1， **Model.py，**我们创建了一个 Face_Detection 类，在这个类中我们实例化了 openvino 网络，预处理了模型输入和输出，并异步运行推理。代码中有足够多的注释很好地解释了代码。

**2、推论. py** 文件。在这个文件中，我们编写了一个 InputFeeder 类，它将图像或视频输入到我们的模型中。然后我们创建一个主函数，它使用 Face_Detection 类和 InputFeeder 类来执行推理。然后通过命令行收集所需的参数。

# 在本地运行我们的模型

要运行我们的人脸检测模型，我们需要确保三件事。

**1** ，我们 openvino 环境的采购。要获取 openvino，运行-
*source/opt/Intel/openvino/bin/setup vars . sh-py ver 3.5，*请注意，根据您的操作系统和安装 open vino 的路径，命令可能会略有不同。
还要注意，您的环境所使用的 python 版本应该是您用来运行 python 文件的版本，以避免冲突。

下载我们的预训练模型。
转到 open vino toolkit 英特尔发行版中的模型下载器目录:
*CD/opt/Intel/open vino/deployment _ tools/tools/model _ downloader* 要下载特定目录中的模型，请运行以下命令
*sudo。/downloader . py—name Face-Detection-ADAS-0001-o/home/kolatimi/Desktop/Face-Detection-with-Intel-open vino/Model*

**3** ，Cd 到我们的项目目录并运行代码！！
你可以决定 [fork my repo](https://github.com/KolatimiDave/Face-Detection-with-Intel-OPENVINO) 并遵循详细的自述文件，或者你可以决定查看 github gist 原始文件并遵循代码。
如果您决定使用后者，在将代码下载到您的本地计算机后，将目录切换到您的 inference.py 文件所在的位置。然后运行-
*python 推论. py -v cam -m../Model/Intel/face-detection-ADAS-0001/FP16/face-detection-ADAS-0001 . XML*

# 再次见到你？

我很高兴我能够让你使用 OpenVINO 创建一个人工智能解决方案。
最后，我想补充一点，这是对 OpenVINO 用途的介绍，我期待着就更多有趣的话题给你写信，比如创建一个可以部署到不同设备的应用运行时包，比如 [VPU](https://en.wikipedia.org/wiki/Vision_processing_unit) 、 [FPGA](https://en.wikipedia.org/wiki/Field-programmable_gate_array) 和[集成 GPU](https://www.pcmag.com/encyclopedia/term/integrated-gpu) 。

![](img/5eec0ddbbc26bc59e6dbdf3ba5e57ba9.png)