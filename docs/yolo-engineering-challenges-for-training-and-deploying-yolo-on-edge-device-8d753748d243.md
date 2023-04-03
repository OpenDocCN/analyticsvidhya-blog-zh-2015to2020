# YOLO:在边缘设备上培训和部署 YOLO 的工程挑战

> 原文：<https://medium.com/analytics-vidhya/yolo-engineering-challenges-for-training-and-deploying-yolo-on-edge-device-8d753748d243?source=collection_archive---------10----------------------->

米提什·帕特尔，大卫·莎玛，尤利乌斯·贾赫迪

![](img/8e1fa824da2b78f982cc8acd24f39c6d.png)

YOLO 建筑

Y OLO(你只看一次)是一种新的对象检测方法，它使用单卷积神经网络，在一次评估中直接从完整图像中同时预测多个边界框和这些框的类别概率。该架构是高速的，可以每秒 45 帧或每秒 155 帧进行预测，对于较小的架构来说([在最初的 YOLO 论文](https://arxiv.org/abs/1506.02640)中声称)。

YOLO 最早由 Redmon 等人提出，在 2016 年 CVPR 上发表，并获得了人民选择奖。此后，人们对 YOLO 进行了大量改进(在撰写本文时，YOLOv4 是最新版本)。YOLO 的代码库， [darknet](https://pjreddie.com/darknet/yolo/) 是开源的，用 C 语言写的(*惊讶，*顺便说一下，《YOLO》的作者，Joseph Redmond 在 UW 教一门很棒的[计算机视觉课程](https://pjreddie.com/courses/computer-vision/))。从那以后，深度学习爱好者和研究人员开发了各种版本的 YOLO，支持不同的深度学习(DL)平台，如 Tensorflow、Pytorch 和 Caffe。

许多有用的帖子/论文解释了 YOLO，如[yolov 3 中的新功能](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)、[了解 YOLO](http://christopher5106.github.io/object/detectors/2017/08/10/bounding-box-object-detectors-understanding-yolo.html) 、[YOLO 的实时物体检测](/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088)等等。

在这篇文章中，我们将重点关注工程挑战，并分享我们在数据集上训练 YOLOv3 以及在英特尔或 Coral 的边缘设备上传输模型的经验。虽然 YOLO 上不乏帖子，但我们希望这篇文章将有助于减少像我们这样试图接触 YOLO 的爱好者的部署时间，并作为解决我们遇到的一些 ***问题*** 时刻的指南。

# 图书馆选择

有各种开源库随时可用，可用于在数据集上执行迁移学习。Joseph Redmond 开发的第一个版本是用 C 写的，在 GitHub 上。YOLO 还被移植到其他流行的 DL 库中，如 Tensorflow、Pytorch 和 CAFFE。这里列出了一些流行的 Github 实现，但并不详尽:

*   由 Joseph Redmond 用 C 实现的 Darknet
*   YOLO 在[实现了张量流](https://github.com/heartkilla/yolo-v3)
*   另一个版本的 [Tensorflow](https://github.com/shahkaran76/yolo_v3-tensorflow-ipynb)
*   YOLO 在 [Pytorch](https://github.com/ayooshkathuria/pytorch-yolo-v3) 实现
*   YOLO 在[韩妃雅](https://github.com/hojel/caffe-yolo-model)实施

在我们的工作中，我们使用了由 Alexey Bochkovskiy 维护的 darknet [分叉代码库。该存储库提供:](https://github.com/AlexeyAB/darknet)

*   更少的兼容性问题，因为它是由作者主动维护的。
*   许多其他人都在关注和使用这个存储库，这帮助我们找到了我们面临的大多数问题的答案，包括设置环境、培训期间的错误以及对一些参数的理解。
*   作者提供了一个有用的工具，可以为您的数据集(我们的例子)生成带标签的数据来训练模型。

# 标签数据

训练 YOLO 或任何其他 DL 模型的第一个要求是有标记的数据。对于我们的应用程序，我们使用 YOLO 来检测外围视觉中的对象，因此对象的视点是倾斜的。此外，在 80 类预训练模型中，我们的应用程序感兴趣的一些对象的类是不可用的。

为了满足我们的需求，我们生成了一个带标签的数据集来训练 YOLOv3 模型(用不同的类生成标签是一个漫长而无聊的过程)。对于训练数据，我们使用了由阿列克谢·博奇科夫斯基开发的名为[YOLO _ 马克](https://github.com/AlexeyAB/Yolo_mark)的标记工具。该存储库提供了一个优秀的 UI 接口，可以在不同类的对象上绘制边界框，并生成符合使用 darknet 存储库训练 YOLOv3 所需格式的带标签数据集。

# 设置 Conda 环境的挑战

我们利用 conda 为 darknet 建立了一个环境(也可以使用 docker)。我们继续康达，因为这是我们所知道的；我们可以很容易地在 Ubuntu 18.04 LTS 上维护所需的依赖关系。需要注意的一点是 OpenCV 的版本。我们面临一些 OpenCV 版本的问题，因此我们建议至少使用 OpenCV 版本 3.4.3。

我们在设置 GPU 支持的环境时遇到的另一个问题是`gcc`和`g++`版本。NVCC 要求`gcc`版本 5，而 LTS Ubuntu 18.04 的默认版本是版本 7。我们用`apt-get install`安装了`gcc`和`g++`版本 5。一旦安装完毕，我们必须在 CUDA 中指向 NVCC 来使用`gcc`和`g++`版本 5，这是通过在`/usr/local/cuda/bin/`文件夹中创建一个符号链接来完成的。

# 培训:问题和预防措施

在设置好我们的环境并生成必要的标记数据之后，很明显下一步是训练模型。我们使用了 Alexey Bochkovskiy 提供的 [*README.md*](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects) 文件中的 *How to Train* 部分，并在 yolov3.cfg 文件中对类的数量、批量大小、细分、最大批量、步骤和过滤器大小(这是必不可少的)进行了必要的更改。有了这些变化，我们就可以用我们的标注数据集来训练 YOLO 模型了。

## 分段故障错误

在第一次训练模型时，我们遇到了分割错误。首先想到的是我们正在耗尽 GPU 内存，但事实并非如此。Darknet 在利用 GPU 资源方面非常明智，不像其他一些框架。在深入研究 GitHub 的问题后，我们发现其他人也面临类似的问题。Github 问题标签上发布的建议之一是确保边界框标签尺寸不小于 0.01。我们发现许多标签属于这一类别，因此通过编程从训练数据中删除。使用上述方法解决了分割问题。

## yolov3.cfg 文件中不同参数的详细信息

中有许多可用的配置参数。cfg 文件，可以对其进行调整以优化整体训练。用户需要了解[这些参数](https://github.com/AlexeyAB/darknet/issues/279)中的每一个，以及它将如何影响整体训练。

## 控制台输出的详细信息

训练 YOLO 的时候，有一大堆[行话](https://github.com/AlexeyAB/darknet/issues/636)比如地域、avg IOU、class、obj、. 5R、. 75R 都印在控制台上。我们发现，这些输出中的一些帮助我们决定更早地终止训练，而不是等待模型在整个 50k 时代中得到训练。

# 在边缘设备中部署 YOLO 模型

有了你的 YOLO 模型，假设你想在一个 30 美元的小树莓上用 10 美元的相机运行它们。

![](img/32f07213c2f05807e1f06f1d89f11c14.png)

使用 YOLO 或类似但不同的移动 SSD(单次检测器)的部分优势是，它们通常很紧凑，非常适合移动或小型设备。事实证明你可以，这取决于你的设置，如果你想多花一点钱的话。首先，有了一个基本的 Raspberry Pi 和 OpenCV，[你就可以加载你的模型并在本地运行它](https://docs.opencv.org/master/da/d9d/tutorial_dnn_yolo.html)。在 YOLO-TINY 模型上，你可以期望得到约 200 毫秒每帧。在 YOLOv3 上，我们最好的成绩是每帧 12 秒。为了让它运行得那么快，你需要[在 RPi 上构建 OpenCV，并打开所有适当的加速和硬件标志](https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/)。简易安装没有成功。

现在，如果你需要一些实时的东西，你可以通过一个加速器来实现，比如 79 美元的英特尔[神经计算棒 2](https://software.intel.com/content/www/us/en/develop/hardware/neural-compute-stick.html) (NCU2)或 59 美元的珊瑚 USB 加速器 (EdgeTPU)。两者都会给你超过 20 FPS。NCS2 使用英特尔的 OpenVino 框架，从安装到运行时相当复杂，但一旦你通过所有的咒语，就可以加载 [YOLOv3。或者，我们的首选方法是 Coral Accelerator，它运行 Tensorflow Lite (TF-Lite)。YOLOv3 tiny](https://github.com/PINTO0309/OpenVINO-YoloV3) 的[流程是将 darknet `.weights`转换为 Keras 模型，然后将 Keras 模型转换为 TF-Lite，然后为 EdgeTPU 编译 TF-Lite。诀窍是确保 EdgeTPU 上的所有操作都可用。如果没有，检测将退回到 CPU 并停留在那里等待操作…你不希望这种情况发生。这听起来有点像瀑布，但是总的来说，比起 OpenVino，我们更喜欢工作和维护 TF 代码。如果您有多个加速器，您可以将它们结合使用，以获得更快的 FPS。虽然它们不会联合，但您可以在交错管道中为每个加速器线程化一个帧。](https://github.com/guichristmann/edge-tpu-tiny-yolo)

# 结论

总之，通过使用工程技巧/方法(无论你怎么称呼它)，我们能够为我们的应用程序训练 YOLOv3，并将其部署在加速的边缘设备上，并以实时性能运行。一旦你有了第一台便宜的计算机视觉设备，你就可以在任何地方嵌入它。我们希望这篇文章能帮助你们做同样的事情，更快更便宜地训练和部署 YOLO。