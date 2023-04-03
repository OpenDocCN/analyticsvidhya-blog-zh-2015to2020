# 如何建立一个实时的面具检测器😷

> 原文：<https://medium.com/analytics-vidhya/real-time-face-mask-detector-8b484a5a6de0?source=collection_archive---------1----------------------->

## 了解如何利用 Tensorflow + Python 和 MobileNet SSD 开发您自己的自定义对象检测程序

我最近真的被人工智能领域迷住了，因为这个领域正在取得许多进步，以教会机器各种事情。因此，我决定建立自己的面具探测器，以确定何时有人戴着面具。我是通过利用 TensorFlow 对象检测 API 和 OpenCV 来训练由 300 多幅图像组成的 SSD(单次多盒检测器)神经网络来进行这些检测的。

![](img/9a175f96a3e05fe2e6c7d89fffba2a56.png)

左边的图像是一个戴着面具的人，右边的图像是我输入图像后，我的模型检测面具的结果，置信度为 94%。

在本文中，我将解释一切，从什么是对象检测和 SSD 算法如何工作，到如何实现这些基本原则来构建自己的自定义对象检测程序，无论是确定一只猫还是一只狗，或者在我的情况下，一张带面具的脸与一张不带面具的脸。请记住，您可以制作一个能够检测两个以上类别的模型，或者使用类似于 [COCO-SSD](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd) 的东西，它可以区分来自 80 个不同类别的对象。

所以，事不宜迟，让我们开始吧！

# 什么是物体检测？👀

从名字就可以看出，物体检测是当今计算机视觉中最突出的研究领域之一。这种技术主要在图像甚至视频流中识别甚至定位属于某些类别(狗、猫、人、建筑物、汽车)的物体。这不要与图像分类混淆，图像分类是计算机视觉的另一个分支，可以很容易地预测图像包含什么。从技术上讲，你可以使用这项技术来制作一个包含两类的屏蔽检测器:屏蔽和无屏蔽。

![](img/cf63c81569db6ed3c78e1b114f2d3951.png)

使用图像分类和图像定位的组合来标记狗(左)、猫(中)和狗(右)的对象检测模型(来源:[来自 Flickr 的 Raneko](https://www.flickr.com/photos/raneko/3051403531)

但是当你在同一个图像中同时有一个戴面具的人和一个没戴面具的人时，你会怎么做？你如何将它们区分开来？您可以使用多标签分类器来区分这两个对象，但您仍然不知道哪一个。这就是图像定位的用武之地，因为它识别图像中对象的位置，并返回识别图像中对象的边界框。

> 目标检测=多标签图像分类器+图像定位

你可以为你的模型制作你自己的类来检测，比如手表、手镯或面具，但是你需要数百张带标签的图像来训练你的模型。但是现在不要太担心它，因为我将在本文中对此进行介绍。

这项技术有许多现实应用，如视频监控或图像检索，面部检测是最常用的领域之一。它还可以用于红绿灯处的行人检测，以更好地引导交通，甚至帮助盲人行走。正如你所看到的，可能性是无限的，这使它更加令人兴奋！

# 什么是 SSD-Mobilenet？它是如何工作的？💭

单触发多盒检测器(SSD)是主要为实时目标检测而设计的单卷积神经网络。它学习预测物体周围的边界框位置。它只在一个镜头中对对象进行分类，这与 R-CNN 不同，R-CNN 使用区域提议网络来创建边界框，然后用于对对象进行分类。所以这个模型可以训练到结束，由 MobileNet 架构组成，后面还有很多其他卷积层。

![](img/c602283071ebd83b688d1b332abdf01f.png)

SSD 从单个图像中做出两个独立类别的预测，通过挑选有界对象的最高分数的类别，保留非有界对象的类别“0”(来源: [Jonathan Hui](https://jonathan-hui.medium.com/about) )

这大大加快了这个过程，但是这个模型在准确性方面受到了影响。SSD 模型包括一些改进，如多尺度功能和默认框，这使它能够匹配 R-CNN 的准确性，因为它使用较低分辨率的图像。如下图所示，在最好的情况下，它比 R-CNN 实现了超过 50 FPS 的实时处理速度，在某些情况下，甚至超过了 R-CNN 的精度。精确度以地图来衡量，地图代表预测的精确度。

![](img/1f6d2f42e8f8d12ea610f992903e524e.png)

SSD 与其他物体检测模型的性能比较(来源:[康乃尔大学](https://arxiv.org/pdf/1512.02325.pdf)

虽然这种模型速度快且准确，但这种框架的最大缺点是性能与对象大小成正比，这意味着检测较小的对象更加困难。这样做的原因是因为小物体有时可能缺乏神经网络深层进行检测的信息。但是有许多像数据增强这样的技术可以对图像进行裁剪、调整大小和处理。这就是为什么对输入图像使用更高的分辨率会提供更好的结果，因为图像中有更多的数据，更多的像素可供使用。

![](img/b7247a839f4ba475179f8b7d61289f58.png)

SSD Mobilenet 架构(来源:[康乃尔大学](https://arxiv.org/abs/1904.09021)

SSD 网络的大部分由主干网络控制，主干网络将是 MobileNet，这是另一种网络架构，由特殊类别的卷积神经模型组成，在参数数量和计算复杂性方面更加轻量级。此外，可以明确输入宽度和分辨率，从而控制卷积层的输入和输出通道的数量，同时还操纵图像分辨率(高度和宽度)。这与网络的延迟和准确性直接相关，后者取决于用户对模型的要求。

卷积层是一个应用于图像的矩阵，它对单个像素执行数学运算以产生新像素，然后将这些新像素作为下一层的输入，以此类推，直到到达网络的末端。最后一层是一个整数，它将图像输出转换成一个数字类预测，该预测对应于我们试图预测的一个对象。例如，如果“1”与一只猫相关联，则类“1”的预测将是一只猫，而“0”将是未知的。

![](img/1fc4eb4c4d686304987216fb11bdd994.png)

一个数据输入通过卷积层的例子，卷积层在应用复杂的数学运算后返回一个输出(来源: [Analytics India](https://analyticsindiamag.com/)

SSD 包括 6 层，进行 8732 次预测，并使用这些预测中的大部分来预测对象最终是什么。

将 MobileNet 集成到 SSD 框架中打开了无限可能性的大门，因为它不是资源密集型的，可以在低端设备上运行。在 R-CNN 的实时报道中，智能手机和笔记本电脑也在苦苦挣扎。

# 是时候开始建造我们的面具探测器了🔨

现在你已经知道了幕后的一切是如何工作的，让我们开始实际实现它。所有的代码都可以在我的 GitHub 知识库中找到，你可以在这里找到。

为了让这个工作，安装 Anaconda Python 3.7.4，用于 Windows T1、T2 Mac T3 或 T4 Linux T5。然后对于 windows，安装 [Visual Studio C++ 2015](https://go.microsoft.com/fwlink/?LinkId=691126) ，你将需要编译 Tensorflow。如果您的系统中有专用 GPU，请安装 [Cuda](https://developer.nvidia.com/cuda-10.1-download-archive-base) ，然后安装 [Cudnn](https://developer.nvidia.com/rdp/cudnn-download) 。

![](img/cf1843fa9a4325792e4c49157fb83a5b.png)

必需的依赖项，[左] Python，[中] Visual Studio C++，[右] Tensorflow

我们将使用 [Tensorflow 对象检测 API](https://github.com/tensorflow/models) ，这是一个利用深度学习网络来解决对象检测问题的框架。如果您有任何问题，请使用本指南作为参考，它对我帮助很大。

接下来，在您的目录中创建一个文件夹，在您选择的代码编辑器中打开它，并使用命令创建您的虚拟 python 环境:**conda create-n name _ of _ your _ choice pip python 3.7**

然后使用命令激活虚拟环境:**conda activate tensor flow**

![](img/5fd396bf25d02ad374e188a32fbbbcde.png)

创建您的虚拟环境[左]并激活环境[右]

然后使用命令克隆我的屏蔽检测器库: **git 克隆**[**https://github.com/ZakiRangwala/Mask-Detector.git**](https://github.com/ZakiRangwala/Mask-Detector.git)

之后，安装所有需要的依赖项和模块，首先使用**CD mask-detector/tensor flow**进入 mask detector 库，然后使用**pip install tensor flow = = 2 . 3 . 1**然后使用 **pip install opencv-python。**

使用命令验证您的安装:**python-c " import tensor flow as TF；print(TF . reduce _ sum(TF . random . normal([1000，1000]))"**

![](img/3f0c74ede248dd55f1dda230e4f2b2e9.png)

您的输出应该是这样的。

## 安装 Tensorflow 对象检测 API💻

我们将通过 **cd** 将 Tensorflow 模型花园中的 Tensorflow 对象检测 API 安装到 **Tensorflow-Models** 目录中，并使用命令**git clone**[**https://github.com/tensorflow/models**](https://github.com/tensorflow/models)**安装所有模型。现在，您将在 tensorflow-models 中看到一个名为 **models** 的新文件夹，其中包含大量不同的模型。**

**我们现在需要安装 Protobuf，谷歌的语言中立编译器，你可以从他们的[发布页面](https://github.com/protocolbuffers/protobuf/releases)获得，用于 [Linux](https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protoc-3.14.0-linux-x86_64.zip) 、 [Windows](https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protoc-3.14.0-win64.zip) 或 [Mac](https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protoc-3.14.0-osx-x86_64.zip) 。然后提取文件的内容并将其添加到您选择的目录中，例如**C:\ Program Files \ Google proto buf**，然后将此路径添加到您的环境变量和 **cd** 到 **models/research** 中并使用命令:**protoco object _ detection/protos/*。proto — python_out=。****

**然而，在我们安装对象检测所需的依赖项之前，我们需要使用 **pycocotools** 安装 COCO API，运行命令: **pip install cython** 和**pip install git+**[**https://github . com/philferriere/COCO API . git # subdirectory = PythonAPI**](https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI)**

**最后，**内的**CD**object _ detection/packages/tf2**并将脚本 **setup.py** 复制到 **Tensorflow\models\research 中。**然后运行命令 **python -m pip install。**安装所需的依赖项。**

**使用命令**python object _ detection/builders/model _ builder _ tf2 _ test . py**在**tensor flow \ models \ research**中测试您的安装**

**![](img/108e6b8f0af4d0e90d643f618805f5cc.png)**

**您的输出应该是这样的。**

## ****安装标签图像🌆****

**现在安装工具， [LabelImg](https://github.com/tzutalin/labelImg.git) 来帮助注释我们的图像，通过 **cd** 到**标签-图像**，并使用命令**git clone**[**https://github.com/tzutalin/labelImg.git**](https://github.com/tzutalin/labelImg.git)**

**目录中应该会创建一个新文件夹。 **Cd** 进入 **labelImg** 并使用命令 **conda install pyqt=5** 和**conda install-c anaconda lxml****

**之后，剩下的就是设置二进制文件，这可以使用命令**pyr cc5-o resources . py resources . qrc**来完成**

**最后，导航到 **LabelImg** 目录，将 **resources.py** 和 **resources.qrc** 文件移动到 **lib** 文件夹中，如下所示**

**![](img/c05ea9d41a85b80ff5e4723e650a9c4c.png)**

**将资源文件移动到 libs 文件夹中以完成程序的设置。**

## ****获取您的图像数据集📷****

**所有的艰苦工作都结束了，是时候开始有趣的事情了。是时候收集一些图片了。如果你非常热心，你可以尝试自己点击它们，或者从网上任何地方获取数据集。如果你选择从网上获取数据集，我建议[使用 Kaggle](https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset) 的这个数据集。**

**然而，如果你选择自己拍照，请确保在明亮的光线下拍摄，从不同的角度拍摄，建议不同的人谈论相同的#带面具的照片到#不带面具的照片。要从模型中获得更高的置信度，请确保数据集至少包含 300 多张图像。**

**![](img/dd316839e335d347ca7d95a296646bd9.png)**

**我的图片集由来自网络的图片和一些我自己拍摄的图片组成。**

**一旦有了图像数据集，使用命令 **cd** 和 **labelImg** 导航到 **labelImg** 目录，并使用命令 **python labelImg.py** 运行图形用户界面(GUI)**

**![](img/9e5300a84686fec5c353b28e093074b3.png)**

**您的 LabelImg GUI 界面应该如下所示。**

**然后点击**打开目录**按钮，导航到保存图片的文件夹，然后点击**保存目录**按钮将图片保存到同一个文件夹中。**

**请确保您的注释格式为 PascalVOC，这是一种用于图像分类和对象检测的数据格式，与 COCO 不同，它会为每个图像创建一个新的注释。**

**然后开始，通过按下' **W** '标记所有图像，然后在蒙版周围创建一个边界框并相应地标记它。这两个职业分别是**面具**和**游牧**。然后点击“ **D** 前进，或“ **S** 后退。这可能是一些繁琐的工作，需要很长时间，但模型必须根据图像如何标记来训练。**

**![](img/7db77cf3c90bf21505deea1cd1323577.png)**

**根据图像，用类“Mask”或“NoMask”标记图像；一定要精确。**

**一旦所有的图像都被标记，你的目录应该有两倍的文件，因为每个图像现在都有一个注释。xml)文件。现在将图像相应地移动到**训练**和**测试**文件夹中，该文件夹可以在**tensor flow/workspace/images 中找到。**每个文件夹里一定要放一些带口罩和不带口罩的图片，多放一点用于训练。尝试 60/40 的比率，因为模型将基于训练数据的准确性进行评估。这被称为**监督学习，**因为算法被给定一个要从中学习的带标签的数据集。**

**![](img/e5e28c795e85839348cf92f666da5bde.png)**

**我的测试和培训文件夹中的标签数据集**

## **制作我们的 Python 脚本📄**

**现在在 **Mask-Detector** 中创建一个名为 **detect.py，**的新文件，让我们开始导入将要使用的库。**

**之后，让我们设置我们的环境路径，以便在整个程序中更容易引用它们。**

> **注意:你已经可以在**tensor flow/workspace/annotations**文件夹和**tensor flow/workspace/pre-trained-models**文件夹中找到包含我的训练数据的文件，以防你懒得标记你的图像和训练你自己的模型。**

**接下来您要做的是创建一个函数，该函数将构造一个标签映射，其中包含与每个图像注释相关联的类名。**

**通过调用以下方法运行此函数: **construct_label_map()** 在**tensor flow/workspace/annotations**中找到一个名为 label_map.pbtxt 的包含您的类的新文件，在我们的示例中，该文件将是“ **Mask** 和“ **NoMask** ”，假设您使用 **LabelImg** 工具对图像进行了注释。**

**现在你要添加一个函数，可以合并所有的 PascalVOC。XML 文件转换为一个 CSV 文件，用于训练和测试数据。**

**你可以使用 convert()命令调用这个函数，应该会在**tensor flow/workspace/annotations**下找到两个新文件，分别名为 **testlabels.csv** 和 **trainlabels.csv****

**现在是时候创建用于训练模型的 TF 记录(Tensorflow 记录)了。**

**为此，请使用以下命令:**

```
**# Train Record** python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/train -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/train.record**# Test Record** python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/test -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/test.record
```

**我们差不多完成了；我们现在需要做的就是从 Tensorflow 模型中下载一个预先训练好的模型，你可以在这里找到。存储库应该看起来像这样。**

**![](img/2e6d250c98bc0ab809b66016f58bf5cf.png)**

**有多种型号可供选择。一定要记住速度和准确性。**

**您不需要下载预先训练的模型，因为我已经在我的存储库中包含了一个。但是，如果您下载了一个预训练的模型，请确保将 **pipeline.config** 文件复制到**tensor flow/workspace/models**目录和一个名为 **my_ssd_mobnet** 的新文件夹中，因为这是我们将进行训练的存储库。**

**现在，在我们训练之前，让我们修改我们的**管道。config** 文件如下**

**确保将“num_classes”保持为 2，因为我们只使用掩码 vs 训练数据集，而不使用掩码；您可以根据您的计算能力来改变“batch_size”变量；推荐 32。**

**现在我们准备训练我们的模型；我们要做的就是运行命令。**

```
**# Train Model**
python Tensorflow/tensorflow-models/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=5000
```

> **注意:训练步数(num_train_steps)当前设置为 5000，但可以增加以提高精度，但训练时间会更长**

**你的模型通常需要 3 到 5 个小时来训练，所以我倾向于通宵运行更长时间来提高效率。**

**![](img/50e5cbff814b3802bce4429654f10092.png)**

**您的输出应该看起来像这样，并且您应该在**tensor flow/workspace/models**文件夹中有新的检查点。**

## **对🟥进行实时预测**

**一旦您的模型完成训练，您就可以开始进行检测了。我们首先要像这样加载我们的模型:**

**确定哪里写着“**ckpt . restore(OS . PATH . join(check point _ PATH，' ckpt-9 ')”。expect _ partial()**"**你在显示' **ckpt-9** '的地方输入你最新的检查点，你可以在**tensor flow/workspace/models**中找到它们****

## ****检测功能🔍****

****现在是我们期待已久的时候了，将我们所有的努力付诸实践。我们需要创建一个函数，它接收图像，加载模型，通过解析到我们的神经网络来预测图像中的内容，并返回 1 或 2 类，其中 1 是“Mask”，2 是“NoMask”。****

****现在，让我们检查一下我们的代码是否可以通过创建另一个函数来将我们的图像转换为张量，张量是一个由我们图像的像素组成的矩阵，当输入到模型中时，将由一个卷积层通过复杂的数学运算进行处理，并作为下一层的输入返回，直到它到达将返回一个类的最后一层。如果类是 0，那么对象是未知的。****

****![](img/aea1b06817b551e15f47b6ce5bf3224f.png)****

****由许多卷积层组成的 SSD 模型架构(来源: [Lilian Weng](http://lilianweng.github.io)****

****一旦我们运行检测功能，我们应该得到的结果给我们检测的数量，对象的位置，置信度得分和检测类本身。然后，我们可以根据该函数返回的坐标绘制标签和边界框，并使用 OpenCV 读取图像，显示检测结果。****

****您可以随时编辑 **"min_score_thresh"** 参数，根据您的置信度显示检测结果，这取决于您的模型的精确度。****

****您尝试通过使用**check(' tensor flow/workspace/images/check/test _ case _ one . jpg ')**运行模型并查看在 **results** 文件夹中弹出的结果来检查模型是否工作。我已经补充了几个例子。随时补充更多！****

****![](img/11b65c7e4939ebcc13af68c8af6eff0d.png)****

****遮罩检测演示->将图像输入模型。****

****正如你所看到的，当输入一张图片时，这个模型工作得非常完美，但是当要求实时检测时，它的表现如何呢？让我们来了解一下！****

## ****实时预测🧔****

****为了进行预测，该函数将图像输入到模型中，该模型返回各种元素，如类、置信度得分甚至边界框。****

****当实时检测物体时，该过程相对相同。尽管如此，我们还是建立了一个视频流，将每一帧输入到图像中，进行检测，然后实时标记视频。****

****为此，我们使用 OpenCV，这是一个开源库；固态硬盘在实时检测方面大放异彩，因为它非常轻便和快速，使这成为可能，因为它提供高达 50 FPS 的速度，这对于了解引擎盖下发生的一切来说是非常惊人的。****

****现在，随着实时预测的成功，我们都完成了！您已经成功地构建了自己的自定义对象检测模型，并了解了幕后的工作原理。如果您想获得本文中概述的所有方法和函数的源代码，您可以在我的 GitHub 资源库中找到，您可以找到下面的链接:****

****[](https://github.com/ZakiRangwala/Mask-Detector) [## ZakiRangwala/面罩检测器

### 构建了一个简单的轻量级对象检测程序，可以确定某人是否戴着面具…

github.com](https://github.com/ZakiRangwala/Mask-Detector) 

## 限制😢

正如您所知，该应用程序有一些限制，其中之一是模型性能如何与对象大小和图像质量成正比，因此请确保您的数据集有良好的高清图像，并准确无误地标注它们。

此外，尽管它号称运行速度为 50 FPS，但实时预测视频流似乎有点慢，这可能是因为运行该模型的设备的规格。为了提高性能，请尝试输入较小的视频帧，并保持在明亮的环境中！

## 后续步骤👣

虽然这个项目超出了我的预期，但我发现它在现实世界中非常有用。但让它更加独特和有用的是增加新的类别来确定一个人是否受到保护，根据面部标志来判断，面具覆盖了哪些类别。这将有助于确保人们遵守世界各地当局制定的法律和法规，以确保每个人的安全。

我还可以向我的数据集中添加更多的图像，以使我的模型更加准确，并修改一些参数，以使实时检测更加平滑和高效。**** 

****如果你喜欢这篇文章，并且它帮助了你，请留下掌声，如果你有任何问题，你可以在下面留下评论，或者更好的是，发电子邮件到[zakirangwala@gmail.com](mailto:zakirangwala@gmail.com)给我。****

****如果你想更多地了解我和我的工作，请访问我在 zakirangwala.com[的网站](http://zakirangwala.com/)****