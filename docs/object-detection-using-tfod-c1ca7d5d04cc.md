# 使用 TFOD 的目标检测

> 原文：<https://medium.com/analytics-vidhya/object-detection-using-tfod-c1ca7d5d04cc?source=collection_archive---------10----------------------->

嗨，在这篇文章中，我们将看到如何使用 Tensorflow 对象检测(TFOD)来训练对象检测模型。大多数计算机视觉从业者都在努力设置 TFOD。因此，目标检测的每个步骤将被清楚地讨论。

![](img/b3e6b8aa87a1e7c873bc79038a4e08bd.png)

TFOD 目标检测

**Step-1:** 每个物体检测项目都是从数据(图像)开始的。我们需要通过各种技术收集相应项目的数据，如图像报废，谷歌图像等等！

**第二步**:在收集了所需的数据后，我们必须对图像进行标注。有一些网站提供收费的注释服务。然而，我们可以自己手动完成。我建议下载这个注释工具 **(** [**链接**](https://tzutalin.github.io/labelImg/) **)** 。根据您的操作系统下载最新版本并安装。

**步骤 3** :为每张图像的注释创建 XML 文件

**步骤-4** :我们必须为培训选择一个模型，此 [**链接**](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) 中提供了各种最先进的模型。从这个链接下载需要的模型并解压。

我们需要一些额外的文件来处理 TFOD。下载这两个文件并解压。**[**link 1**](https://github.com/tensorflow/models/tree/v1.13.0)[**link 2**](https://drive.google.com/file/d/12F5oGAuQg7qBM_267TCMt_rlorV-M7gf/view?usp=sharing)。**

****步骤 6** :创建一个文件夹，将上述下载的文件移动到其中。将下载的模型文件夹(例如:faster_rcnn_inception_v2_coco)移动到 models/research 文件夹。将 utils 文件夹中的内容移动到 models/research 文件夹中。utils 文件夹包含默认图像数据集，用您的数据集替换默认数据集。**

****步骤 7** :将 research/object _ detection/legacy 中的 train.py 复制到 research 文件夹中**

****步骤 8** :根据你下载的模型，从 object _ detection/samples/configs 中复制配置文件，粘贴到 research/training 文件夹中。**

****步骤 8** :将部署和网络文件夹从 research/slim 复制到 research 文件夹**

****Step-9** :打开 anaconda 提示符，将 anaconda 提示符的工作目录改为 research 文件夹**

****步骤 10** :使用以下命令创建一个虚拟环境**

> **conda create -n 环境名称 python=3.6**

****步骤 11** :使用以下命令激活虚拟环境**

> **conda 激活环境名称**

****第 12 步**:使用以下命令安装所需的软件包**

> **pip 安装枕头 lxml cy thon context lib 2 jupyter matplotlib pandas opencv-python tensor flow = = 1 . 14 . 0**

**上面的命令是针对 CPU 的，如果你想在 GPU 上训练使用下面的命令**

> **pip 安装枕头 lxml cy thon context lib 2 jupyter matplotlib pandas opencv-python tensor flow-GPU = = 1 . 14 . 0**

****步骤 13** :使用以下命令安装所需的 protobuf 包**

> **巨蟒原蟾蜍**

****步骤 14** :使用以下命令将 protobuf 文件转换成 py 文件**

> **protocol object _ detection/protos/*。proto — python_out=。**

****步骤-15** :使用以下命令在您的本地系统中安装对象检测**

> **python setup.py 安装**

****步骤-16** :使用以下命令将训练和测试数据的 XML 文件转换成 CSV 文件**

> **python xml_to_csv.py**

****步骤-17**:tensor folow 对象检测模型只接受 TFrecords 格式作为输入。将 CSV 文件和图像转换为 TFrecords。根据 generate_tfrecords.py 文件中的图像类更改图像类，并使用以下命令进行转换**

**对于训练数据:**

**python generate _ TF record . py-CSV _ input = images/train _ labels . CSV-image _ dir = images/train-output _ path = train . record**

**对于测试数据:**

**python generate _ TF record . py-CSV _ input = images/test _ labels . CSV-image _ dir = images/test-output _ path = test . record**

****步骤-18** :在配置文件中做如下修改**

**I:根据你的班级改变班级的数量**

**ii:将 fine_tune_checkpoint:更改为您的模型名称(例如“faster _ rcnn _ inception _ v2 _ coco/model . ckpt”)**

**三:根据你改变步数。我建议保持至少 50，000 步**

**iv:将 input_path 更改为:“train.record”(用于 train)**

**v:更改 label _ map _ path:“training/label map . Pb txt”(用于 train)**

**vi:更改 input _ path:“test . record”(用于测试)**

**七:更改 label _ map _ path:“training/label map . Pb txt”(用于测试)**

****步骤 19** :使用以下命令开始训练**

> **python train . py-logtostderr-train _ dir = training/-pipeline _ config _ path = training/YOUR _ model . config**

****步骤-20** :要停止中间的训练，按 ctrl+c。要再次恢复训练，运行上述命令。训练将从最后一个关卡重新开始。**

****第 22 步**:训练完成后，生成 ckpt 文件。这些 ckpt 文件包含训练模型的权重。**

****第 23 步**:为了预测，我们需要将 ckpt 文件转换成冻结的推理图文件。运行以下命令进行转换**

**python export _ inference _ graph . py-input _ type image _ tensor-pipeline _ CONFIG _ path training/YOUR CONFIG FILE-trained _ check point _ prefix training/YOUR MODEL。CKPT 文件—输出目录 my_model**

**(注意:根据您的配置更改大写字母的文本)**

**我想提一下训练时的一些重要事项。训练模型的最低系统要求是至少 16GB 的 ram，GPU，如英伟达 GTX 1080，采用最新一代的英特尔 CPU。**

**如果您不符合上述规范，您可以在云 GPU 服务器上训练模型，如 Paperspace 和 AWS。**

**确保您有足够的数据用于培训。如果数据非常少，比如每类 100 个图像，那么模型将是不健壮的。每门课至少要拍 500 张不同的照片。**

**我受 c17hawk 的启发写了这篇文章。【https://github.com/c17hawke】T5[T6](https://github.com/c17hawke)**

**如果您在培训过程中遇到任何问题，请点击此 [**链接**](https://c17hawke.github.io/tfod-setup/)**