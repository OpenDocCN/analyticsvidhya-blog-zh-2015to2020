# YOLOv2 和 YOLO9000 的深度探索

> 原文：<https://medium.com/analytics-vidhya/deep-dive-on-yolov2-and-yolo9000-2eba212dcf8a?source=collection_archive---------8----------------------->

## 更好、更快、更强大的 YOLO 对象检测模型

2015 年推出的 **YOLO ( *你只看一次)*** 物体检测模型是当时同类模型中最先进的，拥有惊人的实时预测速度( ***快速 YOLO*** 可以以每秒 155 帧的速度预测物体)和比以前的模型更高的准确性(YOLO 在检测物体方面比 *DPM* 和*泽勒-费尔等模型好得多)*

虽然 YOLO 由于比以前的模型更快的探测速度和更高的精确度表现得相当好，但它也有一些缺点。

相对于一流的快速 R-CNN，YOLO 造成了大量的本地化错误(8.6%对 19%)

![](img/07cfba7ceaca3b81c5acbab39a604364.png)

YOLO 相对于快速 R-CNN 的误差分析([来源](https://arxiv.org/pdf/1506.02640.pdf))

此外，与使用基于区域提议的方法的模型相比，它的**召回率相对较低(例如*更快的 R-CNN***

YOLO 的创造者决定解决这两个主要问题，同时确保检测精度保持不变。

# YOLOv2 和 YOLO9000 简介

雷德蒙和法尔哈迪开发了 YOLO 的改进版本，命名为***【yolov 2】***，它有几个 YOLO 没有的功能，比如`multi-scale training`(对不同分辨率的图像进行训练)，从而在速度和准确性之间提供了一个简单的权衡。

他们还提出了一种新的方法，通过这种方法，他们可以同时在对象检测和图像分类数据集上训练模型，从而弥合这两种数据集之间的差距(我们将在下面讨论)。

一个巴掌拍不响，在同一篇论文中，作者还谈到了另一个模型， ***YOLO9000*** ，一个可以`detect over 9000 object categories`的实时物体检测模型！

YOLO9000 在对象检测和分类数据集上联合训练，可以预测甚至没有标记检测数据的对象类的检测，该模型在 *ImageNet 检测任务*上提供了良好的性能。

听起来很有趣？下面就让我们来深潜一下。

# 比 YOLO 好

正如我们前面所看到的，与快速 R-CNN 相比，YOLO 产生了大量的定位错误，并且当与基于区域提议的方法匹配时，其召回率较低。在 YOLOv2 中，主要关注的是修复这些错误，同时保持检测速度和精度。

作者利用了从实地知识中学到的各种概念以及一些新技术，得出了一个比 YOLO 更好的模型。那么，那些技术是什么？

## 1.批量标准化(BN)

简单地说，BN 为神经网络的每一层提供了独立于其他层进行学习的能力。

它通过减去该批的平均值并除以该批的标准偏差来标准化先前激活层的输出，该输出乘以标准偏差参数(*γ*)，然后添加平均值参数(*β*)。

![](img/e544015457692ddbe5629c6579b7bb0a.png)

批量标准化过程([来源](https://arxiv.org/pdf/1502.03167v3.pdf))

这一过程会导致收敛发生显著变化，并且无需其他形式的正则化(如*丢弃*)，而不会过度拟合模型。

✅当 BN 被添加到 YOLO 中的所有卷积层时，观察到 mAP ( *平均精度* ) 中有
**2%的改善。**

## 2.高分辨率分类器

YOLO 以 224 x 224 的图像大小训练分类网络，并将其增加到 448 x 448 用于检测过程。
这导致网络在短时间内切换到学习对象检测，并将其自身调整到新的分辨率。

在 YOLOv2 中，分类网络首先在 ImageNet 上以
224 x 224 分辨率进行训练，然后在 ImageNet 数据集上以全 448 x 448 分辨率进行 10 个时期的微调，给网络时间来调整其过滤器，以更好地处理更高分辨率的输入。
最终网络在检测时进行微调。

✅这个开发高分辨率分类网络的过程导致**大约 4%的地图增加**。

![](img/1240798fd33176cfe987c41410506439.png)

高分辨率分类器([来源](https://docs.google.com/presentation/d/14qBAiyhMOFl_wZW4dA1CkixgXwf0zKGbpw_0oHK8yEM/edit#slide=id.g23eff386b7_0_24)

## 3.带锚盒的卷积

YOLO 直接使用模型末端的全连接(FC)层预测边界框坐标，该层位于所有卷积层(特征提取器)之上。

***更快 R-CNN*** 有一个更好的技术——使用精选的*先验*(锚盒)，它预测这些盒子在特征图中每个位置的偏移量和置信度。这简化了问题，使网络更容易学习。

我们可以创建 5 个锚盒，如下所示。

![](img/18043f096c766a5c838b0fb1c8a56a9d.png)

锚箱

我们不会预测 5 个任意的边界框，但是会预测上面锚框的偏移量。这样做可以保持预测的多样性。模型的训练会更加稳定。

![](img/5a804db45363580e12e43d74936800e6.png)

保持预测的多样性([来源](https://miro.medium.com/max/864/1*CGWTPTY0sfvQoxsS0X6VFg.jpeg)

相反，如果我们预测每个对象的任意边界框(像 YOLO 那样)，训练将是不稳定的，因为我们的猜测可能对一些对象有效，但对另一些对象无效。

![](img/c480e0bcfe41b35c8e2494ce1b18a934.png)

任意的包围盒预测意味着不稳定的训练([来源](https://miro.medium.com/max/864/1*7iwTsezrn-tSndx96twprA.jpeg))

因此，我们对 YOLO 所做的更改如下所示—

1.  删除预测边界框的 FC 层，并使用锚框代替

![](img/65f066bba1b0dc508e6a45f1bd446d13.png)

移除 FC 层([源](https://miro.medium.com/max/875/1*tavjieD0Bum_uX-svYUTKA.jpeg))

2.使用 416 x 416 的输入尺寸，而不是 448 x 448。物体(尤其是大的)通常占据图像的中心，所以为了预测这些物体，中心的单个位置比附近的 4 个盒子要好。这一步意味着我们的特征地图将有一个 13 x 13 的大小，因为 YOLO 的卷积层以 32 的因子对图像进行下采样。

因此，在我们的特征地图上，奇数位置比偶数位置更好。

![](img/84e6a49f0f6f0235abac79ce2c231d23.png)

特征地图上位置的奇数([来源](https://miro.medium.com/max/875/1*89qezEeLKJLpD8_fM_H4qQ.jpeg))

3.将类别预测机制与空间位置分离，并将其转移到边界框级别。

对于每个边界框，我们现在有边界框的`25 parameters` — 4、1 个框置信度得分(在本文中称为*客观性*)和 20 个类别概率(针对 VOC 数据集中的 20 个类别)。

我们每个单元有 5 个边界框，因此每个单元将有 25×5，即 125 个参数。
类似于 YOLO，对象预测机制预测基础事实和建议框的 IOU ( *交集/并集*)

![](img/9c804793ba958e827375b433784318fd.png)

每个边界框 25 个参数，每个单元 125 个参数([源](https://miro.medium.com/max/875/1*UsqjfoW3sLkmyXKQ0Hyo8A.png))

我们还删除了一个池层，使模型
的空间输出为 13 x 13，而之前是 7 x 7。

✅Using 锚盒**将地图的准确率从 69.5 降低到 69.2，但导致召回率显著提高，从 81%提高到 88%** 。

## 4.维度群

这种方法建议在训练集边界框上运行`**K-means clustering**`来自动找到好的先验，而不是从手动选取的锚框尺寸开始。

我们想要独立于边界框大小的先验，所以我们使用下面的距离度量。

```
***d(box, centroid) = 1 - IOU(box, centroid)***
```

下图显示了对不同的 *K* 值运行 K-means 并绘制质心最近的平均 IOU 的结果。作者指出，K=5 在模型复杂性和高召回率之间提供了一个很好的平衡。

显示了 VOC 和 COCO 数据集的相对质心。两组形心的一个共同点是，与短而宽的盒子相比，更喜欢高而细的盒子。COCO 的尺寸变化比 VOC 大。

![](img/1d0518c7af31dff207addd0e7dbb2ec0.png)

聚类数与平均值。VOC 和 COCO 的 IOU 和相对质心([来源](https://arxiv.org/pdf/1612.08242.pdf)

✅下表将平均 IOU 与聚类策略和预定义锚盒的最接近先验进行了比较。**使用 K-means 生成包围盒，用更好的表示初始化模型，使任务更容易学习**。

![](img/7aab776379adedd8f2bd963acdcabbf0.png)

2007 年 VOC 上最接近的箱子的平均 IOU([来源](https://arxiv.org/pdf/1612.08242.pdf))

## 5.直接位置预测

直接预测边界框的 *(x，y)* 位置导致的问题是模型不稳定。区域建议网络所做的是预测 tₓ和 tᵧ的值，并且(x，y)坐标计算如下

```
x = (tₓ * wₐ) - xₐ
y = (tᵧ * hₐ) - yₐ
```

由于偏移在这里不受约束，任何锚定框都可以在图像中的任何点结束，而不管它是在什么位置预测的-模型需要很长时间来稳定以预测合理的偏移。

另一种方法是预测相对于格网单元位置的位置坐标，就像在 YOLO 一样。该网络预测每个边界框的 5 个坐标-tₓ、tᵧ、tཡ、tₕ和 tₒ，并应用逻辑函数将这些坐标值约束为[0，1]

![](img/bf3a4e615909b5ef641c9f6670ce91dc.png)

网络坐标预测([来源](https://miro.medium.com/max/875/1*38-Tdx-wQA7c3TX5hdnwpw.jpeg))

![](img/6bd89a374b2046411baa18996255ea32.png)

蓝色框是预测的边界框，虚线矩形是锚点([源](https://arxiv.org/pdf/1612.08242.pdf))

这使得网络训练更加稳定。

✅维度聚类和直接预测边界框中心位置的技术**比带有锚框的版本**提高了约 5%的 YOLO。

## 6.精细特征

在 13 x 13 的特征图上预测检测对于大的对象很有用，但是对于定位较小的对象，更细粒度的特征可能被证明是有用的。

不是像在*更快的 R-CNN* 和 *SSD* (单次检测器)中那样在各种特征地图上运行区域提议网络，修改后的 YOLO 版本增加了一个 ***穿透层*** ，它以 26 x 26 的分辨率从早期层带来特征。

该图层通过将相邻要素堆叠到不同的通道来连接高分辨率要素和低分辨率要素，将
26 x 26 x 512 要素地图转换为 13 x 13 x 2048 要素地图，该地图再次与原始的 13 x 13 x 1024 要素连接。

✅这个扩展的特征地图，具有 13 x 13 x 3072 的特征地图，提供了对细粒度特征的访问，**将 YOLO 车型
改进了 1%** 。

![](img/bd7eaf7e3930f5f4dda2665e1c7b6cf3.png)

扩展特征图→细粒度特征([来源](https://miro.medium.com/max/875/1*RuW-SCIML8SHc5_PrIE9-g.jpeg))

## 7.多尺度训练

YOLO 最初使用 448 x 448 的输入分辨率，通过添加锚定框，分辨率变为 416 x 416。但由于修改后的网络没有 FC 层(只有卷积层和池层)，因此可以动态调整大小。

该网络每隔几个迭代就被修改——每 10 批，它随机选择一个新的图像尺寸大小。当模型以因子 32 进行缩减采样时，网络会学习以下分辨率:{320 x 320，352 x 352，… 608 x 608}。

事实上，同一个网络可以预测不同分辨率的检测，这意味着我们现在可以微调速度和准确性的大小。

*   YOLOv2 是一款相当好且精确的检测器，非常适合较小的 GPU、高帧率视频或多个视频流。
    在 288 x 288，它以 91 FPS 运行，并给出 **69.0 mAP** ，在精度上与快速 R-CNN (70.0 mAP)相当，但在速度上比后者好得多(快速 R-CNN 的速度仅为 0.5 FPS)*VOC 2007 的分数*
*   `**At high resolutions**`，YOLOv2 是一款**最先进的检测器，在 VOC2007 数据集上给出了 78.6 的 mAP** ，它的准确性是本文比较的模型中最高的，同时仍具有 40 FPS 的实时速度。

![](img/969ebd68425ac708a7341f4a5e0f0114.png)

VOC 2007 数据集上的检测框架([来源](https://arxiv.org/pdf/1612.08242.pdf))

![](img/38dd4ee7fd60e9c6c2645fe79372e354.png)

VOC 2007 的准确性和速度([来源](https://arxiv.org/pdf/1612.08242.pdf))

## 进一步的实验

在`**VOC 2012 detection dataset**`上，YOLOv2 获得 **73.4 图**，比原来的 YOLO 好，快 R-CNN，更快 R-CNN，SSD 300。

这一性能与 **SSD512** (74.9 图)和 **ResNet** (73.8 图)的性能相当，但 YOLOv2 是其中最快的(2-10 倍)。

![](img/73bc1e461b4482a9b0477a83b470005a.png)

VOC 2012 *(测试)*检测结果([来源](https://arxiv.org/pdf/1612.08242.pdf))

在`**COCO 2015 *test-dev* dataset**`上，使用 VOC 指标，IOU = 0.5，YOLOv2 得到 **44.0 mAP** ，与 SSD512 和更快的 R-CNN 的性能相当。

![](img/15d50e27efc499b5483c4d6785f85ea3.png)

COCO 2015(测试开发)检测结果([来源](https://arxiv.org/pdf/1612.08242.pdf))

综上所述，下表列出了与 YOLO 相比，YOLOv2 中最能提高 mAP 的设计决策，除了两个例外— **使用带锚盒的全卷积网络&使用新网络** (Darknet19，我们稍后会介绍)。

锚盒方法将召回率从 81%提高到 88%，但将 mAP 从 69.5 降低到 69.2，而新网络将计算量降低了 33%。

![](img/c250799c9c81190a0d93a88dfc96621e.png)

从 YOLO 到约洛夫 2 的道路([来源](https://arxiv.org/pdf/1612.08242.pdf))

唷！为了让模型更好，我们做了很多改动，接下来让我们来看看 YOLOv2 是如何比 YOLO 更快的。

# 比 YOLO 还快

**VGG-16** 是大多数检测架构使用的分类特征提取器([来源](https://arxiv.org/pdf/1409.1556.pdf))；虽然它提供了最先进的性能，但它不必要的复杂(它在 224 x 224 的单幅图像上使用了大约 310 亿次浮点运算，因此需要强大的计算能力)。

在`**Googlenet architecture**`的基础上，YOLOv2 使用了一个名为 ***Darknet19*** 的定制框架(它有 19 个卷积层，5 个最大池层)。

✅It 比 VGG-16 更好，因为—

*   它只需要 55.8 亿次浮点运算，相比之下，VGG-16 需要 310 亿次浮点运算，因此速度更快
*   虽然它使用相对较低的浮点运算，但它仍然获得了更高的前 5 名准确率，在 ImageNet 上达到 91.2%，而 VGG-16 的准确率为 90%(YOLO 使用了 85.2 亿次浮点运算，前 5 名准确率为 88%)
*   他们使用当时新颖的 ***全球平均池*** 进行预测，与 VGG 的最大池层数进行比较-16
*   ***批量归一化*** 用于稳定训练和模型正则化，也使收敛更快
*   在 3×3 层之间使用 1×1 卷积滤波器来压缩特征([来源](https://arxiv.org/pdf/1312.4400.pdf)

![](img/5919b30f998d531d375fdff537129c66.png)

Darknet19 在准确性、速度和#参数方面与其他型号进行了比较([来源](https://docs.google.com/presentation/d/14qBAiyhMOFl_wZW4dA1CkixgXwf0zKGbpw_0oHK8yEM/edit#slide=id.g23eff386b7_0_69))

`**Global average pooling**`层用于通过减少网络中的参数数量来减少过拟合。

这是一种降维，将维度为
***(h x w x d)*** 的张量转换为维度为 ***(1 x 1 x d)***

![](img/f39f34319f0ea08d1b89e478b524c97d.png)

全球平均池([来源](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/))

下面给出的是 Darknet19 的模型结构。

![](img/3c3808eabe1b7996d26a6b81648a73be.png)

Darknet19 模型结构([来源](https://miro.medium.com/max/580/1*8FiQUakp9i4MneU4VXk4Ww.png))

上图中删除的部分被替换为三个 3 x 3 的 conv 层(每个层有 1024 个滤镜)，然后是一个 1 x 1 的 conv 层，其输出数量为检测所需的数量，最终输出为 7 x 7 x 125。

![](img/c323847930d78b97b794b2720510ce18.png)

带暗网 19 的 yolov 2([来源](https://media.geeksforgeeks.org/wp-content/uploads/20200401004021/darknet-19-simplified.jpg))

YOLOv2 论文深入解释了作者为获得最佳性能而使用的超参数(及其值)。我不会说得太详细，让你感到厌烦，但只要你感兴趣，就知道一切都在那里。

# 比 YOLO 强

最后，让我们看看 YOLOv2 和 YOLO9000 中的新特性，这些特性使它们比以前的版本更好。

与图像分类数据集相比，对象检测数据集的一个特定特征是，前者具有有限的标记示例。Vanilla `**detection datasets**`包含 10 -10⁵范围内的图像，带有几十到几百个标签。(例如， **COCO 数据集**有 330，000 张图像，仅有 80 个对象类别)

`**Classification datasets**`是巨大的，因为它们包含数百万张图像和数千到数万个类别(例如 **ImageNet** 包含大约 1400 万张图像和大约 22000 个不同的对象类别)。

因为标记用于检测的图像(预测边界框、执行非最大抑制等。)与标记图像用于分类(简单的*标记*)相比是昂贵的，检测数据集难以缩放到分类数据集的水平。

如果我们可以将这两种类型的数据集结合在一起，并构建我们的网络，使其能够执行分类和检测，甚至更好地检测图像中的对象，这些对象具有它尚未看到任何标记检测数据的类，会怎么样？

## 分级分类—结合检测和分类数据集

事实证明，[论文](https://arxiv.org/pdf/1612.08242.pdf)提出了我们上面提出的同一个问题的解决方案。

我们可以在训练过程中组合来自这两种类型数据集的图像，但我们有一个问题——虽然检测数据集有简单的对象标签，如*【瓶子】**【微波】**【公共汽车】*，但分类数据集更多样，分类更详细(我粗略地看了一下，在这里发现了 15 种不同种类的蛇，如*【蟒蛇】**【环颈蛇】*和*那么我们如何合并两个数据集中的标签呢？*

![](img/3de221a2442a3c31b1458ecd7c8e05bf.png)

我们如何合并来自两种数据集类型的类？([来源](https://docs.google.com/presentation/d/14qBAiyhMOFl_wZW4dA1CkixgXwf0zKGbpw_0oHK8yEM/edit#slide=id.g23f389e290_1_78))

通常情况下，我们会使用 ***softmax 分类*** ，但它是`not an option here`，因为这些类不是互斥的。我的意思是，softmax 只有在没有类相互关联时才起作用，因此我们可以很容易地得到一个对象属于一个类的概率，并且所有的概率将
加 1。
在我们的例子中，*【狼蛛】**【园蛛】**【狼蛛】*相互关联，举个例子。

![](img/a2e7670152c8d27f86476225c30914d0.png)

不能使用 softmax 分类([来源](https://docs.google.com/presentation/d/14qBAiyhMOFl_wZW4dA1CkixgXwf0zKGbpw_0oHK8yEM/edit#slide=id.g23f389e290_1_46))

相反，我们可以从零开始，忘记所有关于标签相互关联的知识。

ImageNet 标签提取自 ***WordNet* ，这是一个在线词汇数据库**，它构建了概念以及它们之间的关系。

作为有向图的结构，不同的对象类可以属于同一个父类(下义词)，该父类可以是各种类型的对象，等等。

比如 WordNet 中的*【笑鸮】**【大灰鸮】*都是“猫头鹰”的下位词，是“鸟”的一种，是“哺乳动物”的一种等等。

为了建造这棵树，我们遵循这些步骤—

1.  观察 ImageNet 中的视觉名词，并沿着它们的路径通过 WordNet 图到达根节点(*“物理对象”*)
2.  因为许多同义词集只有一条路径，所以我们将它们全部添加到我们的树中
3.  对于剩下的概念，我们迭代地检查它们，并添加那些尽可能少地增长树的路径(如果一个概念有 2 条路径到根，一条有 5 条边，另一条有 3 条边，我们将选择后者)

![](img/b738671d8c4557925144f1171c68b111.png)

WordNet 的结构可能很难理解

我们因此获得了**单词树，一个视觉概念的层次模型**。

不是在 1000 个类(ImageNet 的)上执行 softmax，单词树将具有对应于原始标签的 1000 个叶节点加上用于它们的父类的 369 个节点。

在每一个节点，我们预测一个同素集*的每个下位词的概率，给定那个同素集*。
说我想求节点*“梗”的概率。* 为此，我们将预测以下概率(假设对象是一只*梗*等等，一只*诺福克梗*的概率)

```
Pr(Norfolk terrier | terrier)
Pr(Airedale terrier | terrier)
Pr(Sealyham terrier | terrier)
Pr(Lakeland terrier | terrier)
...
```

如果我们想计算某个节点的绝对概率，我们沿着树的路径到根节点，然后取所有条件概率的乘积。
以下示例用于计算图片属于*艾尔代尔梗*的概率。

```
Pr(Airedale terrier) = Pr(Airedale terrier | terrier)
                     * Pr(terrier | hunting dog)
                     * ....
                     * Pr(mammal | animal)
                     * Pr(animal | physical object)
```

我们总是假设图片中包含一个物体，所以 *Pr(实物)=* 1。

![](img/e1d26a406b5bc39f04a3d9188366f8dc.png)

Wordnet 中的条件概率([来源](https://docs.google.com/presentation/d/14qBAiyhMOFl_wZW4dA1CkixgXwf0zKGbpw_0oHK8yEM/edit#slide=id.g1f9fb98e4b_0_147))

为了计算条件概率，Darknet19 预测 1369 个值的向量，并且计算作为相同概念的下位词的所有同义词集的 softmax。

![](img/41e955fbe0a7c0a06e7619ef69fd9d9d.png)

WordNet 中的 soft max([来源](https://docs.google.com/presentation/d/14qBAiyhMOFl_wZW4dA1CkixgXwf0zKGbpw_0oHK8yEM/edit#slide=id.g1f9fb98e4b_0_103))

如你所见，synset *“身体”(头、头发、静脉、嘴*等的所有下位词。)归入一个 softmax 计算。

多亏了 WordNet，DarkNet19 在 ImageNet 上实现了 **90.4%的 top-5 准确率，保持了所有参数和之前一样。**

下面是一个代表性的图片，展示了如何使用 WordNet 结构合并 ImageNet 和 COCO 数据集的类。

![](img/0cb0d21c0286f8067871bab9673f90fc.png)

使用 WordNet 组合数据集([来源](https://arxiv.org/pdf/1612.08242.pdf))

## YOLO9000 —同时分类和检测

现在，我们有一种技术可以将对象检测和图像分类数据集结合在一起，以获得一个巨大的数据集，最好的部分是，我们可以训练我们的模型来执行分类和检测。

使用上面的技术，我们组合 COCO 数据集(用于检测)和来自完整 ImageNet 数据集(用于分类)的前 9000 个类—组合的数据集包含`**9418 classes**`。

你可能会注意到 ImageNet 的大小与 COCO 相比非常巨大，它有多个数量级的图像和类。
为了解决这个不平衡的数据集问题，我们对 COCO 进行了过采样，使得 ImageNet 和 COCO 中的图像比例为 4:1

在这个数据集上，我们训练我们的 **YOLO9000 模型**，使用 YOLOv2 的架构，但使用 3 个锚盒而不是 5 个来限制输出大小。

*   `**On a detection image**`，网络照常反向传播损失— *对象性*(框置信度得分，即图像包含对象的置信度)*包围框*，*分类错误*

![](img/d83c17e1982996a949152be90125f72d.png)

检测图像上的反向传播损失([来源](https://docs.google.com/presentation/d/14qBAiyhMOFl_wZW4dA1CkixgXwf0zKGbpw_0oHK8yEM/edit#slide=id.g23f389e290_1_128)

*   `**On a classification image**`，网络反向传播*客观损失*和*分类损失*。

对于客观损失，我们使用预测框与地面真实标签重叠≥ 0.3 IOU 的假设进行反向传播。

对于分类损失，我们找到预测该类最高概率的边界框，并且仅计算其预测树上的损失。

![](img/9cb2f307ec61fc74308978257e328070.png)

分类图像上的反向传播损失([来源](https://docs.google.com/presentation/d/14qBAiyhMOFl_wZW4dA1CkixgXwf0zKGbpw_0oHK8yEM/edit#slide=id.g1f9fb98e4b_0_40)

YOLO9000 是在 [**ImageNet 检测任务**](http://image-net.org/challenges/LSVRC/2017/) 上评估的，该任务有 200 个全标注类别。

这 200 个类别中的 44 个也在 COCO 中，因此该模型预计在分类上比检测上做得更好(因为由于 ImageNet，它有很多分类数据，但相对较少的检测数据)。

尽管 YOLO9000 没有看到剩余的 156 个不相交类的任何标记检测数据，但它仍然在这 156 个类上获得了 **19.7 的总体映射**，16.0 的映射。

这种性能优于 DPM 的性能(*可检测部分模型*，其使用滑动窗口方法进行检测)，但是 YOLO9000 在具有分类和检测图像的组合数据集上进行训练，&它同时检测 9000+个对象类别。

**都是实时的。**

![](img/096bde72799698005362f87fc9890f37.png)

各种架构和特征提取器的精度与速度([来源](https://arxiv.org/pdf/1611.10012.pdf))

在 ImageNet 上，YOLO9000 在检测动物方面做得很好，但在学习类如`clothing`和`equipment`时**表现很差**。

这可以归因于 COCO 有许多动物的标记数据，因此 YOLO9000 很好地概括了那里的动物；但是 COCO 对于*【泳裤】【橡皮】*等物体没有包围盒标签。所以我们的模型在这类分类上表现不佳。

![](img/8bbbcd50b829d15b5f9fab12ad7ecb21.png)

ImageNet 上的 YOLO9000 最佳和最差班级([来源](https://arxiv.org/pdf/1612.08242.pdf))

如果你读到这里，我会向你致敬。现在你知道 YOLOv2 和 YOLO9000 如何比最初的 YOLO 车型更好、更快、更强。

我们看到了如何使用 **WordNet(分级分类)**将`object detection and image classification datasets`结合起来并获得出色的性能。使用这种技术，YOLO9000 可以检测超过 9000 个对象类别，同时执行分类和检测，也是实时的！

使用`multi-scale training`(在不同分辨率的图像上训练模型)，我们看到了如何使 YOLOv2 模型**更健壮**和**使训练更稳定**。

在以后的文章中，我们将看到如何使用 Darknet 库自己实现 YOLOv2，并讨论更好的 YOLO 版本，如 **YOLOv3** 和 **v4** 。

希望这篇文章能让你对 YOLOv2 和 YOLO9000 架构有所了解。
在 [LinkedIn](https://www.linkedin.com/in/anamitra-musib-69694a63/) 和 [Medium](/@anamitramusib) 上关注我的最新文章:)

## 参考

1.  BatchNorm ( [TDS 文章](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c))，([研究论文](https://arxiv.org/pdf/1502.03167v3.pdf))
2.  [网络中的网络](https://arxiv.org/pdf/1312.4400.pdf)研究论文
3.  [全球平均池](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/)
4.  [COCO 数据集](https://cocodataset.org/#explore) (80 类)
5.  [WordNet](https://wordnetcode.princeton.edu/5papers.pdf) —一个在线词汇数据库
6.  [YOLO9000 载玻片](https://docs.google.com/presentation/d/14qBAiyhMOFl_wZW4dA1CkixgXwf0zKGbpw_0oHK8yEM/edit#slide=id.g23f389e290_1_13)
7.  *fast R-CNN*研究论文中的[锚盒](https://arxiv.org/pdf/1506.01497.pdf)(第 3.1.1 节)
8.  使用 YOLO、YOLOv2 和 YOLOv3 进行物体检测( [TDS 文章](/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088)
9.  [1000 类 ImageNet](https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57)
10.  [WordNet owl 示例](http://wordnetweb.princeton.edu/perl/webwn?o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=&o3=&o4=&r=1&s=owl&i=1&h=10000#c)
11.  [WordNet 梗示例](http://wordnetweb.princeton.edu/perl/webwn?o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=&o3=&o4=&r=1&s=terrier&i=1&h=100#c)
12.  [ImageNet 检测任务](http://image-net.org/challenges/LSVRC/2017/)