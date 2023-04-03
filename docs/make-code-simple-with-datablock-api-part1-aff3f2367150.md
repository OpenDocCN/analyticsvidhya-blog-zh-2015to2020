# 使用数据块 API 第 1 部分简化代码

> 原文：<https://medium.com/analytics-vidhya/make-code-simple-with-datablock-api-part1-aff3f2367150?source=collection_archive---------30----------------------->

2020 年 6 月 14 日

# **Blog14**

![](img/307e5f68054b79a30a4c428a7d706827.png)

欢迎光临！！！

写这篇博客的目的是向你介绍 fastai 令人敬畏的数据块 api。这是博客的第一部分，第二部分将是代码方法。

尽管 fastai 遵循自上而下的方法，但我写博客的第一部分时没有代码，在第二部分中，理论解释设置了学习数据块的动机和代码。(当然，在编写代码之前，Jeremy 给出了使用它的动机和令人敬畏的解释，所以我认为在编写代码之前这很重要。)

所以让我们开始吧！！！

如果你使用过任何深度学习框架(我使用 PyTorch，所以说 w.r.t. it)来建立一个模型来解决深度学习问题，你会经历收集数据的步骤，这是什么类型的问题(如图像分类，分割)，看看什么是因变量和自变量，如何将数据分成训练和验证集，应用变换来提高准确性。

在这个过程中，您可能还为所有这些任务编写了冗长代码，但是如果我告诉您，您可以在一个单独的块中完成这些任务，那就太棒了(您也可以用正常的方式完成所有这些任务，然后重构它，但是这种数据块方法对我来说看起来不错，因为我在遵循这种方法时犯的错误较少)

那么什么是**数据块** api 呢？？？

数据块 api 是 fastai 中的高级 api。数据块 API 是一个用于数据加载的表达性 API。这是一种系统地定义为深度学习模型准备数据所需的所有步骤的方法，并为用户提供了一个混合和匹配这些片段(我们称之为数据块)的食谱

当我们构建批处理和数据加载器时，可以把数据块想象成一个指令列表。它不需要明确地完成任何项目，而是一个如何操作的蓝图。编写数据块就像编写蓝图一样。

我们刚才看到了数据加载器这个词。让我们看看。PyTorch 和 fastai 有两个主要的类来表示和访问训练集或验证集:

`Dataset`::返回单个项目的自变量和因变量元组的集合

`DataLoader`::提供小批量流的迭代器，其中每个小批量是一批自变量和一批因变量的组合

有趣的是，fastai 提供了两个类来将您的训练集和验证集结合在一起:

`Datasets`::包含训练数据集和验证数据集的对象

`DataLoaders`::包含训练数据加载器和验证数据加载器的对象。

fastai library 有一种构建数据加载器的简单方法，这种方法足够简单，只需要很少的编码知识就可以获得，并且足够先进，可以进行探索。

创建数据块的步骤如下。

这些步骤由数据块 API 定义，可以在查看数据时作为问题提出:

*   你的输入/目标类型是什么？(`Blocks`)
*   你的数据在哪里？(`get_items`)
*   需要对输入应用什么吗？(`get_x`)
*   需要对目标应用什么吗？(`get_y`)
*   如何拆分数据？(`splitter`)
*   我们需要在成形的物品上应用一些东西吗？(`item_tfms`)
*   我们需要在成型的批次上应用一些东西吗？(`batch_tfms`)

就是这个！！

当你回答这些问题时，你写了一个数据块。

您可以将每个问题或步骤视为构建 fastai 数据块的砖块。

*   阻碍
*   获取 _ 项目
*   获取 x/获取 y
*   分流器
*   item_tfms
*   批处理 _tfms

在构建数据加载器时，查看数据集非常重要。而使用数据块 api 是解决问题的策略或方法。首先要看的是数据是如何存储的，即以何种格式或方式存储，并与著名的数据集进行比较，是否以这种方式存储以及如何处理它。

**块**这里用来定义预定义的问题域。例如，如果这是一个图像问题，我可以告诉图书馆使用枕头，而不用明确地说出来。并说是单标签还是多标签分类。还有很多像 ImageBlock，CategoryBlock，MultiCategoryBlock，MaskBlock，PointBlock，BBoxBlock，BBoxLblBlock，TextBlock 等等。(我将在博客的第二部分解释所有与代码相关的细节)

**get_items** 就是回答数据在哪里？

例如，在图像问题中，我们可以使用`get_image_files`函数获取我们图像的所有文件位置，我们可以查看数据(我将在博客的第 2 部分解释所有代码相关的细节)。

**get_x** 是回答需要对输入应用什么东西吗？

**get_y** 就是你如何提取标签。

**拆分器**就是你想怎么拆分我们的数据。通常，这是训练数据集和验证数据集之间的随机分割。

datablock api 剩下的两块砖是 item_tfms 和 batch_tfms，是增广。

**item_tfms** 是基于单个项目应用的项目转换。这是在 CPU 上完成的。

**batch_tfms** 是应用于批量数据的批量转换。这是在 GPU 中完成的。

使用 datablock 中的这些砖块，我们可以处理和构建数据加载器，为不同类型的问题做好准备，如分类、对象检测、分割和所有其他不同类型的问题。

数据块 API 提供了简明性和表达性的良好平衡。在数据科学领域，scikit-learn 管道方法被广泛使用。该 api 提供了非常高的表达能力，但不够固执己见，无法确保用户完成所有必要的步骤，为建模准备好数据，但这是在 fastai 数据块 API 中完成的。

现在我们已经了解了什么是数据块 api，让我们包装所有的东西并构建一个。

是时候了！！让我们看看牛津 IIIT pets 数据集的单标签分类的代码(仅数据块)。

```
pets = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, 
splitter=RandomSplitter(), 
get_y=Pipeline([attrgetter("name"), RegexLabeller(pat = r'^(.*)_\d+.jpg$')]), 
item_tfms=Resize(128), 
batch_tfms=aug_transforms())
```

这是什么代码？？？

提醒:这是博客的介绍部分，

好奇想知道代码中有什么，以及如何编写代码，请阅读将于 2020 年 6 月 21 日星期日上午 10:30 IST[发表的博客第二部分](https://kirankamath.netlify.app/blog/make-code-simple-with-datablock-api-part2/)。

学分:

*   fastai [docs](https://dev.fast.ai/)
*   感谢扎克·穆勒，[在 datablock api 上的博客](https://muellerzr.github.io/fastblog/datablock/2020/03/21/DataBlockAPI.html)，请继续写博客和制作视频。
*   fastai 深度学习的分层 API[论文](https://arxiv.org/pdf/2002.04688.pdf)

*最初发布于*[*https://kirankamath . netlify . app*](https://kirankamath.netlify.app/blog/fastais-datablock-api/)*。*