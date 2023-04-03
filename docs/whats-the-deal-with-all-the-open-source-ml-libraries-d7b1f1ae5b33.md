# 所有的开源 ML 库是怎么回事？

> 原文：<https://medium.com/analytics-vidhya/whats-the-deal-with-all-the-open-source-ml-libraries-d7b1f1ae5b33?source=collection_archive---------14----------------------->

![](img/2e04b8376f7288710c6cff45aa471349.png)

技术社区**令人惊叹**！我们通过在 GitHub 和 Stack Overflow 等平台上开源资源和分享知识来互相帮助。GitHub 上现在有超过 [1 亿个知识库](https://github.blog/2018-11-08-100m-repos/#:~:text=Today%20we%20reached%20a%20major,collaborating%20across%201.1%20billion%20contributions.)，堆栈溢出上有超过 1800 万个问题和[2700 万个答案](https://www.google.com/search?sxsrf=ALeKk01XgT35CtUawmQcdXegtNgEq4IthQ%3A1604180039002&ei=RtidX9DTPMXI_Qap0YbABw&q=how+many+answers+are+there+an+stack+overflow&oq=how+many+answers+are+there+an+stack+over&gs_lcp=CgZwc3ktYWIQAxgAMgcIIRAKEKABOgQIIxAnOgUIABCRAjoICAAQsQMQgwE6CwguELEDEMcBEKMCOgUIABCxAzoICC4QsQMQgwE6CAgAEMkDEJECOgQIABBDOgIIADoICC4QyQMQkQI6BwgAELEDEEM6CwgAELEDEIMBEMkDOgUIABDJAzoICCEQFhAdEB46BwgAEMkDEA06BAgAEA06BggAEA0QHjoICAAQCBANEB46BggAEBYQHjoJCAAQyQMQFhAeUOvLAVjfjAJgx5ICaARwAXgAgAG9AYgB6yGSAQUzMC4xNJgBAKABAaoBB2d3cy13aXrAAQE&sclient=psy-ab)。

在过去的两周里，当我探索文本的情感分析时，技术社区的这一特性确实帮助了我。我惊讶于所有可用的**开源库**来帮助我修补情感分析！

但这让我想到:*为什么公司开源这些机器学习(ML)库？一般来说，除非有经济上的激励，否则企业不会去做事。什么是 ***激励*** ？*

# 开源 ML 库很受欢迎

谷歌大脑、开放人工智能、脸书研究和谷歌 DeepMind 只是科技巨头开源他们的研究和模型供开发者使用的一些例子。

但是也有无数的开源项目是由开发者社区发起的。下面让我们探索一些可用于自然语言处理(NLP)和计算机视觉的免费库。

## 自然语言处理

![](img/1c3a8abf0c3f533578017d9342e4d67f.png)

主要归功于脸书的人工智能研究实验室， **PyTorch-NLP** 可以用来在 Python 中快速执行 NLP。它带有预先训练的嵌入、采样器、数据集加载器、度量、神经网络模块和文本编码器。

凭借所有这些特性，PyTorch 能够实现快速、灵活的实验和高效的生产。

另一方面， **TextBlob** 是一个用于处理文本数据的 Python 库，它独立于赞助其开发的科技巨头。它为常见的自然语言处理任务提供了一致的 API，如词性标注、名词短语提取、情感分析等。

当我使用 Twitter 的 API 对推文进行情感分析时，我个人发现 TextBlob 直观且易于使用。只需[一行代码](https://colab.research.google.com/drive/1hCYmiatrfe35yBaVg4SS63rFUJjddpf4#scrollTo=6OTJUSVxAsM1)就可以得到文本的**主观性**和**极性**。

![](img/2a263c1cdf84f245dabf37065a2f3821.png)

使用 Textblob 对比尔·盖茨的推文进行情感分析— [来源](https://colab.research.google.com/drive/1hCYmiatrfe35yBaVg4SS63rFUJjddpf4?usp=sharing)

## 计算机视觉

PoseNet 是一种计算机视觉模型，可用于通过估计关键身体关节的位置来估计图像或视频中人的姿势。PoseNet 由 **TensorFlow** 开发。并且 **Google Brain** 团队创建了用于数值计算和大规模机器学习的 TensorFlow 库。

姿势估计是指计算机视觉技术，它检测图像和视频中的人体形状，以便人们可以确定，例如，某人的膝盖在图像或视频中的位置。姿态估计技术不像对象检测那样识别图像中的人/物。相反，该算法只是简单地估计身体关键关节的位置。

当 TensorFlow 推出时，开发人员能够使用 Javascript 和高级层 API，仅在浏览器中建立、训练和运行机器学习模型。由于 [TensorFlow](https://github.com/tensorflow/tensorflow) 是一个开源库，任何人都可以在 GitHub 上查看 repo 并修改它。

同样，像 TextBlob 一样，我发现 PoseNet 使用起来很直观。我不需要制作一个 ML 模型来在浏览器中找到一个人的姿势和骨骼的关键点。

# 为什么 ML 库是开源的？

![](img/ad82a593e23e55637cec6f3afa288b6a.png)

在商业中，很少有事情仅仅是为了友好而发生的。通常，决策是根据公司的财务利益做出的。

尽管我喜欢所有的开源库，但我认为更多的人需要开始问**为什么要制造它们。其实一般来说，人应该多问问为什么。**

![](img/174ba0113dfe60b3a8397a2c2b0a3970.png)

脸书人工智能是为*【赋予世界力量和连接世界的变革性技术的先驱】*而制造的。他们通过与社区的公开合作促进应用研究来完成这一使命。

脸书人工智能在许多领域进行了研究并发表了论文，包括计算机视觉，对话人工智能，自然语言处理，在社交平台上保护人们安全的人工智能解决方案，自然语言处理，排名和推荐，语音和音频等等。

与此同时，TensorFlow 的开发是为了创建*“一个完整的生态系统，帮助你用机器学习解决具有挑战性的现实世界的问题”*。本质上，TensorFlow 使您可以轻松地构建和部署 ML 模型，以便**每个人**都可以使用 ML 的力量。

由于 TensorFlow 易于使用，来自各行各业的各种公司实施 ML 来解决他们的**最大的问题**。专门从事医疗保健、社交网络甚至电子商务的公司使用 TensorFlow 来帮助他们将 ML 集成到他们的产品中。

## 使用 TensorFlow 的公司示例

![](img/f9395fce0d2b12c32c054062fdb6abaa.png)

让我们进一步探索不同领域的公司如何使用 tensor flow——也许我们会发现一种**模式**。

**空中客车**

空中客车公司致力于航空航天的未来，使用 TensorFlow 从卫星图像中提取信息，并向客户提供有价值的见解。使用 ML 可以让他们监测地球表面的变化。

**旋转式传送带**

Carousell 是一个新加坡智能手机和基于网络的消费者对消费者和企业对消费者市场，用于买卖新的和二手商品。他们使用 TensorFlow 来改善买方和卖方的体验。他们建立具有深度图像和 NLP 理解的 ML 模型，以使卖家和买家都受益。

**通用电气医疗保健**

GE Healthcare 使用 TensorFlow 训练了一个神经网络，以识别大脑磁共振成像的解剖结构。使用 TensorFlow，GE Healthcare 正在训练一个神经网络，以在脑部 MRI 检查中识别特定解剖结构，从而帮助提高速度和可靠性。

还有更多例子，请随意查看这里的。

## 让我们看看谷歌收购的公司

谷歌的母公司 Alphabet 已经收购了超过 230 家公司。谷歌在 2020 年进行的最近一次收购是以大约 21 亿美元收购 Fitbit。

![](img/818ac8b9003c6fe03c716164a96ff74b.png)

Fitbit 充电 4

Fitbit 由 James Park 和 Eric Friedman 于 2007 年初在 T21 创立，他们发现了在小型可穿戴设备中使用传感器的潜力。现在让我们想象一些事情…

如果谷歌从 2007 年 Fitbit 成立的第一天起就在开发它会怎么样？让我们想象一下，有 50 名员工在这个项目上工作。谷歌初级软件开发人员的典型薪资是 105，638 美元。因此，从 2007 年到 2020 年，谷歌将不得不支付约 68，664，700 美元来支付员工创建 Fitbits 的费用。

这还没有考虑到制作这个项目所需的硬件和软件。而且在 2007 年，谷歌没有办法确保 Fitbits 会成功，投资会有回报。

当然，现在这个场景完全是虚构的，但它展示了为什么开源 ML 库可能符合科技巨头**的利益**。如果我是一个科技巨头，我也会开源软件，希望有公司使用它，用他们**自己的时间和金钱**开发一些东西。如果我发现他们有巨大的潜力，我就会收购他们。

## 开发者自己做的 ML 库呢？

我们以 TextBlob 为例。在 [116 公司](https://enlyft.com/tech/products/textblob)周围井使用。任何人都可以免费使用这个图书馆。

> 特此免费授予获得本软件和相关文档文件(“软件”)副本的任何人不受限制地经营本软件的权利，包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或出售本软件副本的权利

开源的好处之一是开发者可以随意地为项目做贡献。此外，开源对于建立社区、拓宽营销渠道、提高代码质量以及开发人们喜爱的软件都有很大的帮助。

# 那么，你应该使用 ML 库吗？

![](img/be7c87536e1fd079868206e65d80dad0.png)

当决定是从头开始制作 ML 模型还是使用开源 ML 库时，有几种思想流派。这个决定归结为**速度、知识和可扩展性。**

通常，库允许您更快地创建模型并将模型部署到生产中。然而，如果 ML 工程师不知道**库是如何工作的**,项目将很可能无法扩展，因为当出现 bug 或者模型产生一些奇怪的结果时，ML 工程师将不知道如何修复它们。

还有另一个巨大的因素应该有助于决定使用 ML 库——数据集。但是你怎么能保证库被训练的数据集没有偏见呢？这是我下一篇文章的主题😉

# TL；速度三角形定位法(dead reckoning)

*   科技巨头和开发者为 ML 开发了很多开源库
*   科技巨头可能会开源 ML 库，希望公司可以使用它们来改进。然后，科技巨头可能会收购该公司