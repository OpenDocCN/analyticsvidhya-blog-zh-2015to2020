# 卷积神经网络的迁移学习

> 原文：<https://medium.com/analytics-vidhya/transfer-learning-with-convolutional-neural-networks-e9c513d41506?source=collection_archive---------8----------------------->

## 如何根据自己的需求改造花哨的模型

> 本文概述了卷积神经网络的迁移学习。对于为您自己的计算机视觉任务实现预训练卷积神经网络(InceptionV3)的分步指南，[单击此处](/@mikhaillenko/instructions-for-transfer-learning-with-pre-trained-cnns-203ddaefc01)。(与[赵楷](https://medium.com/u/c2535a8887a4?source=post_page-----e9c513d41506--------------------------------)合作开发。)

[人工神经网络](https://en.wikipedia.org/wiki/Neural_network) (ANNs)是数据科学领域最新、最伟大的机器学习模型。它们的性能[神秘而令人印象深刻](https://arxiv.org/abs/1608.08225)，即使只有一个*单个*隐藏层，它们也可以用来近似*任何*函数到*任何*期望的精度。

那么，考虑到它们卓越的性能和普遍的适用性，为什么还有人会使用其他模型呢？

人工神经网络需要大量的数据和计算能力才能超越其他模型。对于人们通常使用的数据集和硬件，使用人工神经网络就像使用宇宙飞船进行早晨的通勤一样。

因此，有人可能会想，有没有一种方法既能获得人工神经网络的优点，又能避免其缺点？

## 迁移学习

[迁移学习](https://en.wikipedia.org/wiki/Transfer_learning)是一种采用为某种目的构建的模型，并为不同的目的实现它的方法。典型的例子是使用[无监督机器学习](https://en.wikipedia.org/wiki/Unsupervised_learning)的输出(例如 [k 均值聚类](https://en.wikipedia.org/wiki/K-means_clustering))作为某个[有监督机器学习](https://en.wikipedia.org/wiki/Supervised_learning)模型的输入。迁移学习也适用于修改预训练的人工神经网络来执行新功能，这是本文的主题。

在我们讨论人工神经网络的迁移学习之前，我们需要了解一下它们的基本结构。

## 全连接的前馈神经网络

![](img/d37c1134ba2bf63daafc3039ad4ed61a.png)

[By Glosser.ca —自己的作品，衍生文件:人工神经网络. svg，CC BY-SA 3.0](https://commons.wikimedia.org/w/index.php?curid=24913461)

在其[最简单的形式](https://en.wikipedia.org/wiki/Feedforward_neural_network)中，ANN 只是一系列层叠的逻辑回归，其中 observation⁴的每个特征值由[输入层](https://en.wikipedia.org/wiki/Artificial_neural_network#Organization)中的一个节点表示，该节点乘以某个权重，并通过某个激活 function⁵通知下一层的每个节点(称为[隐藏层](https://en.wikipedia.org/wiki/Artificial_neural_network#Organization))。重复这个过程，直到我们到达构成预测的[输出层](https://en.wikipedia.org/wiki/Artificial_neural_network#Organization)。这被称为正向传播阶段。在训练模型时，将预测值与实际值进行比较，然后向后调整每层的权重，以优化某个可微分目标函数。这被称为[反向传播](https://en.wikipedia.org/wiki/Backpropagation)阶段(或简称为反向传播)，正是在反向传播期间，人工神经网络实际进行学习。

学习阶段使得人工神经网络需要所有的数据和处理能力，因为模型必须使用损失函数的梯度下降来调整每个(百万)of)⁶权重。这就像一千万个微积分问题——即使对计算机来说也是很难的。相比之下，前向传播只是将值乘以权重，应用简单的激活函数，然后将它们相加。这就像一千万个算术问题——对计算机来说是小菜一碟。

## 卷积神经网络

[卷积神经网络](https://en.wikipedia.org/wiki/Convolutional_neural_network)是用于计算机视觉任务的特殊神经网络，如图像分类、物体检测和物体定位。它们的结构就像上面描述的简单的人工神经网络一样，只是它们从一个卷积基开始，这个卷积基是为了学习图像中的独特模式而设计的。

![](img/d3fac43fc6926d91875044d625750815.png)

图片来自 [Sumit Saha](https://medium.com/u/631ee5e6343e?source=post_page-----e9c513d41506--------------------------------) 在 CNN 上发表的[精彩文章](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

连续的卷积层学习越来越高阶的模式(例如，从边到角到形状)。在卷积基础之后，完全连接的层学习这些模式和标签之间的关系，这形成了它们预测的基础。

给定一个为一些一般任务而严格训练的复杂的 CNN，例如在一千个不同的 labels,⁷中分类**，我们只需要几个步骤**就可以根据我们的定制需求重新设计模型。

首先，我们移除为模型的原始目的配置的全连接层，并用适合于我们的定制任务的全连接层来替换它们(重要的是，输出 layer⁸的形状及其激活 function⁹以及模型的损失函数⁰可能需要改变)。然后，我们冻结卷积块中的层，以便当模型为我们的任务而训练时，预训练中学习的模式不会被*覆盖*，而只是为了新的目的而被*重新解释*。

瞧啊。现在，我们有了一个拥有数千万或数亿学习参数的模型，准备好适应我们的定制任务*，由于只需要训练完全连接的层*，我们可以使用训练原始模型所需的一小部分数据和处理能力来完成。可以肯定的是，如果整个模型是为定制任务从头开始构建的，性能可能会提高，但对于我们这些没有超级计算机和几百万标记图像的人来说，迁移学习肯定是一个巧妙的小技巧。

###

参见[通用逼近定理](https://en.wikipedia.org/wiki/Universal_approximation_theorem)了解更多。
在使用传统的[中央处理器](https://en.wikipedia.org/wiki/Central_processing_unit)(CPU)时尤其如此，但通过使用[图形处理单元](https://en.wikipedia.org/wiki/Graphics_processing_unit)(GPU)和最近的[张量处理单元](https://en.wikipedia.org/wiki/Tensor_processing_unit) (TPUs)，建模时间可以显著减少，这是由谷歌开发的用于拟合神经网络的特殊 GPU。
参见 [3Brown1Blue](https://www.3blue1brown.com) 的[神经网络介绍](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2&t=0s)以获得对其结构的精彩总结，该总结既全面又易于理解。
⁴例如，给定颜色通道中给定像素的着色
⁵例如，[校正线性单元(ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
在 [ImageNet](http://www.image-net.org/about-overview) 的数据集上预先训练的⁶模型具有数千万个参数的量级，尽管一些具有数亿个参数。
⁷——一个训练有素的美国有线电视新闻网(CNN ),在任何数量的用于癌症诊断的正电子发射断层扫描上，都不能帮助区分猫和车；而被训练来预测一千个类别中的一个类别的 CNN 被期望已经学习了许多可以应用于定制分类问题的一般特征。
⁸对于多类分类，输出层中神经元的数量必须等于类的数量；对于二元分类或回归，输出层中必须只有一个单一神经元。
⁹为多类分类，使用" softmax "激活功能；对于二元分类，使用“sigmoid”激活函数；对于回归，使用“线性”激活函数。
⁰对于多类分类，使用“分类交叉熵”损失函数；对于二元分类，使用“二元交叉熵”损失函数；对于回归，使用“均方误差”损失函数。
为此目的冻结卷积块中的层有时被称为*特征提取*(参见 [TensorFlow 文档](https://www.tensorflow.org/tutorials/images/transfer_learning))。或者，我们可以解冻这些层，但规定一个极低的学习率，因此机器依赖的模式可以针对我们的任务进行调整，这有时被称为*微调*(这比特征提取需要更多的训练数据)。