# 人工智能和数据科学术语

> 原文：<https://medium.com/analytics-vidhya/ai-and-data-science-terminologies-89d77c0b4e52?source=collection_archive---------26----------------------->

人工智能和数据科学入门笔记

## **机器学习**

机器学习是人工智能的一个子集，它使用计算机算法来分析数据，并根据所学做出智能决策。机器学习不是遵循基于规则的算法，而是建立模型来分类，并根据数据进行预测。

机器学习是一个非常广泛的领域，我们可以将其分为三个不同的类别**监督学习、非监督学习和强化学习**，我们可以用这些来解决许多不同的任务。

## 监督学习

当我们在数据集中有了分类标签，我们用它们来建立分类模型。这意味着当我们收到数据时，它有代表数据是什么的标签。

## **无监督学习**

在无监督学习中，我们没有类别标签，我们必须从非结构化数据中发现类别标签。这是通过聚类来完成的，它会调出所有相似的数据并对它们进行标记。

## **强化学习**

强化学习是一个不同的子集，它使用奖励函数来惩罚不好的行为或奖励好的行为。

## **监督学习**可以分为—

*   回归
*   分类
*   神经网络

## **回归模型**

回归模型通过查看解释变量和响应变量之间的关系建立，其中响应变量是连续变量。
`**Regression models estimates the continuous values i.e. the response variable.**`

## 分类模型

分类是一种监督学习。它指定数据元素所属的类，最适合在输出具有有限值和离散值时使用。它也为一个输入变量预测一个类。
`**Classification models estimates the categorical values i.e. the response variable.**`

## **分类模型的形式有:**

*   决策树
*   支持向量机
*   逻辑回归
*   随机森林

分类可用于从数据中提取特征，这些特征是输入模式的独特属性，有助于确定输出类别或输出类。

## **神经网络模型**

神经网络是指模仿人脑结构的结构。这些是被称为`**neurons**` 的更小的处理单元的集合，它们是模仿人类大脑处理信息的方式。它从大脑的生物神经网络中借用了一些想法，以便近似它的一些处理结果。神经元像生物神经元一样接受输入数据，并随着时间的推移学习做出决定。

## **反向传播**

神经网络通过称为`**Backpropagation**` **的过程进行学习。**反向传播使用一组将已知输入匹配到期望输出的训练数据。

*   首先，输入被插入网络，输出被确定。
*   然后一个`error function`决定了给定的输出离期望的输出有多远。
*   最后`adjustments`被制造成`reduce the errors`。

神经元的集合被称为`**Layer**` ，这一层接收输入并提供输出。任何神经网络都会有`**at least one input layer and one output layer**`。它还会有`**one or more hidden layers**`来模拟人类大脑中进行的活动类型。隐藏层接受一组`**weighted inputs**`并通过一个`**activation function**`产生和输出。具有一个以上隐藏层的神经网络称为`**Deep Neural Network**`。

## 感知器

最简单和最古老的神经网络。它们`single-layered neural networks`由直接连接到输出节点的输入节点组成。输入层`forwards the input values to the next layers by multiplying the weights and summing the results`。隐藏层从其他节点获得输入，并将它们的输出转发给其他节点，`hidden`和`output layers`有一个属性叫做`**Bias**` **。**

**偏差**是一种特殊的权重，在考虑其他输入后添加到节点上。

## 激活功能

激活功能**确定节点将如何响应输入**。该函数针对输入和偏差之和运行，然后将结果作为输出转发。它们可以采取不同的形式，选择它们是神经网络成功的关键因素。

## 卷积神经网络/ CNN

**CNN** 是**多层 NN** 从**动物视觉皮层获取灵感。**

CNN 在如下应用中是有用的

*   图像处理
*   视频识别
*   自然语言处理

**卷积**是一种**数学运算**，其中一个函数应用于另一个函数，结果是两个函数的混合。

卷积擅长将图像中的`**detecting simple structures**` 和那些简单的特征放到`**construct more complex features**`中。在卷积神经网络中，这个过程在一系列层上重复，每一层对前一层的输出进行卷积。

## 递归神经网络

递归是因为它们`perform same task for every element of sequence, with prior outputs feeding subsequent stage inputs.`标准的神经网络独立地对待所有的输入，对输出负责，但是在 RNNs 中，对先前观察的依赖是重要的。

每一层代表特定时间的输出，所以在 NLP 中，如果你想知道一个单词的上下文，你需要知道前一层的输出来生成输出。

## 模型参数

模型参数是一个配置变量`internal to the model`，其值可以是`estimated from data`。

## 模型超参数

模型超参数是一种配置，其值为`external to the model`和`cannot be estimated from data`。

以防你对什么是**训练、验证和测试集感到困惑。
参考本** [**条**](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7) **。这是上面的简短注释。**

## **培训** vs **验证 vs 测试集**

数据分为三组**，训练、验证和测试组**。
**训练集**用于`**train our model**` **，** ，**验证集**用于`**fine-tune the model hyperparameters**`。
和**测试集**在模型训练到`**evaluate the performance of trained model**`后使用。

## 深度学习

深度学习是专门的`**subset of Machine learning**` **。它对算法进行分层，以创建一个神经网络，这是大脑结构和功能的人工复制。使人工智能系统能够在工作中不断学习，并提高结果的质量和准确性。**

像图像、视频和音频文件这样的深度学习`enables the systems to learn from unstructured data`也为自然语言处理开辟了可能性，即人工智能系统的自然语言理解能力，并允许它们找出意图和上下文。

深度学习不直接将输入映射到输出，而是依赖于几层处理单元。每一层将其输出传递给下一层，下一层对其进行处理并传递给下一层。有许多层，这就是为什么它被称为深度学习。

当创建深度学习算法时，工程师或开发者配置将该层的输出连接到下一层的输入的层数和函数类型。

深度学习已被证明在各种任务中非常有效，包括..

*   图像字幕
*   语音识别和转录
*   面部识别
*   医学成像
*   语言翻译

我将介绍更多关于人工智能和数据科学的内容。更多类似**数据分析、** **数据挖掘**和**图像处理**的东西，请参考[这里的](/@harshit120299/)。