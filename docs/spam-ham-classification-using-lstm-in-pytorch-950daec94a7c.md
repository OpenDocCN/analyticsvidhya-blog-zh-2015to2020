# PyTorch 中使用 LSTM 的垃圾邮件-火腿分类

> 原文：<https://medium.com/analytics-vidhya/spam-ham-classification-using-lstm-in-pytorch-950daec94a7c?source=collection_archive---------5----------------------->

这就是如何在 PyTorch 中建立和训练 LSTM 模型，并使用它来预测垃圾邮件或火腿。

本指南的 Github 回购是[这里的](https://github.com/sijoonlee/spam-ham-walkthrough)，你可以在回购中看到 [Jupyter 笔记本](https://github.com/sijoonlee/spam-ham-walkthrough/blob/master/walkthrough.ipynb)。我的建议是下载笔记本，看看这个演练，然后玩一玩。

![](img/fe39cf39bdc6f1197b9780df8bb83659.png)

由[网站主持](https://unsplash.com/@webhost?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 一.安然垃圾邮件数据集

研究人员 V. Metsis、I. Androutsopoulos 和 G. Paliouras 将安然语料库中的 3 万多封电子邮件归类为垃圾邮件/垃圾邮件数据集，并向公众开放

1.  进入[网站](http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html)
2.  在站点中找到`Enron-Spam in pre-processed form`
3.  下载恩龙 1、恩龙 2、恩龙 3、恩龙 4、恩龙 5 和恩龙 6
4.  提取每个 tar.gz 文件
5.  目录 enron1，enron2，…，enron6 应该在你放置 [Jupyter 笔记本](https://github.com/sijoonlee/spam-ham-walkthrough/blob/master/walkthrough.ipynb)的同一个目录下

# 二。处理数据

数据将会是

1.  从文件加载
2.  用于建立词汇词典
3.  符号化和矢量化

让我们深入了解每一步

## 二-1。从文件加载数据

需要将 file_reader.py 下载到同一个文件夹中。我简单介绍一下我写的代码( [file_reader.py](https://github.com/sijoonlee/spam-ham-walkthrough/blob/master/file_reader.py) )。

首先，垃圾邮件和火腿集合将被分别加载到`spam`和`ham`中。其次，`ham`和`spam`将合并为`data`。第三，将为垃圾邮件和火腿生成标签，分别为 1 和 0
最后，它返回数据和标签:

加载的数据包括 3000 个 hams 和 3000 个 spam——总共 6000 个
。如果您设置了`max = 0`,您可以从文件中获取所有数据。但是对于这个教程，6000 套就够了

## 二-2。建立词汇词典

词汇词典中有键和值:分别是单词和值。例如{'the': 2，' to': 3}

字典里的整数怎么选？

请想象一个来自 6000 个数据集的单词列表。像“the”、“to”和“and”这样的常用词更有可能在列表中出现多次。我们将计算出现的次数，并根据次数对单词进行排序。

## 二-3。标记化和矢量化数据

让我们先看看示例代码

`Tokenization`在这里表示从数据集到单词列表的转换。例如，假设我们有如下数据

```
"operations is digging out 2000 feet of pipe to begin the hydro test"
```

标记化将产生如下单词列表

```
['operations', 'is', 'digging', ...
```

`Vectorization`在这里表示使用 II-2 中内置的 vocab 字典将单词转换为整数

```
[424, 11, 14683, ...
```

现在我们可以继续我们的数据集

# 三。构建数据加载器

到目前为止，数据是以矢量化的形式处理的。现在，轮到构建数据加载器了，它将把成批的数据集输入到我们的模型中。为此，

1.  需要自定义数据加载器类
2.  需要三个数据加载器:用于训练、验证和测试

## 三-1。自定义数据加载器

`Sequence`这里指的是电子邮件中的矢量化单词列表。
由于我们准备了 6000 封电子邮件，因此我们有 6000 个序列。

由于序列具有不同的长度，所以需要将每个序列的长度传递到我们的模型中，而不是在虚拟数字(0 表示填充)上训练我们的模型。

因此，我们需要定制的数据加载器来返回每个序列的长度以及序列和标签。

另外，数据加载器应该按照每个序列的长度对批处理进行排序，并首先返回批处理中最长的一个，以使用 torch 的`pack_padded_sequence()`(稍后您将看到这个函数)

我使用 torch 的 sampler 构建了 iterable 数据加载器类。

## 三-2。实例化 3 个数据加载器

模型将在**训练**数据集上训练，由**验证**数据集验证，最后在**测试**数据集上测试:

# 四。构建模型

该模型包括

1.  把...嵌入
2.  打包序列(去掉填料)
3.  LSTM
4.  解开序列(恢复填充)
5.  全连接层
6.  乙状结肠激活

## IV-1。把...嵌入

根据 PyTorch.org 的文档，“单词嵌入是一个单词的语义的表示”

> *要知道* `*Word Embeddings*` *是什么，我推荐你去阅读* [*PyTorch 文档*](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)

## IV-2。pack_padded_sequence()的使用

请记住，我们在序列中添加了填充(0)。由于序列具有不同的长度，所以需要将填充添加到较短的序列中，以匹配张量中的维度。问题是模型不应该在填充值上训练。pack_padded_sequence()将删除批数据中的填充并重新组织它

举个例子，

> *要了解更多关于* `*pack_padded_sequence()*` *的内容，推荐你去读一下* [*layog 的栈溢出贴*](https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch/49473068#49473068) *和* [*HarshTrivedi 的教程*](https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial)

## IV-3。LSTM

LSTM 代表“长短期记忆”，一种 RNN 架构。注意，如果没有提供`(h_0, c_0)`，根据 [PyTorch 文档](https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html)，两个 **h_0** 和 **c_0** 都默认为零

> *对于* `*LSTM*` *，我会推荐你去读读* [*colah 的博客*](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

# 动词 （verb 的缩写）培训、验证和测试

## 第一组。培训和验证

## V-2。试验

## 不及物动词预测

这是我最近得到的英语课广告的一部分。预测它是垃圾邮件似乎有点棘手，不是吗？

```
Have you been really busy this week? Then you'll definitely want to make time for this lesson. Have a wonderful week, learn something new, and practice some English!
```

让我们把它放到模型中，看看结果是否是“垃圾邮件”

模型算垃圾！

感谢您的阅读！

我从来没有期望自己写一本指南，因为我仍然认为自己是深度学习的初学者。如果您发现有什么问题，请发邮件给我或留下您的意见，我们将不胜感激。

邮箱:[shijoonlee@gmail.com](mailto:shijoonlee@gmail.com)Github:[github.com/sijoonlee](https://github.com/sijoonlee)