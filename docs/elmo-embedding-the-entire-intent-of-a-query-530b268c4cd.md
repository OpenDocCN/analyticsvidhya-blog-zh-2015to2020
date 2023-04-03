# ELMo 嵌入——查询的全部意图

> 原文：<https://medium.com/analytics-vidhya/elmo-embedding-the-entire-intent-of-a-query-530b268c4cd?source=collection_archive---------1----------------------->

![](img/0df24a664f35536d6de07bbe9d95e760.png)

电子商务搜索

作为搜索查询理解的继续，接下来要解决的问题是[整体查询理解](/@dtunkelang/query-understanding-divided-into-three-parts-d9cbc81a5d09)。我们将详细讨论这一点，因为我们已经在这里讨论了简化查询理解部分[。这里，我们以电子商务搜索为例。问题是在特定的分类中找到查询的意图，如 L1/L2/L3 类别，也称为类别分类。](https://towardsdatascience.com/understanding-the-search-query-part-i-632d1b323b50)

![](img/064adae2423d282530eb1fae1486c9d7.png)

电子商务网站上的客户搜索

让我们来谈谈一个现实生活中的网上购物的例子，你想寻找类似“洋葱 1 公斤鲜”的东西。现在，为了使结果更加精确，搜索结果会与您的查询意图所在的可能类别名称一起显示，即“杂货和美食”。现在，如果你点击预测类别，你会看到属于该特定类别的产品，因此，这是一个增强体验的精确结果。

这个问题的解决方案是我们将要讨论的基于机器学习的。

# 查询的目的

这是一个多类多标签分类问题，其中输入将是一组标有其类别的搜索查询，并且特定的搜索查询可能属于许多互斥类别中的一个以上的类别。更详细地说，假设我们有 3 个层次的类别 L1，L2 和 L3。搜索查询可以属于每个层级的一个或多个类别，例如搜索查询——“苹果”在 L1 可以是“电子产品”和“杂货”，在 L2 可以是“笔记本电脑”和“水果”，等等。我们有一个基于深度学习的模型，使用 Elmo 作为嵌入层。

## ELMo 嵌入

ELMo 是由 [AllenNLP](https://allennlp.org/elmo) 创造的，它不同于 Glove、Fasttext、Word2Vec 等。提供上下文化的单词嵌入，其单词的向量表示在句子与句子之间是不同的。

> ELMo 是一种深度语境化的单词表示，它模拟(1)单词使用的复杂特征(例如，句法和语义)，以及(2)这些使用如何在语言语境中变化(即，模拟多义性)。这些单词向量是深度双向语言模型(biLM)的内部状态的学习函数，该模型是在大型文本语料库上预先训练的。

![](img/05a001fceb31b06c86fbaa46796a7b7b.png)

取自[此处](http://jalammar.github.io/illustrated-bert/)

ELMo 在[**10 亿字基准**](http://www.statmt.org/lm-benchmark/) 上接受训练，这是来自 2011 年 WMT 新闻抓取数据的大约 8 亿个令牌。

# 为什么是埃尔莫？使用 ELMo 的动机

其他单词表示模型如 [Fasttext](https://github.com/facebookresearch/fastText) ，Glove 等。也生成单词的嵌入，但是我们选择 Elmo 有几个原因。让我们一个一个地参观:

1.  Elmo 提供了一个单词在句子中的嵌入，即一个单词可能有不同的含义，这取决于它在上下文中的使用，类似于上面照片中的例子。“苹果”这个词可能是一个“品牌”名称，也可能是一种“水果”。因此，如果给出一个类似'**苹果汁'**的查询，这里为标记' Apple '生成的嵌入将不同于' **Apple Laptop** '中的嵌入。而在一般的电子商务搜索查询中，很可能会发生这种情况。
2.  另一个原因是，由于 LSTM 网络在 ELMo 模型中被内部使用，我们需要担心在训练数据集的字典中不存在的单词，因为它也生成字符嵌入。它允许网络使用形态学线索来为训练中看不见的词汇外标记形成健壮的表示。在品牌名称的情况下，我们通常会面临词汇外的标记问题。
3.  此外，ELMo 还在该领域的顶级会议之一 NAACL-HLT 2018 上获得了最佳[论文奖。](https://naacl2018.wordpress.com/2018/04/11/outstanding-papers/)

# 使用 TF Hub 实现

[TensorFlow Hub](http://tensorflow.org/hub) 是一个发布、发现和重用 TensorFlow 中部分机器学习模块的平台。对于模块，它意味着张量流图的一个独立部分及其权重，可以在其他类似的任务中重用。通过重用模块，开发者可以进行迁移学习。预训练的 Elmo 模型也出现在 [Tensorflow Hub](https://tfhub.dev/google/elmo/2) 上。

```
#Sample Code to get instant embedding
elmo = hub.Module("[https://tfhub.dev/google/elmo/2](https://tfhub.dev/google/elmo/2)", trainable=True)
embeddings = elmo(["apple juice", "apple tablet"],  signature="default", as_dict=True)["elmo"]
```

对于每个单词，嵌入的输出形状将是 1024，因此，在上面的例子中，对于每个都有两个单词的两个句子，形状为[2，2，1024]。如果使用 Keras 建模，则需要在`Lambda layer`中使用该模型。Elmo 模型权重已经下载到一个文件夹中，因此我们在训练时不需要对 Tensorflow hub 进行网络呼叫。

```
curl -L "[https://tfhub.dev/google/elmo/2?tf-hub-format=compressed](https://tfhub.dev/google/elmo/2?tf-hub-format=compressed)" | tar -zxvC /content/elmo_module
```

记得在调用 Elmo 嵌入层之前初始化图形和表格。

```
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())  
  sess.run(tf.tables_initializer())
  embed=sess.run(embeddings)
```

# 集成后的模型

查询分类模型

这里我们通过使用 python 的 [cachetools](https://pypi.org/project/cachetools/) 库来使用内存缓存。由于 Elmo 模型中的权重非常大，所以每次遍历图来寻找之前已经生成过一次的句子的嵌入并不是一个好主意，因为这将增加训练时间，并且我们还可能耗尽 RAM 存储器。一个时期后，所有搜索查询的嵌入将被缓存并轻松返回，而无需通过 TensorFlow hub 库。

在获得句子的嵌入后，我们使用 BiLSTM 来提取更多的上下文表示，用于词汇表外的单词，然后是两个密集层，最后是 sigmoid 激活层。由于这是一个多标签分类，并且一个查询属于三个不同的层次[，我们使用 Sigmoid 层](https://stats.stackexchange.com/questions/395934/activation-function-at-output-layer-for-multi-label-classification)，因为概率之和不需要为 1。这里，在这个例子中，我们有 1500 个类别，比方说 L1、L2 和 L3 每个层级有 500 个。为了显示每个层次结构中的前 2 个类别，我们需要从最后的 sigmoid 层中选择概率最高的类别。

## TF-服务响应

由于模型有点复杂，有大量的权重，这将影响 TF 服务的整体响应时间。它还依赖于最终被预测的类别的数量。模型大小也增长到大约 850MB。

Elmo 也可以用 [BERT](https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1) 代替。BERT 在 TF Hub 上有几个可用的变体。Elmo/BERT 模型可用于无监督学习，也可通过在 `[Keras](https://keras.io/layers/writing-your-own-keras-layers/)`中[创建一个自定义层在自定义数据集上进行训练。但是，请记住，在继续执行此任务之前，请确保您有足够的资源，如 GPU、内存等。](https://keras.io/layers/writing-your-own-keras-layers/)

# 结论

因此，现在我们可以预测特定搜索查询的分类层次结构，我们可以利用它向最终用户显示相同的内容，以便通过增强对引擎的搜索请求来选择和显示更精确的结果，这些结果包含用户想要购买的产品。由于这是 NLP 的 [ImageNet 时刻，像 BERT 这样基于转换器的语言建模算法在这个时代占据统治地位。其他语言模型有](http://ruder.io/nlp-imagenet/)[通用语言模型微调(ULMFiT)](https://arxiv.org/abs/1801.06146) 和 [OpenAI Transformer](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) ，后者在自然语言处理的各种任务上取得了最先进的水平。

感谢您阅读这篇文章。请在未来关注更多的更新，也请阅读其他的故事。