# 用 Spacy 3.0 构建文本分类器

> 原文：<https://medium.com/analytics-vidhya/building-a-text-classifier-with-spacy-3-0-dd16e9979a?source=collection_archive---------1----------------------->

![](img/0cf3237de1c2264f3c5e830fb7d77194.png)

马库斯·温克勒在 [Unsplash](https://unsplash.com/s/photos/review?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

[*爆款 AI*](https://explosion.ai/) 刚刚为他们的自然语言处理工具包 [*SpaCy*](https://spacy.io/) 发布了全新的夜间版本。多年来，我一直是这个包的忠实粉丝，因为它允许快速开发，并且易于用来创建可以处理自然书写文本的应用程序。

过去有太多优秀的文章展示了 SpaCy 的 2.0+版本的 API。他们的 API 最近的变化也影响了大多数教程，现在新发布的 [Spacy V3](https://nightly.spacy.io/) 打破了这些教程。我喜欢这些变化，并想展示用很少几行代码训练一个文本分类器是多么简单。

第一步，我们需要安装一些软件包:

```
pip install spacy-nightly
pip install ml-datasets
python -m spacy download en_core_web_md
```

[*Ml-Datasets*](https://github.com/explosion/ml-datasets) 是一个来自 *Explosion AI* 的数据集管理库，它也提供了一种简单的方法来加载数据。我们将使用这个库获取数据来训练我们的分类器。

[](https://github.com/explosion/ml-datasets) [## 爆炸/ml-数据集

### 用于测试和示例脚本的各种机器学习数据集的加载器。thinc.extra.datasets 前情提要

github.com](https://github.com/explosion/ml-datasets) 

# 让我们建立一个分类器

完整代码也可以在 GitHub 资源库中找到:

[](https://github.com/p-sodmann/Spacy3Textcat) [## p-sodmann/Spacy3Textcat

### SpacyV3 文本分类器教程 GitHub 是超过 5000 万开发者的家园，他们一起工作来托管和审查…

github.com](https://github.com/p-sodmann/Spacy3Textcat) 

我们需要首先设置好一切:

```
import spacy# tqdm is a great progress bar for python
# tqdm.auto automatically selects a text based progress 
# for the console 
# and html based output in jupyter notebooks
from tqdm.auto import tqdm# DocBin is spacys new way to store Docs in a 
# binary format for training later
from spacy.tokens import DocBin# We want to classify movie reviews as positive or negative
# [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)
from ml_datasets import imdb# load movie reviews as a tuple (text, label)
train_data, valid_data = imdb()# load a medium sized english language model in spacy
nlp = spacy.load(“en_core_web_md”)
```

然后，我们需要将文本和标签转换成整洁的空间文档对象。

```
def make_docs(data):
    """
    this will take a list of texts and labels 
    and transform them in spacy documents

    data: list(tuple(text, label))

    returns: List(spacy.Doc.doc)
    """

    docs = []
    # nlp.pipe([texts]) is way faster than running 
    # nlp(text) for each text
    # as_tuples allows us to pass in a tuple, 
    # the first one is treated as text
    # the second one will get returned as it is.

    for doc, label in tqdm(nlp.pipe(data, as_tuples=True), total = len(data)):

        # we need to set the (text)cat(egory) for each document
        doc.cats["positive"] = label

        # put them into a nice list
        docs.append(doc)

    return docs
```

现在，我们只需要转换我们的数据，并将其作为二进制文件存储在光盘上。

```
# we are so far only interested in the first 5000 reviews
# this will keep the training time short.
# In practice take as much data as you can get.
# you can always reduce it to make the script even faster.
num_texts = 5000# first we need to transform all the training data
train_docs = make_docs(train_data[:num_texts])
# then we save it in a binary file to disc
doc_bin = DocBin(docs=train_docs)
doc_bin.to_disk("./data/train.spacy")# repeat for validation data
valid_docs = make_docs(valid_data[:num_texts])
doc_bin = DocBin(docs=valid_docs)
doc_bin.to_disk("./data/valid.spacy")
```

接下来，我们需要创建一个配置文件，告诉 SpaCy 应该从我们的数据中学到什么。
*爆炸 AI* 创建了一个快速制作基础配置文件的工具:[https://nightly.spacy.io/usage/training](https://nightly.spacy.io/usage/training#quickstart)

在我们的例子中，我们将在 components 下选择“Textcat”，在硬件中选择 CPU 优先，以及“Optimize-for”:效率。通常 *SpaCy* 会为每个参数提供相同的默认值。对于您的问题来说，它们不是最佳参数，但对于大多数数据来说，它们会工作得很好。
我们需要改变训练和验证数据的路径:

```
[paths]
train = "data/train.spacy"
dev = "data/valid.spacy"
```

下一步，我们需要将基本配置转变为完整配置。Spacy 将自动使用默认参数填充所有缺失值:

```
python -m spacy init fill-config ./base_config.cfg ./config.cfg
```

最后，我们可以在 CLI 中启动培训:

```
python -m spacy train config.cfg --output ./output
```

对于每个训练步骤，它将产生一个输出及其损失和准确性。损失告诉我们分类器的错误有多大，分数告诉我们二元分类正确的频率。

```
E #    LOSS TOK2VEC LOSS TEXTCAT CATS_SCORE SCORE
— — — — — — — — — — — — — — — — — — — — — — — — — 
0 0    0.00         0.25         48.82      0.49
2 5600 0.00         1.91         92.54      0.93
```

# 根据我们自己的输入运行分类器

训练好的模型保存在“输出”文件夹中。脚本完成后，我们可以加载“输出/模型-最佳”模型，并检查新输入的预测:

```
import spacy# load thebest model from training
nlp = spacy.load("output/model-best")text = ""
print("type : ‘quit’ to exit")# predict the sentiment until someone writes quit
while text != "quit":
    text = input("Please enter example input: ")
    doc = nlp(text)
    if doc.cats['positive'] >.5:
        print(f"the sentiment is positive")
    else:
        print(f"the sentiment is negative")
```

# 进一步的改进

![](img/04ed49e710b366475c5b08db87adde6a.png)

照片由[克莱门特·法利泽](https://unsplash.com/@centelm?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/further?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

我们没有为这个文本分类器使用任何预先训练的向量，我们可能不会得到评论“有多好”的可表示分数。我们将得到一个二元答案:文本输入的情绪是否大于 0.5，它被认为是积极的。

如果我们输入的文本与我们训练分类器所依据的数据不同，输出可能没有意义:

```
Please enter example input: i hate mondays
the sentiment is positive
```

## 改进分类器的步骤如下:

**1。在更多的数据上训练:**
我们只用了 5000 个文本，这只是整个语料库的五分之一。我们可以很容易地改变我们的脚本，以获得更多的例子。我们甚至可以尝试从不同的资源获取数据，或者自己给网站打分。

**2。训练更多的步骤:**
目前，我们的脚本要么在 1600 个训练步骤后停止，而没有在验证数据上找到更好的“解决方案”，要么在总共 20000 个步骤后停止。在我们的情况下，一个步骤是向前传递，进行预测，向后传递，校正神经网络，因此预测和标签之间的误差(损失)变得更小。我们可以增加值[耐心、最大步数和最大次数]，看看优化器是否能在稍后的训练中为我们的网络找到更好的权重。

**2。使用预先训练的单词向量。**
默认情况下，SpaCy 中的训练使用的是 Tok2Vec 层。它使用单词的长度等特征来动态生成向量。优点是它可以处理以前看不见的单词，并用数字表示出来。缺点是它的嵌入不代表它的意义。
预先训练的单词向量是从大量文本中导出的每个单词的数字表示，并试图将单词的意思嵌入到高维空间中。这有助于将语义相似的单词组合在一起。

**3。使用变压器模型。**
Transformer 是一种“较新”的架构，它将单词的上下文包含到它的嵌入中。SpaCy V3 现在支持像 [*Bert*](https://arxiv.org/abs/1810.04805) 这样的模型，这有助于进一步提升性能。

**4。检测输入中的异常值。我们在电影评论上训练我们的网络。这并不意味着模型可以告诉你一个烹饪食谱是好是坏。我们可能希望检查是否应该对输入数据进行预测，或者返回数据与定型相差太大而无法进行有意义的预测。Vincent D. Warmerdam 就此事做了一些很棒的演讲，比如“[如何约束人为的愚蠢](https://www.youtube.com/watch?v=Z8MEFI7ZJlA&ab_channel=PyData)”。**

我也期待着即将到来的 Ines 和 Sofie 的视频，这将为 SpaCy V3 的使用带来更多的见解。