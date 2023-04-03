# 训练时高效批处理 NLP 数据的 bucket_by_sequence_length API 教程。

> 原文：<https://medium.com/analytics-vidhya/tutorial-on-bucket-by-sequence-length-api-for-efficiently-batching-nlp-data-while-training-20d8ef5219d7?source=collection_archive---------2----------------------->

我第一次接触到[bucket _ by _ sequence _ length](https://www.youtube.com/watch?v=RIR_-Xlbp7s)API 是在 2017 time 7:00 tensor flow Dev summit 期间。
该 API 最初位于包“tf.contrib.training”中，现在它已被移动到“tf.data.experimental”中。
从那时起，该 API 经历了一些变化。我们将在本教程中讨论最新的 API。
本教程解释了使用 API 的必要性和重要性，随后展示了一个工作示例。

让我们开始吧。

考虑一下[讥讽检测](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection)的任务。数据看起来像这样

```
{
 “is_sarcastic”: 1,
 “headline”: “thirtysomething scientists unveil doomsday clock of hair loss”,
 “article_link”: “[https://www.theonion.com/thirtysomething-scientists-unveil-doomsday-clock-of-hai-1819586205](https://www.theonion.com/thirtysomething-scientists-unveil-doomsday-clock-of-hai-1819586205)"
}
```

一个字典对象，其中“is _ antonic”是我们的目标，“headline”是我们的特征。

在完整的数据集中，标题将具有不同的长度。这意味着训练一个模型来检测讽刺，我们将不得不使完整数据集中的“标题”的长度相同。这一步必须完成，因为模型的训练是分批进行的，每一批都应该具有相同的形状。

使所有“标题”数据相同的一个方法是填充。将所有标题数据填充到数据库中标题数据的最大长度。用'<pad>'标记填充文本数据。</pad>

这在某种程度上是可行的，但是这并不节省内存。让我们对数据集进行一些分析。

首先，使用下面的代码加载 JSON 数据。从 Kaggle 下载 v2 数据到你的本地文件夹。

```
df = pd.read_json(“./data/Sarcasm_Headlines_Dataset_v2.json”, lines=True)
```

让我们得到标题数据的最大长度。

```
print(‘maximum length of headline data is ‘, df.headline.str.split(‘ ‘).map(len).max())```
We receive result `maximum length of headline data is 151`
```

让我们得到标题数据的最小长度。

```
print(‘minimum length of headline data is ‘, df.headline.str.split(‘ ‘).map(len).min())`
We receive result `minimum length of headline data is 2`
```

现在让我们也得到标题文本数据长度的平均值。

```
print(‘mean of the lengths of the headline data is ‘, df.headline.str.split(‘ ‘).map(len).mean())`
The result is
`mean of the lengths of the headline data is 10.051853663650023`
```

从上面的数据可以看出，如果我们将每个标题数据填充到 151，我们将浪费大量的内存，对于少数数据，我们将拥有比实际单词更多的“<pad>”标记。</pad>

这将我们带到 API“bucket _ by _ sequence _ length ”,此方法更有效，因为它仍然将文本数据填充到相同的长度，但不是完整的数据集(即 151 个长度),而是针对单个批处理。这意味着一批中的每个标题长度相同，但每批的长度不同，这取决于该批中文本数据的最大长度。

我曾试图自己实现这个 API，但是我很难找到合适的文档和合适的例子。所以一旦我想通了，我想如果有更多的人知道并能在工作中使用它会更好。

如果您阅读 API 文档，它说它返回一个转换函数，可以从文档传递到“tf.data.Dataset.apply”.“一个数据集转换函数，可以传递到 tf.data.Dataset.apply。”

这意味着，首先，我们必须将数据帧转换为 TF . data . dataset。tensor flow 建议使用 tf.data.Dataset API，因为它针对输入管道进行了优化。我们可以在数据集上进行多次转换，但在其他时间可以在数据集上进行更多转换。

但是在开始之前，让我们先把文本转换成整数。为此，我们将使用 TensorFlow 标记器。我们将设置词汇大小、嵌入 _dim 以及词汇外令牌。我们也将设置 batch_size，但是稍后您会看到我们也可以进行动态批处理。

```
vocab_size = 1000
embedding_dim = 16
oov_tok = “<OOV>”
batch_size = 64
```

设置好以上参数后，让我们将文本转换成整数。

```
# Creating an instance of tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, lower=True)
# Creates and updates the internal vocabulary based on the text.
tokenizer.fit_on_texts(df.headline)# Add padding token.
tokenizer.word_index[‘<pad>’] = 0
tokenizer.index_word[0] = ‘<pad>’# Transforms the sentences to integers
sentences_int = tokenizer.texts_to_sequences(df.headline)
```

让我们把标签放在一个列表中。

```
labels = df.is_sarcastic.values
```

现在，让我们创建数据集，该数据集是为创建输入管道而推荐的。

```
 # Using generator for creating the dataset.
def generator():
  for i in range(0, len(sentences_int)):
    # creates x’s and y’s for the dataset.
    yield sentences_int[i], [labels[i]]# Calling the from_generator to generate the dataset.
# Here output types and output shapes are very important to initialize.
# the output types are tf.int64 as our dataset consists of x’s that are int as well as the labels that are int as well.
# The tensor shape for x is tf.TensorShape([None]) as the sentences can be of varied length.
# The tensorshape of y is tf.TensorShape([1]) as that consists of only the labels that can be either 0 or 1.
dataset = tf.data.Dataset.from_generator(generator, (tf.int64, tf.int64),
 (tf.TensorShape([None]), tf.TensorShape([1])))
```

我们的数据集已经准备好了，现在让我们使用 bucket_by_sequence_length API 来生成批处理，并根据我们将提供的上限 bucket 大小填充我们的句子。让我们创建不同桶的上部长度。我们可以随心所欲地创建存储桶。我建议首先分析数据集，以了解您可能需要的不同存储桶。

```
# These are the upper length boundaries for the buckets.
# Based on these boundaries, the sentences will be shifted to #different buckets.
boundaries = [df.headline.map(len).max() — 850, df.headline.map(len).max() — 700, df.headline.map(len).max() — 500,
 df.headline.map(len).max() — 300, df.headline.map(len).max() — 100, df.headline.map(len).max() — 50,
 df.headline.map(len).max()]
```

我们还必须为不同的存储桶提供 batch _ sizes。batch _ sizes 的长度应为 len(bucket_boundaries) + 1

```
batch_sizes = [batch_size] * (len(boundaries) + 1)
```

bucket_by_sequence_length API 还需要传递一个确定句子长度的函数。一旦 API 知道了句子的长度，就可以将它放入适当的桶中。在这里的理想场景中，您将创建不同大小的批，这取决于哪个存储桶包含的句子多还是少，但是这里我让所有存储桶的批大小保持不变。

```
# This function determines the length of the sentence.
# This will be used by bucket_by_sequence_length to batch them according to their length.
def _element_length_fn(x, y=None):
 return array_ops.shape(x)[0]
```

现在我们已经准备好了调用 bucket_by_sequence_length API 所需的所有参数，下面是我们对 API 的调用。

```
# Bucket_by_sequence_length returns a dataset transformation function that has to be applied using dataset.apply.
# Here the important parameter is pad_to_bucket_boundary. If this is set to true then, the sentences will be padded to
# the bucket boundaries provided. If set to False, it will pad the sentences to the maximum length found in the batch.
# Default value for padding is 0, so we do not need to supply anything extra here.
dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(_element_length_fn, boundaries,
 batch_sizes,
 drop_remainder=True,
 pad_to_bucket_boundary=True))
```

“边界”中的一个重要因素是拥有数据集句子的最大长度。如果你不知道，我会建议让 pad_to_bucket_boundary = False

一旦我们对数据集进行了适当的批处理和填充，使每个桶具有相同的形状，那么我们就可以分割数据集进行训练和测试。
我无法找到比这里提供的答案更好的分割数据集的解决方案—【https://stackoverflow.com/a/58452268/7220545 

```
# Splitting the dataset for training and testing.
def is_test(x, _):
 return x % 4 == 0def is_train(x, y):
 return not is_test(x, y)recover = lambda x, y: y# Split the dataset for training.
test_dataset = dataset.enumerate() \
 .filter(is_test) \
 .map(recover)# Split the dataset for testing/validation.
train_dataset = dataset.enumerate() \
 .filter(is_train) \
 .map(recover)
```

在这一步之后，我们已经为训练和验证准备好了数据集。模型的训练超出了本教程的范围，但是我在 [GitHub](https://github.com/PratsBhatt/sarcasm_detection_with_buckets) 上提供了演示模型训练的代码。
要运行该代码，您必须从 ka ggle—[https://www . ka ggle . com/RMI SRA/news-headlines-dataset-for-scara-detection](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection)下载数据集，并将其粘贴到。/data 文件夹。然后你就可以走了。

我希望你能愉快地阅读这篇文章，并希望它对你有用。