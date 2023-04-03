# 了解 Keras 中的嵌入层

> 原文：<https://medium.com/analytics-vidhya/understanding-embedding-layer-in-keras-bbe3ff1327ce?source=collection_archive---------0----------------------->

在深度学习中，嵌入层听起来像一个谜，直到你抓住它。由于嵌入层是神经网络的重要组成部分，所以理解它的工作原理是很重要的。在这篇文章中，我将尝试解释什么是嵌入层，它的需求是什么，它是如何工作的，以及一些编码示例。所以让我们开始吧。

## **什么是嵌入层**

嵌入层是 Keras 中可用的层之一。这主要用于自然语言处理相关的应用，如语言建模，但它也可以用于涉及神经网络的其他任务。在处理 NLP 问题时，我们可以使用预先训练的单词嵌入，如 GloVe。或者，我们也可以使用 Keras 嵌入层来训练我们自己的嵌入。

## **需要嵌入**

> 词嵌入可以被认为是一种与降维一起的一键编码的替代方案。

正如我们所知，在处理文本数据时，我们需要在输入任何机器学习模型(包括神经网络)之前将其转换为数字。为简单起见，单词可以比作范畴变量。我们使用一次性编码将分类特征转换成数字。为此，我们为每个类别创建虚拟特征，并用 0 和 1 填充它们。

类似地，如果我们对文本数据中的单词使用一次性编码，我们将为每个单词创建一个虚拟特征，这意味着 10，000 个单词的词汇表有 10，000 个特征。这不是一种可行的嵌入方法，因为它需要用于单词向量的大存储空间，并且降低了模型效率。

嵌入层使我们能够将每个单词转换成定义大小的固定长度向量。结果向量是一个密集的向量，具有真实的值，而不仅仅是 0 和 1。固定长度的单词向量有助于我们以更好的方式表示单词，同时降低维数。

这种嵌入层的工作方式就像一个查找表。单词是这个表中的键，而密集的单词向量是值。为了更好地理解它，我们来看看 Keras 嵌入层的实现。

## **在 Keras 实施**

让我们从导入所需的库开始。

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import numpy as np
```

我们可以通过添加一个嵌入层来创建一个简单的 Keras 模型。

```
model = Sequential()
embedding_layer = Embedding(input_dim=10,output_dim=4,input_length=2)
model.add(embedding_layer)
model.compile('adam','mse')
```

嵌入层有三个参数

*   **input_dim** :词汇的大小
*   **output_dim** :每个单词的向量长度
*   **输入长度**:序列的最大长度

在上面的例子中，我们将 10 设置为词汇大小，因为我们将对数字 0 到 9 进行编码。我们希望字向量的长度为 4，因此 output_dim 被设置为 4。嵌入层的输入序列的长度将是 2。

现在，让我们将一个样本输入传递给我们的模型，看看结果。

```
input_data = np.array([[1,2]])
pred = model.predict(input_data)
print(input_data.shape)
print(pred)
```

上述代码的输出如下。

```
(1, 2)
[[[ 0.04502351  0.00151128  0.01764284 -0.0089057 ]
  [-0.04007018  0.02874336  0.02772436  0.00842067]]]
```

如您所见，每个单词(1 和 2)都由一个长度为 4 的向量表示。如果我们打印嵌入层的权重，我们得到下面的结果。

```
[array([[-0.04333381, -0.02326865, -0.00812379,  0.02167496],
        [ 0.04502351,  0.00151128,  0.01764284, -0.0089057 ],
        [-0.04007018,  0.02874336,  0.02772436,  0.00842067],
        [ 0.00512743,  0.03695237, -0.02774147, -0.03748262],
        [ 0.02066498, -0.01512628, -0.03989452,  0.00809463],
        [-0.02207369,  0.02889762, -0.01229819, -0.03157005],
        [ 0.02565557,  0.02931032, -0.01611946, -0.00105535],
        [ 0.03920721,  0.04009463, -0.04943105,  0.04145898],
        [ 0.04208959, -0.00412361, -0.04585704,  0.03489918],
        [-0.04016889,  0.03448426,  0.00623332,  0.02844917]],
       dtype=float32)]
```

这些权重基本上是词汇中单词的向量表示。正如我们之前讨论的，这是一个大小为 10 x 4 的查找表，用于单词 0 到 9。第一个单词(0)由该表中的第一行表示，即

```
[-0.04333381, -0.02326865, -0.00812379,  0.02167496]
```

**注意:**在这个例子中，我们没有训练嵌入层。分配给单词向量的权重被随机初始化。

这是一个很好的例子。但是在处理实际的文本数据时，我们需要训练嵌入层来获得正确的单词嵌入。让我们看看如何使用餐馆评论数据来做到这一点。

## **餐厅点评分类**

在解决这个问题时，我们将执行以下步骤。

1.  把句子标记成单词。
2.  为每个单词创建一个独热编码向量。
3.  使用填充以确保所有序列长度相同。
4.  将填充的序列作为输入传递给嵌入层。
5.  展平并应用密集层来预测标签。

我们从导入所需的库开始

```
from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Embedding,Dense
```

为了简单起见，我们将使用总共 10 个评论。其中一半是正的，用 0 表示，另一半是负的，用 1 表示。

```
# Define 10 restaurant reviews
reviews =[
          'Never coming back!',
          'horrible service',
          'rude waitress',
          'cold food',
          'horrible food!',
          'awesome',
          'awesome services!',
          'rocks',
          'poor work',
          'couldn\'t have done better'
]#Define labels
labels = array([1,1,1,1,1,0,0,0,0,0])
```

我们将把词汇量定为 50，并使用 Keras 的 one_hot 函数对单词进行 one-hot 编码。

```
Vocab_size = 50
encoded_reviews = [one_hot(d,Vocab_size) for d in reviews]
print(f'encoded reviews: {encoded_reviews}')
```

我们将得到如下编码审查的结果。

```
encoded reviews: [[18, 39, 17], [27, 27], [5, 19], [41, 29], [27, 29], [2], [2, 1], [49], [26, 9], [6, 9, 11, 21]]
```

这里你可以看到每个编码评论的长度等于评论中的字数。Keras one_hot 基本上是将每个单词转换成它的 one-hot 编码索引。现在，我们需要应用填充，以便所有编码的评论长度相同。让我们将 4 定义为最大长度，并在编码向量的最后填充 0。

```
max_length = 4
padded_reviews = pad_sequences(encoded_reviews,maxlen=max_length,padding='post')
print(padded_reviews)
```

填充和编码的评论将是这样的。

```
[[18 39 17  0]
 [27 27  0  0]
 [ 5 19  0  0]
 [41 29  0  0]
 [27 29  0  0]
 [ 2  0  0  0]
 [ 2  1  0  0]
 [49  0  0  0]
 [26  9  0  0]
 [ 6  9 11 21]]
```

在创建了评论的填充的一键表示之后，我们准备将它作为输入传递给嵌入层。在下面的代码片段中，我们创建了一个简单的 Keras 模型。我们将每个单词的嵌入向量的长度固定为 8，输入长度将是我们已经定义为 4 的最大长度。

```
model = Sequential()
embedding_layer = Embedding(input_dim=Vocab_size,output_dim=8,input_length=max_length)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])print(model.summary())
```

模型摘要将是。

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 4, 8)              400       
_________________________________________________________________
flatten (Flatten)            (None, 32)                0         
_________________________________________________________________
dense (Dense)                (None, 1)                 33        
=================================================================
Total params: 433
Trainable params: 433
Non-trainable params: 0
_________________________________________________________________
None
```

接下来，我们将对模型进行 100 个纪元的训练。

```
model.fit(padded_reviews,labels,epochs=100,verbose=0)
```

一旦训练完成，嵌入层已经学习了权值，它只不过是每个单词的矢量表示。让我们检查权重矩阵的形状。

```
print(embedding_layer.get_weights()[0].shape)
```

这个嵌入矩阵本质上是一个 50 行 8 列的查找表，从输出可以明显看出。

```
(50, 8)
```

如果我们检查第一个单词的嵌入，我们得到下面的向量。

```
[ 0.056933    0.0951985   0.07193055  0.13863552 -0.13165753  0.07380469    0.10305451 -0.10652688]
```

这就是我们如何在我们的文本语料库上训练嵌入层，并获得每个单词的嵌入向量。这些向量然后被用来表示句子中的单词。

## **结论**

嵌入是处理 NLP 问题的好方法，原因有二。首先，由于我们可以控制特征的数量，它有助于减少一键编码的维数。第二，它能够理解单词的上下文，使得相似的单词具有相似的嵌入。这篇文章详细解释了单词嵌入的工作原理。

如果你觉得这篇文章有用，请在评论中告诉我。我是一名数据科学爱好者和博客作者。你可以通过我的 LinkedIn [个人资料](https://www.linkedin.com/in/sawan-saxena-640a4475/)联系我。

感谢阅读。

**参考文献**

*   杰夫·希顿的 Keras (11.3)中的嵌入层是什么:[https://www.youtube.com/watch?v=OuNH5kT-aD0](https://www.youtube.com/watch?v=OuNH5kT-aD0)。