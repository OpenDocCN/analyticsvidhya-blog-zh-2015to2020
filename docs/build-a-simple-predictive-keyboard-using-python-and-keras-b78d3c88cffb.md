# 使用 python 和 Keras 构建一个简单的预测键盘

> 原文：<https://medium.com/analytics-vidhya/build-a-simple-predictive-keyboard-using-python-and-keras-b78d3c88cffb?source=collection_archive---------0----------------------->

![](img/2cff00107cf3fd521529447ef9816fb2.png)

照片由[凯特琳·贝克](https://unsplash.com/@kaitlynbaker?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

键盘是我们生活的一部分。我们在每个计算环境中都使用它。为了减少我们打字的工作量，今天大多数键盘都提供了先进的预测工具。它预测下一个字符，或下一个单词，甚至可以自动完成整个句子。因此，让我们讨论一些使用 python 中的 Keras 构建简单的下一个单词预测键盘应用程序的技术。本教程的灵感来自于[维尼林·瓦尔科夫](https://medium.com/u/102e34a0beb1?source=post_page-----b78d3c88cffb--------------------------------)在[下一个字符预测键盘](/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218)上写的博客。

![](img/f40b81dffca206fdb0ead751e66f500a.png)

下一个单词的预测

为此，我们使用递归神经网络。选择这个模型是因为它提供了一种检查先前输入的方法。LSTM，一种特殊的 RNN 也用于此目的。LSTM 提供了保存误差的机制，这些误差可以通过时间和层反向传播，这有助于减少[消失梯度](https://en.wikipedia.org/wiki/Vanishing_gradient_problem##targetText=In%20machine%20learning%2C%20the%20vanishing,based%20learning%20methods%20and%20backpropagation.&targetText=The%20problem%20is%20that%20in,weight%20from%20changing%20its%20value.)问题。

## 我们来编码吧！

首先，我们需要安装几个库。

```
pip install numpy
pip install tensorflow
pip install keras
pip install nltk
```

现在让我们导入所需的库。

```
import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq
```

**加载数据集**是下一个要做的重要步骤，这里我们使用 [*《福尔摩斯探案集》*](https://www.gutenberg.org/files/1661/1661-0.txt) *作为数据集。*

```
path = '1661-0.txt'
text = open(path).read().lower()
print('corpus length:', len(text))**Output** corpus length: 581887
```

现在，我们希望将整个数据集按顺序拆分成每个单词，而不要出现特殊字符。

```
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)**Output**
['project', 'gutenberg', 's', 'the', 'adventures', 'of', 'sherlock', 'holmes', 'by', ............................... , 'our', 'email', 'newsletter', 'to', 'hear', 'about', 'new', 'ebooks']
```

接下来，对于特征工程部分，我们需要有唯一的排序单词列表。我们还需要一个字典(<key: value="">)，将 unique_words 列表中的每个单词作为键，将其对应的位置作为值。</key:>

```
unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))
```

**特色工程**

根据维基百科的说法，**特征工程**是使用数据的[领域知识](https://en.wikipedia.org/wiki/Domain_knowledge)来创建[特征](https://en.wikipedia.org/wiki/Feature_(machine_learning))的过程，这些特征使[机器学习](https://en.wikipedia.org/wiki/Machine_learning)算法工作。特征工程是机器学习应用的基础，既困难又昂贵。

我们定义了一个单词长度，这意味着前一个单词的数量决定了下一个单词。此外，我们创建一个名为 prev_words 的空列表来存储一组五个前面的单词及其在 next_words 列表中对应的下一个单词。我们通过在小于单词长度的范围内循环来填充这些列表。

```
WORD_LENGTH = 5
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])
print(prev_words[0])
print(next_words[0])**Output** ['project', 'gutenberg', 's', 'the', 'adventures']
of
```

现在，是生成特征向量的时候了。为了生成特征向量，我们使用**一键编码**。

解释:一键编码

这里，我们创建两个 numpy 数组 X(用于存储特性)和 Y(用于存储相应的标签(这里是下一个单词))。我们迭代 X 和 Y，如果单词存在，那么相应的位置为 1。

```
X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1
```

让我们看一个单一的序列:

```
print(X[0][0])Output
[False False False … False False False]
```

## 构建模型

我们使用具有 128 个神经元的单层 LSTM 模型、全连接层和用于激活的 softmax 函数。

```
model = Sequential()
model.add(LSTM(128, input_shape=(WORD_LENGTH, len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))
```

## **培训**

该模型将用 RMSprop 优化器用 20 个时期来训练。

```
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history
```

训练成功后，我们将保存训练好的模型，并在需要时重新加载。

```
model.save('keras_next_word_model.h5')
pickle.dump(history, open("history.p", "wb"))model = load_model('keras_next_word_model.h5')
history = pickle.load(open("history.p", "rb")) 
```

**评估**

模型在训练成功后输出训练评估结果，我们也可以从历史变量中访问这些评估。

```
{‘val_loss’: [6.99377903472107, 7.873811178441364], ‘val_accuracy’: [0.1050897091627121, 0.10563895851373672], ‘loss’: [6.0041207935270124, 5.785401324014241], ‘accuracy’: [0.10772078, 0.14732216]}# sample evaluation ---- # only 2 epochs
```

**预测**

现在，我们需要使用这个模型来预测新单词。为此，我们输入样本作为特征向量。我们将输入字符串转换成一个单一的特征向量。

```
def prepare_input(text):
    x = np.zeros((1, WORD_LENGTH, len(unique_words)))
    for t, word in enumerate(text.split()):
        print(word)
        x[0, t, unique_word_index[word]] = 1
    return xprepare_input("It is not a lack".lower())**Output**
array([[[ 0.,  0.,  0., ...,  0.,  0.,  0.],
        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
        ..., 
        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
        [ 0.,  0.,  0., ...,  0.,  0.,  0.]]])
```

在通过样本函数从模型中进行预测之后，选择最佳可能的 n 个单词。

```
def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)
```

最后，对于预测，我们使用函数 predict_completions，它使用模型来预测并返回 n 个预测单词的列表。

```
def predict_completions(text, n=3):
    if text == "":
        return("0")
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [unique_words[idx] for idx in next_indices]
```

现在让我们看看它是如何预测的，我们使用 tokenizer.tokenize 来删除标点符号，我们还选择了前 5 个单词，因为我们的预测基于前 5 个单词。

```
q =  "Your life will never be the same again"
print("correct sentence: ",q)
seq = " ".join(tokenizer.tokenize(q.lower())[0:5])
print("Sequence: ",seq)
print("next possible words: ", predict_completions(seq, 5))**Output**
correct sentence:  Your life will never be the same again
Sequence:  your life will never be
next possible words:  ['the', 'of', 'very', 'no', 'in']
```

**弊端**

*   这里，在准备唯一的单词时，我们只从输入数据集中收集唯一的单词，而不是从英语词典中收集。因为这个原因，很多都被忽略了。(要创建如此大的输入集(根据 nltk，英语词典包含约 23000 个单词，我们需要执行**批处理)**

**参考文献**

*   [tensor flow 中的递归网络](https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/)
*   [使用递归神经网络制作预测键盘](/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218)
*   [递归神经网络的不合理有效性](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
*   [了解 LSTM 网络](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
*   [如何用 Python 实现 RNN](https://peterroelants.github.io/posts/rnn_implementation_part01/)
*   [用于情感分析的 LSTM 网络](http://deeplearning.net/tutorial/lstm.html)
*   [cs231n —递归神经网络](http://cs231n.stanford.edu/slides/2016/winter1516_lecture10.pdf)