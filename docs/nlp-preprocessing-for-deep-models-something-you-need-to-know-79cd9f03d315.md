# NLP——深度模型的预处理，一些你需要知道的关于填充的东西

> 原文：<https://medium.com/analytics-vidhya/nlp-preprocessing-for-deep-models-something-you-need-to-know-79cd9f03d315?source=collection_archive---------13----------------------->

自然语言处理在机器学习领域，特别是序列到序列学习，目前非常流行。互联网上有很多资源。然而，我写这篇文章是因为我很难理解文本是如何作为输入进入深度神经网络模式的。为了解决这个问题，我有很多解决方案，但是我寻找的解决方案没有提供太多信息。

在本文中，我将主要展示如何准备递归/卷积/前馈机器学习模型的输入。例如，我正在使用名为“[的 Kaggle 数据集，这是真的还是假的？带有灾难推文的 NLP](https://www.kaggle.com/c/nlp-getting-started)

所以，我们来看看资料

正如我们所看到的，有许多行和列带有文本。如果我们随机选择一列，我们可以看到

```'.https://t.co/gFJfgTodad'``[@挪威外交部](http://twitter.com/NorwayMFA)#巴林警察此前死于一场交通事故，他们并非死于爆炸

所以在这里我们可以看到？我们看到它包括标签，句子标点，和一些不会处理的网络链接。因此，首先我们将删除使用此代码的标点符号

```
import nltk
bOfWords = set(nltk.corpus.words.words())
def removeNonEng(s):
    return " ".join(w for w in nltk.wordpunct_tokenize(s) if w.lower() in bOfWords or not w.isalpha()) def removepunctuation(reviews):
    all_reviews=list()
    for text in reviews:      
      text = "".join([ch for ch in text if ch not in punctuation])
      text = text.lower()
      all_reviews.append(removeNonEng(text).split())
    return all_reviews
```

以后我们会用这种格式的句子。提到我在这里导入了一个自然语言处理语料库工具 [NLTK](https://www.nltk.org/data.html) ，它包含大约 23 万个英语单词。所以，如果你句子中的任何一个单词超出了英语单词的范围，我们就跳过这个单词。这是我们执行后的结果

```
['police', 'had', 'previously', 'in', 'a', 'road', 'accident', 'they', 'were', 'not', 'by', 'explosion']
```

我们再说一句

```
['i', 'still', 'have', 'not', 'church', 'of', 'coming', 'forward',
 'to', 'comment', 'on', 'the', 'accident', 'issue', 'and', 'disciplinary']
```

因此，我有两个句子作为演示，但在数据集中，我将有一堆句子，所有的标记都将从单词包中分离并丢弃。然而，我们仍然有深度模型不接受的文本。所以我们必须把这些文本转换成数字。那么，我们如何将文本转换成数字呢？

这里有 3 个最受欢迎策略

1.  [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
2.  嵌入层
3.  [手套](https://nlp.stanford.edu/projects/glove/)

在这里，我将展示如何使用 Word2Vec 和嵌入层来实现这一点，因为我尝试了这两种方法。所以，让我们从 Word2vec 开始

**Word2Vec**

Word2Vec 是一个预先训练好的模型，根据这个模型，如果你输入一个单词，它会生成这个单词相对于数十亿个单词和句子的向量值。你可以简单地从他们的网站上把它安装成一个包。我用的是 [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) ，因为它是最流行的一款。让我们看看这个预训练模型的参数是什么

```
model = Word2Vec(xClean, min_count=1, size= 50, workers=5, window =5, sg = 1)
```

这里我们可以看到，它包含 6 个参数，所以参数是

1.  xClean:您的标点符号删除数据集。
2.  min_count:一个单词出现的最少次数。
3.  大小:你的单词对相应的数字向量会有多大
4.  工人:流程
5.  窗口:考虑有多少前一个和后一个单词将被视为序列
6.  sg: CBoW /跳跃图

如果你对参数很不了解，请随意访问这个[网站](https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92)，希望你会有好的想法。

所以，现在我们可以根据我们的数据训练 word2Vec，保存模型。然而，到目前为止，我们已经训练了预训练模型，但是我们现在应该做什么呢？好了，现在让我们回到句子。

**要点**来了——如果我们看到 Word2Vec size 参数，它将是一个单词的 dimension ```[1*size]` `矢量化输出。例如，如果我们认为第一句话的第一个词是“警察”。因此，对于警察，只有 word2Vec 将生成[1*50]向量。然后，如果我们考虑整个句子:让我们计算单词的数量，12 个单词，这意味着对于整个句子，word2vec 将产生[12*50]矩阵，或者你可以考虑作为列表的列表。

对于数据集中的每一个句子，你可能都要做同样的事情。那么我们如何做到这一点呢？

```
encoded_docs = [[model.wv[word] for word in post] for post in xClean]
```

这里有一个快照，它将如何寻找一个单词。但是，如果任何单词超出了自然英语数据，它将调用和例外，如' noi '。

```
In: model[‘noise’]Out: 
array([ 0.24653986,  0.02332969,  0.11663471, -0.08650146,  0.03783561,
       -0.06231214,  0.01733457, -0.08161656, -0.03913275, -0.1288227 ,
        0.10245258, -0.07542043,  0.1505393 ,  0.03581995,  0.02998094,
        0.07354113, -0.08015608,  0.06226369,  0.03420312, -0.03641276,
        0.01748629, -0.09113364, -0.2961975 , -0.04497311,  0.05542335,
       -0.10479202, -0.07443704, -0.04653041, -0.14211805, -0.09488385,
        0.06362087,  0.00408071, -0.05070934, -0.03291701,  0.00774795,
       -0.2704018 , -0.0356291 , -0.10112489, -0.06312449,  0.13630798,
       -0.10666654,  0.06010102, -0.05819688, -0.19708745, -0.13682345,
       -0.12157825, -0.26685408, -0.1362774 , -0.13189617, -0.08612245],
      dtype=float32)In: model['noise'].shape
Out: (50,)
```

现在我们将得到 as List[List[List <int>]。现在，如果我们思考这些句子，我们会发现字数并不相等。S100 有 12 个字，S101 有 16 个字。因此，对应的矩阵将是[12*50]和[16*50]，这两个矩阵在维数上是不相等的。现在想想，如果我们想把这个矩阵输入到 LSTM 模型中，我们该如何解决这个问题？因为你不能根据你的数据定义一个变分模型。因此你必须对数据进行填充。</int>

应用 Word2Vec 数据的填充对我来说是完全陌生的，我在努力如何使向量相等。后来我发现这篇[文章](https://towardsdatascience.com/padding-sequences-with-simple-min-max-mean-document-encodings-aa27e1b1c781)真的很有帮助。因为每篇文章/教程都在说如何将单词转换成向量**，但是没有人说在将单词转换成向量之后，在输入之前，我们如何让它们等于输入模型。**

所以，现在我们必须使不等长的向量变成相等的向量长度，并且对于每个句子都是一致的。为此，我们可以定义一个这样的函数

```
import math
MAX_LENGTH =100padded_posts = []for post in encoded_docs:
    # Pad short posts with alternating min/max
    if len(post) < MAX_LENGTH:
        pointwise_avg = np.mean(post)
        padding = [pointwise_avg]
        post += padding * math.ceil((MAX_LENGTH - len(post) / 2.0))

    # Shorten long posts or those odd number length posts we padded to 100
    if len(post) > MAX_LENGTH:
        post = post[:MAX_LENGTH]    
    # Add the post to our new list of padded posts
    padded_posts.append(post)
```

在这里，我将 100 定义为最大的固定长度。因此，如果一个句子只有 12 或 17 个单词，它也将转换成 100 个，另一方面，如果任何一个句子超过 100 个单词，它将被截断并合并成 100 个。如果任何句子不能满足 100 个单词的平均值，它将填充空值

```
S101 = ['i', 'still', 'have', 'not', 'church', 'of', 'coming', 'forward',
 'to', 'comment', 'on', 'the', 'accident', 'issue', 'and', 'disciplinary'] = > Word2Vec => [17*50] andS100 = ['police', 'had', 'previously', 'in', 'a', 'road', 'accident', 'they', 'were', 'not', 'by', 'explosion'] = > Word2Vec => [12*50]After padding 
S101 = [100*50]
S100 = [100*50]
```

现在，所有维度都相同了，我们可以开始了，现在我们可以定义 100 个 LSTM 单元格，并通过 LSTM 单元格推送 Word2Vec 数据。

**嵌入层**

嵌入层来自深度学习库，如 PyTorch 或 Tensorflow。然而，仍然嵌入层不会接受直接的文本，我们必须以不同的方式处理。所以我们必须玩弄文字。因此，同样的方式，我们将删除所有的标点符号，并有干净的数据。我们必须应用一个热编码。那么，这是什么 OHE？如果你不知道，请仔细阅读这篇文章，它会让你的想法变得清晰。让我们试一试相同句子的热门代码-

```
from keras.preprocessing.text import one_hot
def oneHotEncode(docs):
    vocab_size = 50
    encoded_docs = [one_hot(d, vocab_size) for d in docs]
    return encoded_docs
xOneHot = oneHotEncode(xClean) print(xOneHot[100])
'[3, 1, 8, 48, 2, 16, 4, 41, 40, 25, 17, 3, 9, 22, 19, 45, 48, 1, 16, 42]'print(xOneHot[101])
[48, 30, 9, 9, 35, 9, 15, 11, 48, 24, 48, 6, 22, 49, 6, 25, 49, 23, 30, 38, 19]
```

正如我们在 OHE 之后看到的，它创造了一个单词序列，单词对应一个整数值。所以，对于两个不同的句子，它产生两个不同的向量。所以，现在我们也做填充。

```
from keras.preprocessing.sequence import pad_sequences
max_seq_length =25
def makePadded(xOneHot): 
    padded_docs = pad_sequences(xOneHot, maxlen=max_seq_length, padding='post')
    return padded_docs
```

所以这里每一个和每一个句子的列表长度都是相等的。所以填充后的列表将是类似的东西

```
S100 = [ 3,  1,  8, 48,  2, 16,  4, 41, 40, 25, 17,  3,  0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0], dtype=int32)
```

如您所见，在列表末尾添加了值为 0 的填充，因为我添加了 post 填充作为参数。前填充它就像[1*12]向量，后填充它是[1*25]向量。类似地，整个数据集将是 List[List[List <int>]，每个列表包含 25 个整数序列。现在你已经准备好给深度模型添加一个嵌入层了。</int>

```
model = Sequential()
model.add(Embedding(vocab_size, emb_out_len, input_length=max_seq_length))   #vocabsize, emb_out, inp len
model.add(LSTM(emb_out_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())model.fit(paddedMat, Y, epochs=EPOC, verbose=0)
loss, accuracy = model.evaluate(paddedMat, Y, verbose=0)
print('Accuracy: %f' % (accuracy*100))
```

到目前为止，我们已经回顾了如何使用 Word2Vec 和嵌入模型这两种不同的策略为自然语言处理准备数据。我希望你喜欢，虽然它很长。