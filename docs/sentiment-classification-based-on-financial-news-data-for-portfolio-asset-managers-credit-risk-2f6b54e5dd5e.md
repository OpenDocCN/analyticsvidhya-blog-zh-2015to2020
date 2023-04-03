# 投资组合/资产经理和信用风险官基于金融新闻数据的情绪分类

> 原文：<https://medium.com/analytics-vidhya/sentiment-classification-based-on-financial-news-data-for-portfolio-asset-managers-credit-risk-2f6b54e5dd5e?source=collection_archive---------10----------------------->

![](img/faea13ac3be5446fbd9fa35b0396bd6e.png)

图片来源:miro.medium.com

投资组合经理和信用风险管理人员的工作相当令人生畏。他们在任何时候都需要对当前的市场状况保持乐观。这些投资银行专业人士被认为是股票/工具/交易对手/发行人的技术和基本面分析专家，或者至少擅长使用各种 COTS(商业现货)工具来生成数字/指标并识别市场模式，以帮助指导他们的投资决策。

有了这些数据，投资组合经理就可以决定在他们的投资组合中添加/删除工具，因为信贷员设定/调整交易限额或向他们的投资银行有兴趣与之交易的交易对手提供贷款。这是金融工作家族中最突出的工作之一，因为即使是一个错误的评估也很容易产生牵强的影响，导致投资者或银行损失大量投资者的资金——这对银行或金融机构来说是一场噩梦。

让我们为投资银行家的工具箱添加另一个工具——这个工具将帮助他们确定他们目前正在研究的股票/交易对手的市场情绪。

在本文的其余部分，我们将专注于创建一个深度神经网络模型，该模型将接受一个文本体(预计是一篇金融新闻文章或 Twitter 讨论的摘录，甚至是一封包含金融内容的电子邮件！)并将其转化为市场情绪(而不需要手动逐字阅读文本)。

该模型的编码将包括以下步骤:

1.下载财经新闻情感数据；希望有人已经为我们收集了这些数据；

2.选择用于将文本数据转换成嵌入向量的单词嵌入；我们可以使用 word2vec 模型从头开始创建这个嵌入，但是我们将把这部分留给另一篇文章；现在我们将重用互联网上免费提供的一个嵌入。

3.预处理文本数据；在将输入输入到我们的模型之前，我们需要将文本转换成数字。

4.设计一个由输入层、隐层和输出层组成的定制神经网络；我们将在隐藏层中使用 LSTM，因为文本数据是连续的，而 LSTM 层是处理连续数据的理想选择。GRU 层是另一种选择，我们可以选择。你可以自由地尝试自己的 GRU 层。

5.基于财经新闻情感数据训练网络。

6.预测我们选择的金融文本的情绪。

在下一节中，我将更详细地介绍上述步骤，并在需要的地方内嵌代码片段。该代码也可从以下 Github 链接获得:

[https://github . com/rohitar/my projects/tree/master/counter party-情操](https://github.com/rohitar/myprojects/tree/master/counterparty-sentiment)

**下载财经新闻舆情数据**

与任何机器学习模型一样，我们需要数据来训练我们的模型。因为我们想衡量金融文本的极性，我们需要既包含金融文本又包含其相应极性的数据。幸运的是，这些数据已经存在(由 Pekka Malo 等人发表),您可以从这里免费下载:

[https://www . research gate . net/publication/251231364 _ FinancialPhraseBank-v10](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10)

**选择文字嵌入**

我们可以选择从头开始构建整个单词嵌入，但是当有人已经在大量文本数据上运行严格的训练来生成嵌入时，我们不想重新发明轮子。

那些不理解嵌入的人——这是一个简单的映射，将词汇表中的所有单词映射到它们对应的向量表示(在 n 维向量空间中——在创建嵌入时可以定义维度)。通常，word2vec 模型用于使用 Skipgram 方法或连续单词包(CBOW)方法来创建单词嵌入。这些单词嵌入捕获单词之间的语义关系，并且相似的单词将在向量空间中聚集在一起。

其中一个可以免费下载的预训练单词嵌入是基于包含 300 万个单词的谷歌新闻，并且已经从谷歌新闻档案中训练了大约 1000 亿个单词(你可以想象训练这些嵌入将是多么艰巨的任务)。另一个流行的单词嵌入选项是脸书训练的 fastText。

我们将在磁盘上下载 Google 新闻嵌入，并将使用 Gensim 库将这些嵌入加载到内存中(注意，这将消耗大约 5 GB 的内存，因此请确保您运行该模型的机器有足够的 RAM！).以下代码行将下载嵌入内容并将其加载到内存中(由 Douwe Osinga 提供—深度学习食谱):

```
MODEL = 'GoogleNews-vectors-negative300.bin'
path = get_file(MODEL + '.gz', 'https://s3.amazonaws.com/dl4j-distribution/%s.gz' % MODEL)if not os.path.isdir('generated'):
    os.mkdir('generated')unzipped = os.path.join('generated', MODEL)
if not os.path.isfile(unzipped):
    with open(unzipped, 'wb') as fout:
        zcat = subprocess.Popen(['zcat'],
                          stdin=open(path),
                          stdout=fout
                         )
        zcat.wait()from gensim.models import KeyedVectors
word2vec = KeyedVectors.load_word2vec_format(unzipped, binary=True)
```

如果需要，您可以通过使用 load_word2vec_format 方法中的“limit”参数来减少加载到 RAM 中的单词向量的数量，但是这可能会对您构建的最终模型的准确性产生影响。

**预处理文本数据**

注意，任何深度学习或机器学习模型都不理解文本数据；模型只理解数字。因此，我们需要将输入文本中的单词标记为数字形式。我们将使用 Keras 库适当地标记输入金融文本中的字符串；本质上，我们将把输入数据中的每个单词映射到一个唯一的数字标记，最后使用这个映射把输入文本转换成数字串。这方面的代码如下所示:

```
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequencestokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)X = pad_sequences(sequences, maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)
y = to_categorical(labels)
```

接下来，我们需要将分配给每个单词的标记转换成它们各自的嵌入向量。为此，我们需要在标记输入数据时，使用上面创建的单词索引来重新索引预训练的 Gensim 嵌入。下面给出了这样做的代码(由 Antonio Gulli 提供 Keras 的深度学习):

```
embedding_weights = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, index in word_index.items():
  try:
    embedding_weights[index,:] = word2vec[word]
  except KeyError:
    pass
```

**设计神经网络**

现在终于到了定义神经网络的时候了。我们将使用以下代码来定义我们的神经网络:

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTMmodel = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH, weights=[embedding_weights]))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(64, activation="relu"))
model.add(Dense(3, activation="softmax"))
```

请注意，上面的嵌入层使用了我们之前创建的变换后的嵌入权重。然后，矢量化输入将通过几个 LSTM 层和一个完全连接的密集层。

在高层次上，LSTM 层包括长期存储单元和短期存储单元。长期单元本质上捕捉文本主体中距离较远的单词的关联，而短期单元捕捉看起来距离较近的单词的关联。LSTM 是一个复杂的话题，因此详细的讨论超出了本文的范围。

最后，来自完全连接的密集层的数据被传递到 3 个类的“softmax”输出。这 3 个类将指定文本极性，即。积极、中立或消极。

太好了！现在我们已经定义了神经网络，我们将跳到下一步。

**训练网**

这一步将涉及使用我们在第一步中下载的金融新闻情绪数据来训练网络。然而，我们不想在整个数据上训练模型；我们希望保留一些数据，这些数据对模型来说是不可见的，以便以后可以用来评估我们训练好的模型。为此，我们将使用以下代码将数据分为训练和测试，测试规模为 30%:

```
from sklearn.model_selection import train_test_splitX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
```

接下来，我们将编译模型，并随后根据测试数据对其进行拟合。为此，我们将使用以下代码:

```
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_test, y_test))
```

请注意，我们选择了“亚当”优化器，因为众所周知，它最适合基于序列的模型，如 LSTM。损失标准被定义为分类交叉熵，用于多类分类问题，如我们的问题。我建议在 GPU 机器上运行训练——因为训练的速度预计比在 CPU 上快 10 倍。如果您没有物理 GPU，可以考虑使用 Google Colab(基于云的 GPU)。

在运行 20 个时期的训练之后，您将获得大约 99%的测试准确率& 75%的验证准确率——这确实是一个非常好的开始。因此，你现在有了一个可靠的模型，可以根据输入的金融文本预测市场情绪。

我鼓励你尝试不同的模型参数/模型架构来进一步提高准确性。此外，由于训练和测试准确性之间的差距很大，该模型似乎有点过度拟合——我建议您尝试正则化技术，如辍学。

**情绪预测**

现在是时候带着你的模型上路了。使用以下代码，通过输入您选择的金融文本来产生市场情绪。

```
pred_sentences = ["<input the financial text of your choice>"]
pred_sequences = tokenizer.texts_to_sequences(pred_sentences)
X_pred = pad_sequences(pred_sequences, maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)
y_pred = model.predict(X_pred)
y_pred
```

你相信情绪预测的神圣性吗？如果是的话，现在是时候围绕我们刚刚开发的模型设计一个用户界面，并将其发送给你的信贷人员/投资组合经理，使他们能够使用该模型做出更好的投资决策。

我真心希望你喜欢这篇文章——如果你能在下面留下评价/反馈，我将不胜感激。

学分:

1.  由 Anuj Kumar(【https://www.linkedin.com/in/anujchauhan/】T2)合著
2.  概念:Simarjit Singh Lamba 和 Rohit Arora
3.  佩卡·马洛等人发表的财经新闻情感数据集
4.  《深度学习食谱》
5.  Antonio Gulli——Keras 深度学习
6.  霍布森·莱恩等——自然语言处理在行动
7.  图片-来源于 miro.medium.com([https://miro . medium . com/max/1196/1% 2 altpcct 2 mya 2-edjdti 8s 4 q . JPEG](https://miro.medium.com/max/1196/1%2aLTpcCt2mYa2-edjdti8S4Q.jpeg))