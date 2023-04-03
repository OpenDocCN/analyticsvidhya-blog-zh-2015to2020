# 在 Keras 中使用 Flask 和 LSTM 的电影分级预测服务器(第 1 部分)

> 原文：<https://medium.com/analytics-vidhya/movie-ratings-prediction-server-using-flask-and-lstm-in-keras-part-1-90933e8dbf88?source=collection_archive---------0----------------------->

![](img/cb3663828c09d4071c4cf55b842d9ee5.png)

深度学习是目前最热门的流行语之一，其理由在于它的大量应用以及它解决从计算机视觉到 NLP 等复杂问题的能力。在这一系列的帖子中，我将向您简要介绍如何使用深度学习来解决 NLP 问题。我们试图根据电影评论来预测电影的评分(满分 5 分)。在这一节中，我们将了解文本数据的基本预处理，这些预处理是为了使我们的数据可以用作 LSTM 模型的输入而必须完成的。

RNNs 是一类适合学习序列数据的神经网络。由于我们的数据由电影评论组成，我们肯定知道这是一种序列数据——因为一个词的情感取决于它之前或之后的词。例如，“这部电影很好”和“这部电影不好”描绘了相反的情绪，但为了找出这一点，我们必须跟踪“好”之前的单词。已经制定了 rnn 来解决这个具体问题。LSTM 只是 RNN 的一个变种，它在各种问题上被观察到比普通的 RNNs 给出更好的结果。关于 LSTM 的深入分析，请参考[这篇](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)牛逼的博客。

关于 RNN 的更多应用，请参考[这篇](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)博客。

# **预处理**

让我们从用 python 导入必要的模块开始。

```
import numpy as np
import pandas as pd
import refrom collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cross_validation import train_test_split
```

接下来，让我们定义从文本中提取干净标记的函数，将其添加到 python 中作为计数器存储的词汇表中，将计数器保存在文本文件中，加载该文本文件，最后根据加载的标记清理数据。我们还必须删除一些在默认 nltk 停用词中作为停用词出现的词。我们还添加了一些空格作为停用词，在清理数据后可以在语料库中找到。

```
# removing some words and adding some to increase accuracy
stopwords = stopwords.words('english')
newStopWords = ['', ' ', '  ', '   ', '    ', ' s']
stopwords.extend(newStopWords)
stopwords.remove('no')
stopwords.remove('not')
stopwords.remove('very')
stop_words = set(stopwords)def clean_doc(doc, vocab=None):
    tokens = word_tokenize(doc)
    # keeping only alphabets    
    tokens = [re.sub('[^a-zA-Z]', ' ', word) for word in tokens] 
    # converting to lowercase
    tokens = [word.lower() for word in tokens]
    # removing stopwords
    tokens = [w for w in tokens if not w in stop_words]
    # removing single characters if any
    tokens = [word for word in tokens if len(word) > 1]
    if vocab:
        tokens = [w for w in tokens if w in vocab]
        tokens = ' '.join(tokens)        
    return tokensdef add_doc_to_vocab(text, vocab):
    tokens = clean_doc(text)
    vocab.update(tokens)def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
```

数据集由 156060 行和 4 列组成，其中只有两列，即“短语”和“情感”对我们很重要。“短语”列包含各种短语，而“情感”列包含分配给相应短语的从 0 到 4 的情感编号。0 表示非常消极的情绪，4 表示非常积极的情绪。

加载数据集后，我们将它分成训练和测试数据，然后对加载的数据应用我们之前编写的各种函数。你可以在这里得到数据集[。](https://drive.google.com/file/d/1aw3cItu0Cs-54qoYa1CFB09wECNySDlN/view?usp=sharing)

```
df = pd.read_csv('train.tsv', delimiter='\t')
X = df['Phrase']
y = df['Sentiment']
y = np_utils.to_categorical(y)# splitting into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)# removing unnecessary data
del df, X, y# creating a vocabulary of words
vocab = Counter()
len_train = len(X_train)
for i in range(len_train):
    text = X_train.iloc[i]
    add_doc_to_vocab(text , vocab)print(len(vocab))
# print the 20 most common words
print(vocab.most_common(20))# removing tokens which occur less than 3 times.
min_occurance = 2
tokens = [k for k,c in vocab.items() if (c >= min_occurance & len(k) > 1)]# saving the vocabulary for futute use
save_list(tokens, 'vocab.txt')# loading the saved vocabulary
vocab = load_doc('vocab.txt')
vocab = vocab.split()
vocab = set(vocab)train_doc = []
for i in range(len_train):
    text = X_train.iloc[i]
    doc = clean_doc(text, vocab)
    train_doc.append(doc)test_doc = []
len_test = len(X_test)
for i in range(len_test):
    text = X_test.iloc[i]
    doc = clean_doc(text, vocab)
    test_doc.append(doc)
```

我们创建了一套词汇库，以消除重复。然后，我们使用这个集合来清理我们的数据，并准备将其输入到我们的 LSTM 模型中。

数据清理后，我发现某些行中没有标记，因为数据清理后每个标记都被删除了。这是我们在训练模型之前应该注意的事情。

```
# storing indexes where no tokens are present
index_train = []
for i in range(len(train_doc)):
    if len(train_doc[i]) == 0 :
        index_train.append(i)

index_test = []
for i in range(len(test_doc)):
    if len(test_doc[i]) == 0 :
        index_test.append(i)# dropping the unnecessary data
train_doc = np.delete(train_doc, index_train, 0)
test_doc = np.delete(test_doc, index_test, 0)
y_train = np.delete(y_train, index_train, 0)
y_test = np.delete(y_test, index_test, 0)
```

清理完所有数据后，我们必须将文本数据转换成适合 LSTM 模型的形式。为此，我们可以使用 Keras Tokenizer 类将数据转换成单词包模型。

```
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_doc)X_train = tokenizer.texts_to_matrix(train_doc, mode='binary')
X_test = tokenizer.texts_to_matrix(test_doc, mode='binary')
n_words = X_test.shape[1]
```

# 模型结构

现在是最后一步。让我们开始构建我们的模型，并在我们的测试数据中找到它的准确性。我使用双向 LSTM 作为我们的模型，因此该模型的性能甚至比普通的 LSTM 还要好。在双向 LSTM 中，在对当前输入进行预测时，会考虑以前和将来的输入。关于双向网络的详细解释，请参考这个[维基百科](https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks)页面。

```
# LSTM Model
model = Sequential()
model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(None,n_words)))
model.add(Dropout(0.2))
model.add(Dense(units=50, input_dim=100, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])# fitting the LSTM model
model.fit(X_train.reshape((-1, 1, n_words)), y_train, epochs=20, batch_size=100)# finding test loss and test accuracy
loss_rnn, acc_rnn = model.evaluate(X_test.reshape((-1, 1, n_words)), y_test, verbose=0)# saving model weights
model.model.save('rnn.h5')# loading saved weights
model_rnn = load_model('rnn.h5')
```

你可以通过使模型更深，并在每层中添加不同数量的神经元来进行实验。使用这个模型，我们可以得到大约 60-65%的准确率。我们还保存模型，以便我们可以进一步使用模型权重来进行预测，而无需为将来使用或当您想要在 web 应用程序上进行实时预测时进行培训。为了使用 LSTM 制作这样一个 web 应用程序，请继续关注本帖的第二部分[，我们将探索如何将你的模型转换成 web 应用程序。](/@animeshsharma97/movie-ratings-prediction-server-using-flask-and-lstm-in-keras-part-2-b0a84fb30ab7)

你可以在这里找到完整的实现。