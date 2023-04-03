# 使用文本的情感识别

> 原文：<https://medium.com/analytics-vidhya/song-recommendation-based-on-textual-and-facial-emotion-recognition-a95e6259c5d8?source=collection_archive---------1----------------------->

![](img/df7dfae1624d570585906fe7d5c9243a.png)

照片由 [**滕亚尔**](https://unsplash.com/@tengyart?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上[下**上**下](https://unsplash.com/s/photos/emotions?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

在这篇文章中，我们将探索编写一个使用文本特征检测情绪的程序所需的步骤。

![](img/687dc7b63b23ede01035b3e8d6750d93.png)

程序是用 Python**假装震惊** 写的。

现在让我们看看如何着手创建一个文本模型。

![](img/ef04db588c54eb0abc45b4340db3b935.png)

对于文本模型的创建，我们将使用[**【LSTM(长短期记忆)**](https://en.wikipedia.org/wiki/Long_short-term_memory) ，因为与其他学习模型如 SVM、随机福里斯特、朴素贝叶斯等相比，它给出了更高的训练精度。

最重要的是，你需要一个数据集来训练和测试，你可以从 [**这里**](https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp) 得到。

首先，我们导入所需的包。如果你没有安装这些软件包，你可以 [**pip**](https://pip.pypa.io/en/stable/) 安装它们。

```
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
import urllib.request
import zipfile
import os
from keras.models import Sequential
from keras.layers import Embedding,Bidirectional,LSTM,GRU,Dense
import nltk
from nltk.tokenize import word_tokenize
import warnings
import tensorflow as tf
nltk.download('punkt')
warnings.filterwarnings('ignore')
```

既然已经导入了包，那么您需要提取句子和它们各自的情感，并将它们分别插入到训练、测试和验证数据框架中。

因为文件在里面。txt 格式，我们使用下面的代码将它们放入训练和测试(包括验证)数据帧中。

```
f=open('train.txt','r')
x_train=[]
y_train=[]
for i in f:
    l=i.split(';')
    y_train.append(l[1].strip())
    x_train.append(l[0])
f=open('test.txt','r')
x_test=[]
y_test=[]
for i in f:
    l=i.split(';')
    y_test.append(l[1].strip())
    x_test.append(l[0])
f=open('val.txt','r')
for i in f:
    l=i.split(';')
    y_test.append(l[1].strip())
    x_test.append(l[0])
data_train=pd.DataFrame({'Text':x_train,'Emotion':y_train})
data_test=pd.DataFrame({'Text':x_test,'Emotion':y_test})
data=data_train.append(data_test,ignore_index=True)
```

现在句子已经插入，我们需要清理它们。基本上，我们去掉所有的介词、冠词、标点符号、停用词，只留下句子中重要的词。

这里，被去除的单词充当噪声，这就是为什么为了获得期望的结果，即高测试精度，必须消除它们的原因。

```
def clean_text(*data*):
    data=re.sub(r"(#[\d\w\.]+)", '', data)
    data=re.sub(r"(@[\d\w\.]+)", '', data)
    data=word_tokenize(data)
    return data
texts=[' '.join(clean_text(text)) for text in data.Text]
texts_train=[' '.join(clean_text(text)) for text in x_train]
texts_test=[' '.join(clean_text(text)) for text in x_test]
```

[**标记化**](https://www.tutorialspoint.com/python_text_processing/python_tokenization.htm) 是自然语言处理分析中的一个重要过程。它标记每个句子，提取每个独特的词，并创建一个字典，其中每个独特的词被分配一个索引。

```
tokenizer=Tokenizer()
tokenizer.fit_on_texts(texts)
sequence_train=tokenizer.texts_to_sequences(texts_train)
sequence_test=tokenizer.texts_to_sequences(texts_test)
index_of_words=tokenizer.word_index
vocab_size=len(index_of_words)+1
```

我们得到的数据集有六种独特的结果或情绪，即:愤怒、悲伤、恐惧、喜悦、惊讶和爱。

因此，类的数量是六个。此外，这里我们使用 300 个嵌入维度，序列的最大长度被赋值为 500。

当我们填充训练和测试序列时，必须将它们的“maxlen”参数设置为相同的值，否则将显示错误。

在后面的阶段，每种情绪都被赋予一个[](https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical)**(0-5)。因此，使用了“编码”字典和“to _ categorical”函数。当我们试图获得一个结果时，我们将类别值映射回情感。**

```
num_classes=6
embed_num_dims=300
max_seq_len=500
class_names=['anger','sadness','fear','joy','surprise','love']X_train_pad=pad_sequences(sequence_train,maxlen=max_seq_len)
X_test_pad=pad_sequences(sequence_test,maxlen=max_seq_len)encoding={'anger':0,'sadness':1,'fear':2,'joy':3,'surprise':4,'love':5}
y_train=[encoding[x] for x in data_train.Emotion]
y_test=[encoding[x] for x in data_test.Emotion]
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
```

**为了创建这个模型，我们使用 1M(在维基百科上训练的 100 万个词向量)版本的预训练词向量。**

**有训练过的词向量可以用于这个目的，你可以在这里 查看 [**。**](https://fasttext.cc/docs/en/english-vectors.html)**

**使用这些词向量有助于我们以更有效和更彻底的方式训练我们的模型，从而产生更高的训练准确度。**

```
def create_embedding_matrix(*filepath*,*word_index*,*embedding_dim*):
    vocab_size=len(word_index)+1
    embedding_matrix=np.zeros((vocab_size,embedding_dim))
    with open(filepath) as f:
        for line in f:
            word,*vector=line.split()
            if word in word_index:
                idx=word_index[word]
                embedding_matrix[idx] = np.array(vector,dtype=np.float32)[:embedding_dim]
    return embedding_matrixfname='embeddings/wiki-news-300d-1M.vec'
embedd_matrix=create_embedding_matrix(fname,index_of_words,embed_num_dims)
```

**现在，我们创建一个用于训练模型的架构。为此，我们首先创建一个 [**嵌入**](https://keras.io/api/layers/core_layers/embedding/) 层，其权重从单词矢量文件中获得。**

**我们还增加了一个 [**双向**](https://keras.io/api/layers/recurrent_layers/bidirectional/) 层，其特点是:**

*   **gru_output_size = 128**
*   **辍学= 0.2**
*   **经常性辍学= 0.2**

**最后，添加一个 [**密集**](https://keras.io/api/layers/core_layers/dense/) 层，该层具有“软最大”激活。**

**Adam 的优化器用作优化器，损失使用“分类 _ 交叉熵”计算。**

**“model.summary()”可用于查看模型中的特征、图层类型、输出形状和参数数量。**

```
embedd_layer=Embedding(vocab_size,embed_num_dims,input_length=max_seq_len,weights=[embedd_matrix],trainable=False)
gru_output_size=128
bidirectional=True
model=Sequential()
model.add(embedd_layer)
model.add(Bidirectional(GRU(units=gru_output_size,dropout=0.2,recurrent_dropout=0.2)))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
```

**最后，我们 [**使用我们的训练集训练模型**](https://www.tensorflow.org/api_docs/python/tf/keras/Model) ，同时测试准确性，因为模型的度量被设置为“准确性”。**

**这里，批次大小取为 128，时期数为 8。**

**可以改变 [**批次大小**](https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/) 和 [**历元数**](/@upendravijay2/what-is-epoch-and-how-to-choose-the-correct-number-of-epoch-d170656adaaf) 。为了避免过度拟合，时期的数量不应该太高。批处理大小也可以变化，在许多情况下，较大的批处理大小会产生更好的结果，但它们也会占用大量内存，因此在许多系统中是不可能执行的。**

```
batch_size=128
epochs=8
hist=model.fit(X_train_pad,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test_pad,y_test))
```

**8 个时代完成后，我们可以测试模型。**

**请记住，我们已经将情感转换为分类或数值，因此为了获得准确的情感，我们需要将分类值映射回其实际的英语情感。**

```
message=['I am sad.']
seq=tokenizer.texts_to_sequences(message)
padded=pad_sequences(seq,maxlen=max_seq_len)
pred=model.predict(padded)
print('Message:'+str(message))
print('Emotion:',class_names[np.argmax(pred)])
```

**对于上面这句话，我们得到的结果是‘悲伤’，大概是对的。**

**通过合并上述代码片段，可以获得整个程序。为了执行这个程序，我建议使用 Google Colab，因为它有一个内置的 GPU，这对训练机器学习模型很有用，除非你的系统已经有一个 GPU，在这种情况下，你是一个幸运的人。**

**一个有用的花絮， [**保存文本模型**](https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model) 以备将来使用(你永远不知道什么时候 [**你会需要它**](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model) )。**

```
tf.keras.models.save_model(model,'textmodel',overwrite=True,include_optimizer=True,save_format=None,signatures=None,options=None)
```

**使用这个文本模型，你可以创建一个测验来检测用户的情绪，或者对 tweets 或 Reddit 帖子进行情绪分析，这种可能性是无限的。**

**祝您在数据科学之旅中好运，感谢您的阅读:)**