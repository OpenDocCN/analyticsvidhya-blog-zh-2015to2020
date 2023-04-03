# 使用长短期记忆网络的文本生成

> 原文：<https://medium.com/analytics-vidhya/text-generation-using-long-short-term-memory-network-2c288966f648?source=collection_archive---------24----------------------->

![](img/a3072ca57310660f0b7a68cfa3d02e0f.png)

我们将在文本数据上训练一个 LSTM 网络，它自己学习生成与训练材料形式相同的新文本。如果你在文本数据上训练你的 LSTM，它会学习产生新的单词，类似于我们训练的单词。LSTM 通常会从源数据中学习人类语法。当用户像聊天机器人一样输入文本时，你也可以使用类似的技术来完成句子。

使用 *tensorflow 2.x* 导入我们的依赖项—

```
*import string
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences*
```

*读取数据*

```
*file=open('t8.shakespeare.txt','r+')
data=file.read()*
```

## *文本清理*

*获得文本数据后，清理文本数据的第一步是对你要达到的目标有一个清晰的认识，并在这种背景下回顾你的文本，看看到底什么会有帮助。*

*在数据中有许多标点符号和数字字符，以消除它*

```
*data=data.split('\n') 
data=data[253:]
data=' '.join(data)*
```

*cleaner 函数有助于删除数据中的标点和数字，并将中的所有字符转换为小写*

```
*def cleaner(data):
    token=data.split()
    table=str.maketrans('','',string.punctuation)
    token=[w.translate(table) for w in token]
    token=[word for word in token if word.isalpha()]
    token=[word.lower() for word in token]
    return tokenwords=cleaner(data=data)*
```

## *创造一个单词序列*

*seed_length 是 50，这意味着前 50 个单词将是我的输入，下一个单词将是我的输出。它需要大量的计算能力和内存来处理所有数据。所以我只用前 10 万个单词来训练我的神经网络。*

```
*seed_length=50+1
sentence=list()
for i in range(seed_length,len(words)):
    sequence=words[i-seed_length:i]
    line=' '.join(sequence)
    sentence.append(line)
    if i >100000:
        break*
```

*神经网络要求对输入数据进行整数编码，这样每个单词都由一个唯一的整数表示。编码后将整数转换成整数序列*

```
*tokenizer=Tokenizer()
tokenizer.fit_on_texts(sentence)
sequence=tokenizer.texts_to_sequences(sentence)
sequence=np.array(sequence)*
```

*分离自变量和目标变量*

```
*X,y=sequence[:,:-1],sequence[:,-1]
vocab_size=len(tokenizer.word_index)+1
y=to_categorical(y,num_classes=vocab_size)*
```

## *创建 LSTM 网络*

*嵌入层被定义为网络的第一个隐藏层。它必须需要 3 个参数*

1.  *vocab_size —文本数据中词汇的大小。*
2.  *output_dim —单词将嵌入其中的向量的大小。*
3.  *输入长度—输入序列的长度。*

```
*model=Sequential()
model.add(Embedding(vocab_size,50,input_length=50))
model.add(LSTM(100,return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100,activation='relu'))
model.add(Dense(vocab_size,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()*
```

## *训练我们的模型*

*训练你的模型更多的时代，我们的网络将能够学习如何生成单词。*

```
*model.fit(X,y,batch_size=256,epochs=1000)*
```

*generate 函数帮助我们生成 50 个单词之后的单词，作为模型的输入*

```
*def generate(text,n_words):
    text_q=[]
    for _ in range(n_words):
        encoded=tokenizer.texts_to_sequences(text)[0]
        encoded=pad_sequences([encoded],maxlen=sequence_length,truncating='pre')
        prediction=model.predict_classes(encoded)
        for word , index in tokenizer.word_index.items():
            if index==prediction:
                predicted_word=word
                break
        text=text+' '+predicted_word
        text_q.append(predicted_word)
    return ' '.join(text_q)*
```

*使用函数并生成接下来的 100 个单词*

```
*input = sentence[0]
generate(input,100)*
```

*感谢阅读！我希望这篇文章是有帮助的。*

*你们的评论和掌声让我有动力创作更多的材料。我很欣赏你！😊*