# 自然语言处理

> 原文：<https://medium.com/analytics-vidhya/natural-language-processing-bedb2e1c8ceb?source=collection_archive---------7----------------------->

![](img/34b7116f9af8edb5019023825ac3dca8.png)

当智能设备理解我们告诉它们的内容时，我们最初不都感到惊讶吗？事实上，它也以最友好的方式回答了，不是吗？就像苹果的 Siri 和亚马逊的 Alexa 一样，当我们问天气、问路或播放某种类型的音乐时，它们都能理解。从那时起，我就想知道这些计算机是如何获得我们的语言的。这种久违的好奇心重新点燃了我，我想以一个新手的身份写一篇博客。

在本文中，我将使用一个流行的叫做 NLTK 的 NLP 库。自然语言工具包(NLTK)是最强大的，也可能是最流行的自然语言处理库之一。它不仅拥有最全面的基于 python 的编程库，还支持最多的不同人类语言。

**什么是自然语言处理？**

自然语言处理(NLP)是语言学、计算机科学、信息工程和人工智能的一个子领域，涉及计算机和人类语言之间的交互，特别是如何训练计算机处理和分析大量自然语言数据。

**为什么非结构化数据类型的排序如此重要？**

在时钟的每一秒，世界都会产生大量的数据！！，是啊，这真是令人难以置信！！大多数数据属于非结构化数据类型。文本、音频、视频、图像等数据格式是非结构化数据的典型例子。非结构化数据类型不会像关系数据库的传统行和列结构那样具有固定的维度和结构。因此，它更难分析，也不容易搜索。话虽如此，对于商业组织来说，找到应对挑战和抓住机遇的方法以获得洞察力并在竞争激烈的环境中取得成功也很重要。然而，在自然语言处理和机器学习的帮助下，这种情况正在迅速改变。

计算机和我们的自然语言混淆了吗？

人类语言是强有力的交流工具之一。我们使用的词语、语气、句子、手势都在传达信息。在一个短语中有无数种不同的组词方法。单词也可以有多种含义，理解人类语言的意图是一个挑战。语言悖论是一个自相矛盾的短语或句子，例如，“哦，这是我公开的秘密”，“你能自然地行动吗”，尽管这听起来很愚蠢，但我们人类可以理解并在日常生活中使用，但对于机器来说，自然语言的模糊性和不准确性是航行的障碍。

![](img/089f0d492613511a37dd0408ec9f25b4.png)

**最常用的 NLP 库**

在过去，只有那些在数学、计算机学习和自然语言处理中的语言学方面拥有卓越知识的先驱者才能成为 NLP 项目的一部分。现在，开发人员可以使用现成的库来简化文本的预处理，以便他们可以专注于创建机器学习模型。这些库只需要几行代码就可以实现文本理解、解释和情感分析。最受欢迎的 NLP 库有:

Spark NLP，NLTK，PyTorch-Transformers，TextBlob，Spacy，Stanford CoreNLP，Apache OpenNLP，Allen NLP，GenSim，NLP Architecture，sci-kit learn。

*问题是我们应该从哪里开始，如何开始？*

你有没有观察过孩子们是如何开始理解和学习语言的？是的，通过挑选每个单词和句子结构，对吧！让计算机理解我们的语言或多或少与它相似。

**预处理步骤:**

1.  句子标记化
2.  单词标记化
3.  文本词汇化和词干化
4.  停止言语
5.  词性标注
6.  组块
7.  Wordnet
8.  词汇袋
9.  TF-IDF

1.  **句子标记化(句子切分)** 要让计算机理解自然语言，第一步就是把段落分解成句子。标点符号是把句子分开的一种简单方法。

```
import nltk
nltk.download('punkt')text = "Home Farm is one of the biggest junior football clubs in Ireland and their senior team, from 1970 up to the late 1990s, played in the League of Ireland. However, the link between Home Farm and the senior team was severed in the late 1990s. The senior side was briefly known as Home Farm Fingal in an effort to identify it with the north Dublin area."sentences = nltk.sent_tokenize(text)
print("The number of sentences in the paragrah:",len(sentences))for sentence in sentences:
print(sentence)**OUTPUT:** The number of sentences in the paragraph: 3 Home Farm is one of the biggest junior football clubs in Ireland and their senior team, from 1970 up to the late 1990s, played in the League of Ireland. However, the link between Home Farm and the senior team was severed in the late 1990s. The senior side was briefly known as Home Farm Fingal in an effort to identify it with the north Dublin area.
```

**2。到目前为止，我们已经分离出句子，下一步是将句子分解成单词，这些单词通常被称为记号。**

在自己的生活中创造空间的方式有助于好的方面，同样地，单词之间的空间有助于在一个短语中把单词分开。我们也可以将标点符号视为独立的符号，因为标点符号也有其用途。

```
for sentence in sentences:
words = nltk.word_tokenize(sentence)
print("The number of words in a sentence:", len(words))
print(words)**OUTPUT:** The number of words in a sentence: 32 
['Home', 'Farm', 'is', 'one', 'of', 'the', 'biggest', 'junior', 'football', 'clubs', 'in', 'Ireland', 'and', 'their', 'senior', 'team', ',', 'from', '1970', 'up', 'to', 'the', 'late', '1990s', ',', 'played', 'in', 'the', 'League', 'of', 'Ireland', '.'] The number of words in a sentence: 18 
['However', ',', 'the', 'link', 'between', 'Home', 'Farm', 'and', 'the', 'senior', 'team', 'was', 'severed', 'in', 'the', 'late', '1990s', '.'] The number of words in a sentence: 22 
['The', 'senior', 'side', 'was', 'briefly', 'known', 'as', 'Home', 'Farm', 'Fingal', 'in', 'an', 'effort', 'to', 'identify', 'it', 'with', 'the', 'north', 'Dublin', 'area', '.']
```

在程序中使用`word_tokenize()`或`sent_tokenize()`功能的先决条件是，我们应该下载 **punkt** 包。

**3。词干和文本词条化**

在每个文本文档中，我们通常会遇到不同形式的单词，如 write，writes，writing，意思相同，基本单词相同。但是如何让计算机来分析这样的单词呢？这时就出现了文本词汇化和词干化。

词干化和文本词汇化是**规范化**技术，它们提供了同样的想法，即把一个单词的词尾砍向核心单词。虽然他们都想解决同一个问题，但他们却以完全不同的方式去做。词干化通常是一个粗略的启发式过程，而词汇化是一个基于词汇的形态学基础词。让我们仔细看看！

*词干化* -单词被简化为它们的词干。词干不必与基于词典的形态(最小单位)词根是同一个词根，它只是等于或小于单词的形式。

```
from nltk.stem import PorterStemmer#create an object of class PorterStemmer
porter = PorterStemmer()#A list of words to be stemmed
word_list = ['running', ',', 'driving', 'sung', 'between', 'lasted', 'was', 'paticipated', 'before', 'severed', '1990s', '.']print("{0:20}{1:20}".format("Word","Porter Stemmer"))for word in word_list:
print("{0:20}{1:20}".format(word,porter.stem(word)))OUTPUT:
Word                Porter Stemmer       
running             run                  
,                   ,                    
driving             drive                
sung                sung                 
between             between              
lasted              last                 
was                 wa                   
paticipated         paticip              
before              befor                
severed             sever                
1990s               1990                 
.                   .
```

词干提取并不像看起来那么简单:(
我们可能会遇到两个问题，比如一个单词的**词干提取不足**和**词干提取过度**。

*词汇化*-当我们认为词干是根据单词出现的方式来删减单词的最佳估计方法时，另一方面，词汇化似乎是一种更有计划地删减单词的方法。他们的字典过程包括解析单词。事实上，一个词的引理就是它的字典或标准形式。

```
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()#A list of words to lemmatizeword_list = ['running', ',', 'drives', 'sung', 'between', 'lasted', 'was', 'paticipated', 'before', 'severed', '1990s', '.']print("{0:20}{1:20}".format("Word","Lemma"))for word in word_list:
      print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word)))**OUTPUT:** Word                Lemma                
running             running             
 ,                   ,                    
drives              drive                
sung                sung                 
between             between              
lasted              lasted               
was                 wa                   
paticipated         paticipated          
before              before               
severed             severed              
1990s               1990s                
.                   .
```

如果需要速度，那么使用词干会更好。但是在需要准确性的情况下，最好使用引理化。

**4。停止词
'** 中的'，' at '，' on '，' so '..etc 被认为是停用词。停用词在自然语言处理中并不起重要作用，但是停用词的去除在情感分析中必然起重要作用。

NLTK 附带了 16 种不同语言的停用词，它们包含停用词列表。

```
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))print("The stop words in NLTK lib are:", stop_words)para="""Home Farm is one of the biggest junior football clubs in Ireland and their senior team, from 1970 up to the late 1990s, played in the League of Ireland. However, the link between Home Farm and the senior team was severed in the late 1990s. The senior side was briefly known as Home Farm Fingal in an effort to identify it with the north Dublin area."""tokenized_para=word_tokenize(para)
modified_token_list=[word for word in tokenized_para if not word in stop_words]
print("After removing the stop words in the sentence:")
print(modified_token_list)**OUTPUT:** The stop words in NLTK lib are: {'about', 'ma', "shouldn't", 's', 'does', 't', 'our', 'mightn', 'doing', 'while', 'ourselves', 'themselves', 'will', 'some', 'you', "aren't", 'by', "needn't", 'in', 'can', 'he', 'into', 'as', 'being', 'between', 'very', 'after', 'couldn', 'himself', 'herself', 'had', 'its', 've', 'him', 'll', "isn't", 'through', 'should', 'was', 'now', 'them', "you'll", 'again', 'who', 'don', 'been', 'they', 'weren', "you're", 'both', 'd', 'me', 'didn', "won't", "you'd", 'only', 'itself', 'hadn', "should've", 'than', 'how', 'few', 're', 'down', 'these', 'y', "haven't", "mightn't", 'won', "hadn't", 'other', 'above', 'all', "doesn't", 'isn', "that'll", 'not', 'yourselves', 'at', 'mustn', "it's", 'on', 'the', 'for', "didn't", 'what', "mustn't", 'his', 'haven', 'doesn', "you've", 'are', 'out', 'hers', 'with', 'has', 'she', 'most', 'ain', 'those', 'when', 'myself', 'before', 'their', 'during', 'there', 'or', 'until', 'that', 'more', "hasn't", 'o', 'we', 'and', "shan't", 'which', 'because', "don't", 'why', 'shan', 'an', 'my', 'if', 'did', 'having', "couldn't", 'your', 'theirs', 'aren', 'just', 'further', 'here', 'of', "wouldn't", 'be', 'too', 'her', 'no', 'same', 'it', 'is', 'were', 'yourself', 'have', 'off', 'this', 'needn', 'once', "wasn't", 'against', 'wouldn', 'up', 'a', 'i', 'below', "weren't", 'over', 'own', 'then', 'so', 'do', 'from', 'shouldn', 'am', 'under', 'any', 'yours', 'ours', 'hasn', 'such', 'nor', 'wasn', 'to', 'where', 'm', "she's", 'each', 'whom', 'but'} After removing the stopwords in the sentence: 
['Home', 'Farm', 'one', 'biggest', 'junior', 'football', 'clubs', 'Ireland', 'senior', 'team', ',', '1970', 'late', '1990s', ',', 'played', 'League', 'Ireland', '.', 'However', ',', 'link', 'Home', 'Farm', 'senior', 'team', 'severed', 'late', '1990s', '.', 'The', 'senior', 'side', 'briefly', 'known', 'Home', 'Farm', 'Fingal', 'effort', 'identify', 'north', 'Dublin', 'area', '.']
```

**5。回想一下我们早期的英语语法课，我们还记得老师是如何围绕基本词类进行相关指导以进行有效交流的吗？是啊，美好的旧时光！！让我们也教电脑词性吧。:)**

八个词类分别是*名词、动词、代词、形容词、副词、介词、连词、*和*感叹词。*

词性标注是一种识别和分配句子中单词的词性的能力。有不同的标记方法，但我们将使用通用的标记样式。

```
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
pos_tag= [nltk.pos_tag(i,tagset="universal") for i in words]
print(pos_tag)[[('Home', 'NOUN'), ('Farm', 'NOUN'), ('is', 'VERB'), ('one', 'NUM'), ('of', 'ADP'), ('the', 'DET'), ('biggest', 'ADJ'), ('junior', 'NOUN'), ('football', 'NOUN'), ('clubs', 'NOUN'), ('in', 'ADP'), ('Ireland', 'NOUN'), ('and', 'CONJ'), ('their', 'PRON'), ('senior', 'ADJ'), ('team', 'NOUN'), (',', '.'), ('from', 'ADP'), ('1970', 'NUM'), ('up', 'ADP'), ('to', 'PRT'), ('the', 'DET'), ('late', 'ADJ'), ('1990s', 'NUM'), (',', '.'), ('played', 'VERB'), ('in', 'ADP'), ('the', 'DET'), ('League', 'NOUN'), ('of', 'ADP'), ('Ireland', 'NOUN'), ('.', '.')]
```

POS 标记的应用之一是分析反馈中的产品质量，通过对客户评论中的形容词进行排序，我们可以评估反馈的情绪。举例来说，您在我们这里购物感觉如何？

6。组块
组块用于通过标记以下词类(POS)为句子添加更多结构。也称为浅层解析。由此产生的单词组被命名为“组块”没有这样的预定义规则来执行分块。

短语结构约定:

*   s(句子)→ NP VP。
*   NP →{限定词，名词，代词，专名}。
*   VP → V (NP)(PP)(副词)。
*   PP →代词(NP)。
*   AP →形容词(PP)。

我从来没有享受过复杂正则表达式的美好时光，我曾经尽可能地远离它，但后来意识到，在数据科学中掌握正则表达式是多么重要。让我们从理解这个简单的实例开始。

如果我们需要从句子中标记名词、动词(过去式)、形容词和并列连词。您可以使用下面的规则

组块:{ <nn.>* <vbd.>* <jj.>* <cc>？}</cc></jj.></vbd.></nn.>

```
import nltk
from nltk.tokenize import word_tokenizecontent = "Home Farm is one of the biggest junior football clubs in Ireland and their senior team, from 1970 up to the late 1990s, played in the League of Ireland. However, the link between Home Farm and the senior team was severed in the late 1990s. The senior side was briefly known as Home Farm Fingal in an effort to identify it with the north Dublin area."tokenized_text = nltk.word_tokenize(content)
print("After Split:",tokenized_text)
tokens_tag = pos_tag(tokenized_text)
print("After Token:",tokens_tag)patterns= """mychunk:{<NN.?>*<VBD.?>*<JJ.?>*<CC>?}"""chunker = RegexpParser(patterns)
print("After Regex:",chunker)
output = chunker.parse(tokens_tag)
print("After Chunking",output)**OUTPUT:** After Regex: chunk.RegexpParser with 1 stages: RegexpChunkParser with 1 rules: <ChunkRule: '<NN.?>*<VBD.?>*<JJ.?>*<CC>?'> After Chunking 
(S   (mychunk Home/NN Farm/NN)   is/VBZ   one/CD  of/IN   the/DT   
(mychunk biggest/JJS)   
(mychunk junior/NN football/NN clubs/NNS)   in/IN  
(mychunk Ireland/NNP and/CC)   their/PRP$   
(mychunk senior/JJ)   
(mychunk team/NN)   ,/,   from/IN   1970/CD   up/IN   to/TO   the/DT   (mychunk late/JJ)   1990s/CD   ,/,   played/VBN   in/IN   the/DT   (mychunk League/NNP)   of/IN   (mychunk Ireland/NNP)   ./.)
```

7 .**。Wordnet**

Wordnet 是一个 NLTK 语料库阅读器，一个英语词汇数据库。它可以用来生成同义词或反义词。

```
from nltk.corpus import wordnetsynonyms = []
antonyms = []for syn in wordnet.synsets("active"):
        for lemmas in syn.lemmas():
            synonyms.append(lemmas.name())for syn in wordnet.synsets("active"):
        for lemmas in syn.lemmas():
            if lemmas.antonyms():
                antonyms.append(lemmas.antonyms()[0].name())print("Synonyms are:",synonyms)
print("Antonyms are:",antonyms)**OUTPUT:** Synonyms are: ['active_agent', 'active', 'active_voice', 'active', 'active', 'active', 'active', 'combat-ready', 'fighting', 'active', 'active', 'participating', 'active', 'active', 'active', 'active', 'alive', 'active', 'active', 'active', 'dynamic', 'active', 'active', 'active'] Antonyms are: ['passive_voice', 'inactive', 'passive', 'inactive', 'inactive', 'inactive', 'quiet', 'passive', 'stative', 'extinct', 'dormant', 'inactive']
```

**8。单词袋** 单词袋模型将原始文本转化为单词，并计算单词在文本中的出现频率。

```
import nltk
import re # to match regular expressions
import numpy as nptext="Home Farm is one of the biggest junior football clubs in Ireland and their senior team, from 1970 up to the late 1990s, played in the League of Ireland. However, the link between Home Farm and the senior team was severed in the late 1990s. The senior side was briefly known as Home Farm Fingal in an effort to identify it with the north Dublin area."sentences = nltk.sent_tokenize(text)
for i in range(len(sentences)):
  sentences[i] = sentences[i].lower()
  sentences[i] = re.sub(r'\W', ' ', sentences[i])
  sentences[i] = re.sub(r'\s+', ' ', sentences[i])bag_of_words = {}
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    for word in words:
       if word not in bag_of_words.keys():
          bag_of_words[word] = 1
       else:
          bag_of_words[word] += 1
print(bag_of_words)**OUTPUT:** {'home': 3, 'farm': 3, 'is': 1, 'one': 1, 'of': 2, 'the': 8, 'biggest': 1, 'junior': 1, 'football': 1, 'clubs': 1, 'in': 4, 'ireland': 2, 'and': 2, 'their': 1, 'senior': 3, 'team': 2, 'from': 1, '1970': 1, 'up': 1, 'to': 2, 'late': 2, '1990s': 2, 'played': 1, 'league': 1, 'however': 1, 'link': 1, 'between': 1, 'was': 2, 'severed': 1, 'side': 1, 'briefly': 1, 'known': 1, 'as': 1, 'fingal': 1, 'an': 1, 'effort': 1, 'identify': 1, 'it': 1, 'with': 1, 'north': 1, 'dublin': 1, 'area': 1}
```

**9。TF-IDF**

TF-IDF 代表**词频—逆文档频率**。

文本数据需要转换为数字格式，其中每个单词都以矩阵形式表示。给定单词的编码是向量，其中对应的元素被设置为 1，所有其他元素为零。因此 TF-IDF 技术也被称为**字嵌入**。

TF-IDF 基于两个概念:

**TF(t) =(术语 t 在文档中出现的次数)/(文档中的总术语数)**

**IDF(t) = log_e(文档总数/包含术语 t 的文档数)**

```
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pddocs=["Home Farm is one of the biggest junior football clubs in Ireland and their senior team, from 1970 up to the late 1990s, played in the League of Ireland",
"However, the link between Home Farm and the senior team was severed in the late 1990s",
" The senior side was briefly known as Home Farm Fingal in an effort to identify it with the north Dublin area"]#instantiate CountVectorizer()
cv=CountVectorizer()# this steps generates word counts for the words in your docs
word_count_vector=cv.fit_transform(docs)tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])# sort ascending
df_idf.sort_values(by=['idf_weights'])# count matrix
count_vector=cv.transform(docs)# tf-idf scores
tf_idf_vector=tfidf_transformer.transform(count_vector)feature_names = cv.get_feature_names()#get tfidf vector for the document
first_document_vector=tf_idf_vector[0]#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)tfidf
of      0.374810
ireland 0.374810
the     0.332054
in      0.221369
1970    0.187405
football 0.187405
up      0.187405
as      0.000000
an      0.000000and so on..
```

这些分数告诉我们什么？单词在文档中越常见，得分越低，越独特的单词得分越高。

到目前为止，我们学习了清理和预处理文本的步骤。在这一切之后，我们可以用排序后的数据做什么呢？我们可以用这些数据进行情感分析，聊天机器人，市场情报。也许建立一个基于用户购买或商品评论或客户分类的推荐系统。

计算机对人类语言的准确性仍然不如对数字的准确性。随着每天产生大量的文本数据，自然语言处理对于理解数据变得越来越重要，并被用于许多其他应用中。因此，有无尽的方法来探索自然语言处理。