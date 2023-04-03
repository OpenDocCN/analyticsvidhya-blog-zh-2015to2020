# 在 Python 中构建基本的文本预处理管道

> 原文：<https://medium.com/analytics-vidhya/building-a-basic-text-preprocessing-pipeline-in-python-affd82d2471b?source=collection_archive---------7----------------------->

![](img/cf7fa3f57e018e2dc19bf20d85d9f52b.png)

安妮·斯普拉特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

## 如何在 Python 中为非结构化文本数据构建文本预处理管道

尽管最先进的 NLP 语言模型，如 [BERT](https://github.com/google-research/bert) ，都有自己的标记化代码，不需要任何预处理，**许多更基本的 NLP 技术在实现之前仍然需要不同级别的预处理**。Python 的大量库为此提供了帮助，但是有时很难决定要实现哪些预处理任务，以及如何成功地实现它们。

在本文中，我们将通过下面的标准预处理技术来理解它们做什么，什么时候使用它们，以及如何在 Python 中实现它们。
此列表绝非详尽无遗:

*   空白规范化
*   小写字母盘
*   标点符号和数字删除
*   停用词删除
*   拼写纠正
*   词干化和词汇化

每当我从事一个新的 NLP 项目时，我发现自己在 Python 中创建了一个新的预处理管道。为了减少这种工作量，随着时间的推移，我收集了不同预处理技术的代码，并将它们合并到一个[text preprocessor Github repository](https://github.com/jonnyndavis/TextPreProcessor)中，它允许您用几行 Python 代码创建一个完整的文本预处理管道。下面所有的代码都改编自这个库。

# 1.空白规范化

这是用单个空格替换多个连续空格，并删除前导和尾随空格(字符串开头或结尾的空格)。

![](img/ea64dff53c5d44977f420f11f2409774.png)

[Kelly Sikkema](https://unsplash.com/@kellysikkema?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

## 你什么时候使用它？

*   删除因删除停用词、标点符号等而产生的空白。
*   如果使用基于单个空格进行拆分的记号赋予器，则停止向记号添加空格。
*   提高可读性，如果文本将被输出的人审查。
*   何时使用基本匹配短语在文本中搜索多个单词或短语。

## 你是怎么实现的？

**Python 代码:**

```
s = ‘   The quick brown    fox jumps over the lazy dog   ‘print(‘ ‘.join(s.split()))
```

**输出:**

```
The quick brown fox jumps over the lazy dog
```

# 2.小写字母盘

用小写字母替换所有大写字母。

![](img/4d8fe9d87bb2560ae67e6665279d6902.png)

亚历山大·安德鲁斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

## 你什么时候使用它？

*   当 case 没有提供额外信息时。一般来说，这是你不需要知道一个单词是否在句子的开头，或者你对语气不感兴趣(“够了”vs .“够了”)。
*   当使用[预训练的语言模型](https://towardsdatascience.com/pre-trained-language-models-simplified-b8ec80c62217)时，如果不使用预构建的标记器，只对小写文本进行训练。
*   你可能想避免使用小写字母的语言，比如德语中所有的名词都是大写的。在这种情况下，保留大写字母有助于识别和区分名词和其他单词。

## 你是怎么实现的？

**Python 代码:**

```
s = ‘The quick brown Fox jumps over the lazy Dog’
print(s.lower())
```

**输出:**

```
the quick brown fox jumps over the lazy dog
```

# 3.标点符号和数字删除

删除文本中的所有标点符号和数字字符。在某些情况下，数字也可能被替换为一个令牌，如' numbers '。

![](img/071e8477d7b0fea85a4efe5356589fb6.png)

尼克·希利尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

## 你什么时候使用它？

*   如果使用[单词包](https://en.wikipedia.org/wiki/Bag-of-words_model#:~:text=The%20bag%2Dof%2Dwords%20model,word%20order%20but%20keeping%20multiplicity.)，尤其是基于统计单词出现次数的 unigrams。由于标点符号和数字通常只在给定的上下文中保留重要的意义，它们孤立地提供的有用信息很少。
*   用二元模型、三元模型等去除数字。如果没有兴趣知道对象的数量(我们要知道有猫，不是说有三只)。
*   对语气不感兴趣的时候去掉标点(“就是这样！”以及“就这样？”有不同的含义)。

## 你是怎么实现的？

以下示例删除了下列标点符号:！"#$%&\'()*+,-./:;<=>？@[\\]^_`{|}~

**Python 代码:**

```
import string
import regex as re
punctuation = string.punctuation
s = 'The 1 quick, brown fox jumps over the 2 lazy dogs.'
print(s.translate(str.maketrans('', '', punctuation)))
print(re.sub(r'\d+', '', s)
```

**输出:**

```
The quick brown fox jumps over the lazy dog
The  quick, brown fox jumps over the  lazy dogs.
```

*注意:在删除数字的情况下，为什么需要进行空白规范化。*

# 4.停用词删除

删除通常不提供任何额外信息的常用词，如“a”、“and”和“is”。

![](img/78f64249ac3eaab0766148ec0586a101.png)

何塞·阿拉贡内塞斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

## 你什么时候使用它？

*   [单词袋](https://en.wikipedia.org/wiki/Bag-of-words_model#:~:text=The%20bag%2Dof%2Dwords%20model,word%20order%20but%20keeping%20multiplicity.)，其中普通单字的计数不增加任何信息，因此不必要地增加了特征空间的大小。在某些情况下，您可能会考虑保留二元模型、三元模型等。
*   你对[负反馈](https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52)不感兴趣，也就是说，你不需要知道一个单词前面是否有‘不是’或‘不’这样的词。
*   您可能希望向停用字词列表中添加额外的字词，以便删除在特定语料库中没有添加任何有用信息的字词或任何字符序列(例如，汽车评论语料库中的字词“汽车”)。

## 如何实施？

这里，我们使用[空格](https://spacy.io/)停用词。然而，其他库，比如 [NLTK](https://www.nltk.org/) ，有它们自己的停用词表。如上所述，你甚至可以创建自己的列表。

**Python 代码:**

```
import spacy
spacy.load('en_core_web_sm')
stopwords = spacy.lang.en.stop_words.STOP_WORDS
s = 'the quick brown fox jumps over the lazy dog'
print(' '.join([word for word in s.split(' ') if word.lower() not in stopwords]))
```

**输出:**

```
quick brown fox jumps lazy dog
```

# 5.拼写纠正

用预定义列表中的相似单词替换该列表中不存在的单词。

![](img/802fb88afb296933ddddd5d18ff7a1be.png)

照片由[金·戈尔加](https://unsplash.com/@kimgorga?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

## 你什么时候使用它？

*   有效的拼写纠正在几乎所有的 NLP 用例中都是有用的。但是，查找替换单词的计算开销可能非常大。因此，需要根据所需的处理计算的增加来权衡优势。
*   在某些情况下，如果计算成本是一个重要因素，那么只删除未识别的单词可能比替换它们更有效。

## 如何实施？

pyspellchecker 库提供了一个基于 [Levenshtein 距离](https://en.wikipedia.org/wiki/Levenshtein_distance)的拼写纠正实现。这是从一个单词到另一个单词所需的最少单字符编辑(删除、插入、替换)次数。

**Python 代码:**

```
from spellchecker import SpellChecker
spell = SpellChecker()
s = 'The qiuck brown fox jmps over the lazy dog'
print(' '.join([spell.correction(word) for word in s.split(' ')]))
```

**输出:**

```
The quick brown fox mps over the lazy dog
```

*注意:这里的“jmps”一词被错误地替换为“mps ”,表明这种方法并不完美。*

# 6.词干化或词汇化

[通常在文本中](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)一个单词可以以几种不同的形式出现(如 jump、jumps、jumping ),在其他情况下，单词可能源自一个共同的意思(如 democracy、democratic)。词干化和词汇化的目标都是用单词的单一基本形式替换这些不同的形式。

**词干**通过遵循一套启发法来实现这一点，这些启发法切断，有时替换单词的结尾。例如
快→快
快→快
快→快
加快→加快

**词条化**是一个更复杂的过程，它涉及到单词的研究，只用字典定义的词条替换它们。例如
去→去
去→去
去→去
来→来

![](img/983940bc2474345fac74224642e18480.png)

西蒙·伯杰在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

## 你什么时候使用它？

*   较小的文本语料库，其中没有足够多的单词不同形式的出现来独立学习它们的意思。
*   信息检索，你在寻找某个主题(如果你搜索跑步，你也想要跑步、跑步、跑步者等的结果。).
*   文档聚类，您希望按主题分割文档。

## 你是怎么实现的？

有许多不同的库实现了词干化和词汇化，还有许多不同的词干化方法。在本例中，我们将使用:

*   [NLTK](https://www.nltk.org/) 库用[雪球去梗器](https://www.nltk.org/howto/stem.html)进行去梗
*   用于词汇化的空间库

**Python 代码:**

```
import spacy
from nltk.stem.snowball import SnowballStemmer
s = 'The quickest brown fox was jumping over the lazy dog'
sp = spacy.load('en_core_web_sm')
stemmer = SnowballStemmer(language='english')
print(f"Stemming: {' '.join([stemmer.stem(word) for word in s.split(' ')])}")
print(f"Lemmatization: {' '.join([token.lemma_ for token in sp(s)])}")
```

**输出:**

```
Stemming: the quickest brown fox was jump over the lazi dog
Lemmatization: the quick brown fox be jump over the lazy dog
```

# 结论

知道使用哪种文本预处理技术与其说是一门科学，不如说是混合了经验的反复试验。对于许多 NLP 任务，比方说一个使用单词包的分类模型，最好先尝试用很少或没有预处理的方法训练你的模型。从那里，您可以返回并添加预处理步骤，以查看是什么提高了模型性能。

这个列表并不详尽，但我希望它能为您面临的任何 NLP 任务提供很好的参考！