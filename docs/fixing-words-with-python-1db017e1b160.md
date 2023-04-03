# 用 Python 修正单词

> 原文：<https://medium.com/analytics-vidhya/fixing-words-with-python-1db017e1b160?source=collection_archive---------18----------------------->

## 我的文本分析之旅:文本预处理

我最近刚在大学里开设了一个新的模块，内容是文本分析。我被教导扮演文字技术员的角色，这很有趣。

![](img/d0ea93599f11dfb7e42066c342307af2.png)

由[萨法尔·萨法罗夫](https://unsplash.com/@codestorm?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

文本预处理是文本分析的重要组成部分。尽管人类阅读和解释大块文本很容易，但对计算机来说却很难，因为它们没有语言的概念。语境也很关键，因为它决定了意义，比如“我喜欢苹果”是指水果还是公司。

但首先，在我们试图解读文本之前，我们需要对它们进行预处理！以下是阶段和定义，以及示例。

# 标记化

当我们收到一堆要分析的文本时，我们要做的第一件事就是把它们分成单词和标点符号。你也可以使用 python 的`split()`函数来实现。

```
from nltk.tokenize import word_tokenizetweets = ["This year General Elections is really intense!","Wah GE queues sibeh long... #hot #sweaty", "I have been queueing for too long!", "It will be troubling if youths today do not vote wisely."]tokenized_tweet = [word_tokenize(tweet) for tweet in tweets]
```

# 文本规范化

我们将每个单词标准化，以准备对它们进行统一处理。这是通过把所有的词放在一个公平的竞技场上来完成的。有许多方法可以规范化文本，但这是两种流行的方法。

## 堵塞物

词干化就是去掉单词的后缀或前缀。尽管这可能会导致无效或不相关的单词，如源于“麻烦”的“troubl ”,但词干化通常比变元化在更短的运行时间内完成。

```
import nltk
from nltk.tokenize import word_tokenizeporter = nltk.PorterStemmer()tweets = ["This year General Elections is really intense!","Wah GE queues sibeh long... #hot #sweaty", "I have been queueing for too long!", "It will be troubling if youths today do not vote wisely."]
tokenized_tweets = [word_tokenize(tweet) for tweet in tweets]
stemmed_tweets = []for tweet in tokenized_tweets:
    stemmed_tweets.append([porter.stem(w) for w in tweet])
```

## 词汇化

词汇化就是从一篇文章的实际词根处推导出来。例如，“困扰”的*引理*(词根)会是“困扰”。这通常更准确，因为引用了语料库(词典)，但这通常会导致比词干提取更长的运行时间。

```
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizerWNL = WordNetLemmatizer()tweets = ["This year General Elections is really intense!","Wah GE queues sibeh long... #hot #sweaty", "I have been queueing for too long!", "It will be troubling if youths today do not vote wisely."]
tokenized_tweets = [word_tokenize(tweet) for tweet in tweets]
lemma_tweets = []for tweet in tokenized_tweets:
    lemma_tweets.append([WNL.lemmatize(t,'v') for t in tweet])
print(lemma_tweets)
```

# 停用词移除

停用词是对给定的句子没有多少意义的词。这些词包括“the”、“a”、“I”。它们通常对句子的整体情绪没有贡献，因此被删除。

```
from nltk.corpus import stopwordsclean_tweets = []
stopwords_list = stopwords.words('english')
tweets = ["This year General Elections is really intense!","Wah GE queues sibeh long... #hot #sweaty", "I have been queueing for too long!", "It will be troubling if youths today do not vote wisely."]
tokenized_tweets = [word_tokenize(tweet) for tweet in tweets]
lemma_tweets = []
for tweet in tokenized_tweets:
    lemma_tweets.append([WNL.lemmatize(t,'v') for t in tweet])for t in lemma_tweets:
    clean_tweet = [w for w in t if w.lower() not in stopwords_list]
    clean_tweets.append(" ".join(clean_tweet))
print(clean_tweets)
```

# 噪声消除

噪音是指标签、网址、表情符号等等。它们要么没有给句子增加价值，要么没有意义。去除这一点将有助于突出数据源中句子的关键词。下面的例子将探索移除标签。

```
import rehash_tweets = ["Wah GE queues sibeh long... #hot #sweaty"]for t in hash_tweets:
    t = re.sub('\s#[\S]+',"",t)
    print(t)
```

后来，我意识到' \s#[\w]+'也有效，尽管它的意思不同。\s#[\S]+'实际上是指删除#符号后的任何非空白字符，但是' \s#[\w]+'仅指删除任何字母或数字。我想使用' S '更全面，所以我们可以坚持这样做！

# 这还不是全部，但我们会在这里休息一下

文本处理帮助我理解计算机是如何理解单词的。从这个简短的教程中，我能够理解如何在句子中修复单词，删除或替换，直到我们剩下最有价值的单词。

下次见！