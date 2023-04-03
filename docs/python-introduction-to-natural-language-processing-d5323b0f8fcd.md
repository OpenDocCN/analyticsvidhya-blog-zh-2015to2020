# Python -自然语言处理简介

> 原文：<https://medium.com/analytics-vidhya/python-introduction-to-natural-language-processing-d5323b0f8fcd?source=collection_archive---------13----------------------->

![](img/6fe6264de7da225c74e54457eabafc91.png)

大家好，在这篇博文中我将介绍自然语言处理这个主题。自然语言处理是人工智能的一个子类。今天它被用于许多领域。语言翻译、自动文本翻译、自动语音和命令理解、通过搜索给定文本或单词找到所需结果、自动拼写纠正键盘、单词预测等等。

在这篇博文中，我将只讲述我在 Python 中对一些文本进行自然语言处理工作的一部分。我会在我的下一篇博文中谈论另一部分。可以在这里获取完整代码:[**https://github . com/minnela/IntroductionNaturalDataProcessing**](https://github.com/minnela/IntroductionNaturalDataProcessing)

# **RegEx(正则表达式)**

正则表达式是一种算法，可以很容易地在字符串中找到某个字符集。

假设我想找到给定文本中的所有句子及其编号。首先获取文本:

```
def getText():
    file = open("hw01_FireFairies.txt")
    data = file.read()
    return data
```

然后拆分成句子:

```
def split_into_sentences(text):
   text = text.lower()
   sentences = re.split(r'(?<!\w\.\w.)(?<![A**-**Z][a**-**z]\.)(<=\.|\?)\s', text)
   return sentences
```

这里，我们看到一个正则表达式，我想解释一下它是如何工作的:

```
**1)** **(?<!\w\.\w.) ----->** "**?<!**" This expression indicates a negative loop.It ensured that an abbreviation term such as "one character, dot, another character, dot" is not perceived as a sentence due to dots. To give an example of this: "I like ice cream i.e. chocolate ice cream". Here "i.e."  abbreviation, was not taken as a sentence thanks to this regular expression.**2) (?<![A-Z][a-z]\.) ----->** "**?<!**" This expression indicates a negative loop.This expression was put in order to avoid taking abbreviations starting with a capital letter and a lowercase letter as sentences. For example, "Mr." The abbreviation was not taken as a sentence.**3)(?<=\.|\?)---->** "**?<=**" This expression indicates a positive loop.This expression accepts the part up to the point or the question mark as a sentence and cuts off after the period. **4) \s ---->** space expressionThis expression says that a space must come after a period at the end of a sentence. For example there is a mail in the  text "cheapsite.com" and it does not take it as a sentence since the period is combined with both words.
```

# **创建单字和双字**

我们希望机器能够理解语言。对于语言分析，我们将文本中的单词一个接一个地或者两个或更多地分开。让我们想想你的电话键盘。当你输入信息时，它会自动完成你的句子，当你写错一个单词时，它会猜测你想写什么。为此，句子被分组。将单词放在一起，计算句子概率，让机器完成句子。这叫做 n-gram。一克是一克。句子中的每一个单词都叫做单字。如果每个单词的前后都有一个单词，那么就产生了二元模型。比如我有一句“我要去上学了”。

**单字:**“我”、“我”、“去”、“去”、“学校”

**二元组:**"(我，我)"，"(我，去"，"去，到"，"到，学校"，"学校，。)"

为了进行文本分析，我接受了点(。)作为一个单词，因为它在将文本拆分成句子时很重要。在找到二元模型和一元模型后，我们找到二元模型的概率来确定哪两组单词最常一起使用。

要找出二元模型(I，am)的概率，

**二元组总数(I，am) /一元组总数(I)**

在 python 中，我创建了给定文本的一元模型、二元模型及其概率，如下所示:

**创建二元模型:**

```
def createBigram(data):
    listOfBigrams = []
    bigramCounts = {}
    unigramCounts = {}
    text = data.lower()
    words= re.findall(r'\b[a**-**zA**-**Z]+|[.!?]', text)

    for i in range(len(words)):
        if i < len(words) - 1:

          listOfBigrams.append((words[i], words[i + 1]))

          if (words[i], words[i + 1]) in bigramCounts:
              bigramCounts[(words[i], words[i + 1])] += 1
          else:
              bigramCounts[(words[i], words[i + 1])] = 1

        if words[i] in unigramCounts:
            unigramCounts[words[i]] += 1
        else:
            unigramCounts[words[i]] = 1

    return words,listOfBigrams, unigramCounts, bigramCounts
```

**计算他们的概率:**

```
def calcBigramProb(words,listOfBigrams, unigramCounts, bigramCounts):
    listOfProbBigram = {}
    listOfProbUnigram = {}

    for bigram in listOfBigrams:
        word1 = bigram[0]
        word2 = bigram[1]
        listOfProbBigram[bigram] = (bigramCounts.get(bigram)) / (unigramCounts.get(word1))

    for unigram in words:
        word = unigram
        listOfProbUnigram[unigram]=(unigramCounts.get(word)) / len(words)

    return listOfProbBigram, listOfBigrams,listOfProbUnigram,words
```

我们现在有了二进制词组的可能性。我们可以在给定的文本中计算一个句子的概率。

P(我要去上学)= P(我，是)。P(am，走了)。p(去，去)。警察(到学校)

在我的下一篇博客中，我将解释**平滑方法**,该方法用于计算一个句子包含文本中没有的单词的概率。谢谢你看我的博客。:)