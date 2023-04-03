# NLP——使用 NLTK: TextRank 算法的句子抽取

> 原文：<https://medium.com/analytics-vidhya/sentence-extraction-using-textrank-algorithm-7f5c8fd568cd?source=collection_archive---------2----------------------->

## 使用 Python 和 NLTK 轻松实现

![](img/b3443fcfeedb28e648d10ef803b55904.png)

[北人](https://unsplash.com/@northfolk?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

# 介绍

**TextRank** 是基于 **PageRank** 的算法，常用于关键词抽取和文本摘要。

我们将用 Python 实现用于句子提取的 **TextRank 算法**。该算法的关键是从文本中提取最相关的句子，这是文摘的重要任务之一。

## 但是，让我们不要重新发明轮子

这篇文章的先决条件是理解 **PageRank 算法，**你可以从以下关于 Medium 的文章中读到:

> PageRank (PR)是一种用于计算网页权重的算法，它被[谷歌搜索](https://en.wikipedia.org/wiki/Google_Search)用来在[搜索引擎](https://en.wikipedia.org/wiki/Search_engine)的结果中对[网页](https://en.wikipedia.org/wiki/Webpages)进行排名。

请参考以下文章之一，以获得基本的了解:

[Brendan Massey](https://medium.com/u/c7d314e61b4f?source=post_page-----7f5c8fd568cd--------------------------------) 用大量图片解释了 PageRank 算法背后的症结所在。

[](/hackernoon/implementing-googles-pagerank-algorithm-88069314fb3d) [## 实现 Google 的 Pagerank 算法

### 作为一名学习编程的学生，在某种程度上也是计算机科学的学生，我经常发现新的和令人兴奋的历史…

medium.com](/hackernoon/implementing-googles-pagerank-algorithm-88069314fb3d) 

徐良用 python 实现很好地解释了这一点。

[](https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0) [## Python 关键字提取的文本排名

### Python 和 spaCy 的一个 scratch 实现，帮助你理解用于关键词提取的 PageRank 和 TextRank。

towardsdatascience.com](https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0) 

我们在做什么？

我们将尝试使用 TextRank 算法从文本片段中提取顶级句子。

```
**Those Who Are Resilient Stay In The Game Longer
“On the mountains of truth you can never climb in vain: either you will reach a point higher up today, or you will be training your powers so that you will be able to climb higher tomorrow.” — Friedrich Nietzsche
Challenges and setbacks are not meant to defeat you, but promote you. However, I realise after many years of defeats, it can crush your spirit and it is easier to give up than risk further setbacks and disappointments. Have you experienced this before? To be honest, I don’t have the answers. I can’t tell you what the right course of action is; only you will know. However, it’s important not to be discouraged by failure when pursuing a goal or a dream, since failure itself means different things to different people. To a person with a Fixed Mindset failure is a blow to their self-esteem, yet to a person with a Growth Mindset, it’s an opportunity to improve and find new ways to overcome their obstacles. Same failure, yet different responses. Who is right and who is wrong? Neither. Each person has a different mindset that decides their outcome. Those who are resilient stay in the game longer and draw on their inner means to succeed.**
```

**方法:**

我们将分三步完成。是的，答应我！(好吧，可能 4。)

1.  **对每个句子中的单词进行分词**

这将生成一个**标记化句子列表列表:**

```
[['Those', 'Who', 'Are', 'Resilient', 'Stay', 'In', 'The', 'Game', 'Longer', '*', '“', 'On', 'the', 'mountains', 'of', 'truth', 'you', 'can', 'never', 'climb', 'in', 'vain', ':', 'either', 'you', 'will', 'reach', 'a', 'point', 'higher', 'up', 'today', ',', 'or', 'you', 'will', 'be', 'training', 'your', 'powers', 'so', 'that', 'you', 'will', 'be', 'able', 'to', 'climb', 'higher', 'tomorrow.', '”', '—', 'Friedrich', 'Nietzsche', 'Challenges', 'and', 'setbacks', 'are', 'not', 'meant', 'to', 'defeat', 'you', ',', 'but', 'promote', 'you', '.'], ...]
```

**2。构建相似性矩阵**

我们会用 [**余弦相似度**](https://www.machinelearningplus.com/nlp/cosine-similarity/) 来求两个句子之间的相似度，这个相似度会用来衡量两个句子之间的距离。

> **余弦相似度:**余弦相似度是一种度量，用于确定文档的相似程度，而不考虑它们的大小。
> 
> 余弦相似度作为一种相似度度量，和常用词的数量有什么不同？
> 
> 当绘制在多维空间上时，其中每个维度对应于文档中的一个单词，余弦相似性捕捉文档的方向(角度)而不是大小。

即。考虑下面一对句子及其余弦相似度

```
**1\. "NLP is interesting field of AI"**, And 
   **"NLP makes machine understand language"** Consine Similarity: **0.7238730093922876****2**. **"Alan turing proposed the imitation game"**, And 
   **"Alan turing hated other games"** Consine Similarity: **0.91504896898793****3\. "NLP is not just about chatbots"**, And 
   **"NLP makes chatbots widely used"** Consine Similarity: **0.7764453512724155****4\. "NLP is interesting field of AI"**, And 
   **"NLP is interesting field of AI"** Consine Similarity: **1.0****5\. "Those Who Are Resilient Stay In The Game Longer"**, And 
   **"01234567890"** Consine Similarity: **0.0**
```

类似地，我们需要测量所有句子之间的相似性矩阵。

我们将得到如下相似矩阵:

```
[[0\.         0.11395845 0.35076499 0.10679337 0.22030768 0.09455048
  0.08854208 0.05757988 0.09019807 0.04691224 0.05544081 0.20244412] [0.09867469 0\.         0\.         0.10985716 0.10198264 0.11671565
  0.11710577 0.10661723 0.22268582 0.08686458 0.03421881 0.08820086]...
```

**3。运行 PageRank 算法**

现在我们有了相似矩阵，我们可以对它运行 PageRank 算法。如果你已经按照 [**PageRank 的文章**](https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0) 做了，下面的代码也差不多，很好理解。

我们将生成 PageRank 矩阵，它将包含所有句子的得分，其中最重要的句子得分最高。

我们将得到如下的 PageRank 矩阵:

```
[1.26778728 1.09189214 0.49899918 1.24743264 1.08652989 1.24053865
 1.24198464 0.99690567 0.61024186 0.81140037 0.85049162 1.05579606]
```

**4。提取热门句子**

瞧，我们完成了。你已经猜到这一步了吧？

现在我们将从 PageRank 矩阵中提取顶级句子。

以下是前 5 个句子:{句子:得分}对

```
{'\nThose Who Are Resilient Stay In The Game Longer\n“On the mountains of truth you can never climb in vain: either you will reach a point higher up today, or you will be training your powers so that you will be able to climb higher tomorrow.” — Friedrich Nietzsche\nChallenges and setbacks are not meant to defeat you, but promote you.': **1.2712476366837975**, 'However, I realise after many years of defeats, it can crush your spirit and it is easier to give up than risk further setbacks and disappointments.': **1.0914952900297563**, 'However, it’s important not to be discouraged by failure when pursuing a goal or a dream, since failure itself means different things to different people.': **1.2401359537858996**, 'To a person with a Fixed Mindset failure is a blow to their self-esteem, yet to a person with a Growth Mindset, it’s an opportunity to improve and find new ways to overcome their obstacles.': **1.2415147486210913**,'To be honest, I don’t have the answers.': **1.2468967412932273**}
```

# #一切都在一个地方:和平！

## 找到完整的代码[在这里](https://github.com/akashp1712/nlp-akash/blob/master/text-summarization/text_rank_sentences.py)并且玩它！

 [## akashp1712/nlp-akash

### 此时您不能执行该操作。您已使用另一个标签页或窗口登录。您已在另一个选项卡中注销，或者…

github.com](https://github.com/akashp1712/nlp-akash/blob/master/text-summarization/text_rank_sentences.py) 

> 技术不会自动改进，当许多人努力工作使它变得更好时，它就会改进——埃隆·马斯克