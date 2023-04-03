# 使用提取方法的 Python 文本摘要(包括端到端实现)

> 原文：<https://medium.com/analytics-vidhya/text-summarization-in-python-using-extractive-method-including-end-to-end-implementation-2688b3fd1c8c?source=collection_archive---------3----------------------->

![](img/df944ce03f7633dce4c89e5ec73f66f8.png)

[真诚媒体](https://unsplash.com/@sincerelymedia?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

## **什么是文本摘要**

文本摘要是创建原始文本的简明版本同时保留关键信息的过程。人类天生是很好的总结者，因为我们有能力通过阅读理解文本的整体意思。但是机器如何做到这一点呢？这就是我们在这篇文章中要讨论的内容。

自动文本摘要有许多应用，包括如下:

1.  阅读冗长的客户评论，并将其转换成更小、更有意义的版本，用于采取必要的行动。
2.  将新闻文章转换成简短的摘要。手机应用程序 **inshorts** 就是一个例子。
3.  根据商务会议记录创建简明摘要报告。

如果这足以让你对 NLP 的惊人应用有更多的了解，那么让我们一起深入了解细节。

## **工作原理**

文本摘要有两种方法:

1.  **摘录摘要:**这种方法通过从原始文本中选择最重要的句子子集来概括文本。顾名思义，它从文本中提取最重要的信息。这种方法本身没有生成文本的能力，因此输出总是包含原始文本的某个部分。
2.  **抽象概括:**这种方法背后的思想是理解原文的核心语境，并基于这种理解产生新的文本。它可以比作人类以自己的方式阅读和总结文本的方式。抽象概要的输出可以具有原始文本中不存在的元素。

虽然抽象摘要听起来更有前途，因为它具有更深层次的理解和文本生成的能力，但是摘录摘要也有其自身的优势，例如:

*   比抽象更容易实现，因为不需要语言生成能力。
*   使用无监督的方法实现起来更快，不需要任何事先培训。

我们将更详细地讨论提取总结。

## **摘录摘要**

这种方法背后的核心思想是找到所有句子之间的相似性，并返回具有最大相似性得分的句子。我们使用**余弦相似度**作为相似度矩阵，使用 **TextRank** 算法根据句子的重要性进行排序。

在了解 TextRank 算法之前，有必要简单说一下 PageRank 算法，TextRank 背后的影响。PageRank 是 Google 使用的一种基于图形的算法，用于根据搜索结果对网页进行排序。PageRank 首先创建一个图，以页面为顶点，页面之间的链接为边。对每个页面计算 PageRank 得分，基本上就是用户访问那个页面的概率。[这里](https://web.stanford.edu/class/cs54n/handouts/24-GooglePageRankAlgorithm.pdf)是一篇很好的解释 PageRank 算法的论文。

## **TextRank(文本摘要背后的魔力)**

TextRank 与 PageRank 的相似性可以用以下几点来强调:

1.  文本单元(句子)被用来代替页面作为图形中的顶点。
2.  句子之间的相似度被用作边缘而不是链接。
3.  代替页面访问概率，句子相似度被用来计算排名。

TextRank 算法从自然语言文本生成图。任何基于图的算法的基本思想都是基于“投票”或“推荐”。当一个顶点链接到另一个顶点时，它基本上是在为那个顶点投票。一个顶点的票数越高，该顶点的重要性就越高。顶点得分取决于两个因素:

*   投票数。
*   为它投票的顶点的分数(重要性)。

**text rank 中要遵循的步骤**

1.  从原文中提取所有句子。
2.  根据句子中存在的标记(单词)为所有句子创建向量。
3.  计算每个句子对之间的余弦相似度。创建一个 n×n 相似度矩阵，其中 n 是句子的数量。
4.  使用相似性矩阵创建一个图，其中每个顶点代表一个句子，两个顶点之间的边代表相似性。
5.  根据相似性得分对句子进行排序，并返回要包含在摘要版本中的前 N 个句子。

> **注意:**余弦相似性的一个简短说明在这里会有帮助。多维空间中任意两个向量之间的余弦距离是使用它们之间角度的余弦来计算的。向量 Va 和 Vb 之间的余弦距离公式可以写成:
> 
> **余弦距离(Va，Vb) = 1-余弦(Va，Vb 之间的角度)**
> 
> 我们可以说，对于相似的向量，余弦距离将是低的，余弦相似度将是高的。

今天的理论到此为止。让我们直接进入有趣的部分，那就是实现。:)

## **用 Python 实现**

让我们从导入所需的库开始。

```
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
import networkx as nx
import re
```

第一个函数用于读取文本并将其转换成句子。我们也将做基本的文本清理，以消除所有的特殊字符。

```
def read_article(text):        
  sentences =[]        
  sentences = sent_tokenize(text)    
  for sentence in sentences:        
    sentence.replace("[^a-zA-Z0-9]"," ")     
return sentences
```

接下来，我们将从句子中创建向量，并计算这些向量之间的余弦相似度。

```
def sentence_similarity(sent1,sent2,stopwords=None):    
  if stopwords is None:        
    stopwords = []        
  sent1 = [w.lower() for w in sent1]    
  sent2 = [w.lower() for w in sent2]

  all_words = list(set(sent1 + sent2))   

  vector1 = [0] * len(all_words)    
  vector2 = [0] * len(all_words)        
  #build the vector for the first sentence    
  for w in sent1:        
    if not w in stopwords:
      vector1[all_words.index[w]+=1                                                             
  #build the vector for the second sentence    
  for w in sent2:        
    if not w in stopwords:            
      vector2[all_words.index(w)]+=1 

return 1-cosine_distance(vector1,vector2)
```

接下来我们创建一个 n×n 维的相似性矩阵来存储相似性值。

```
def build_similarity_matrix(sentences,stop_words):
  #create an empty similarity matrix
  similarity_matrix = np.zeros((len(sentences),len(sentences)))for idx1 in range(len(sentences)):
    for idx2 in range(len(sentences)):
      if idx1!=idx2:
        similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1],sentences[idx2],stop_words)return similarity_matrix
```

最后 main 函数调用管道中的所有上述函数。

```
def generate_summary(text,top_n):
  nltk.download('stopwords')    
  nltk.download('punkt') stop_words = stopwords.words('english')    
  summarize_text = [] # Step1: read text and tokenize    
  sentences = read_article(text) # Step2: generate similarity matrix            
  sentence_similarity_matrix = build_similarity_matrix(sentences,stop_words) # Step3: Rank sentences in similarity matrix
   sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
   scores = nx.pagerank(sentence_similarity_graph) # Step4: sort the rank and place top sentences
  ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)

  # Step5: get the top n number of sentences based on rank
  for i in range(top_n):
    summarize_text.append(ranked_sentences[i][1]) # Step6 : output the summarized version
  return " ".join(summarize_text),len(sentences)
```

详细解释请参考这里的完整代码[。](https://github.com/sawansaxena/Extractive-Text-Summarization)

**让我们看看结果**

以下是我作为输入给出的原文。

```
WASHINGTON - The Trump administration has ordered the military to start withdrawing roughly 7,000 troops from Afghanistan in the coming months, two defense officials said Thursday, an abrupt shift in the 17-year-old war there and a decision that stunned Afghan officials, who said they had not been briefed on the plans.
President Trump made the decision to pull the troops - about half the number the United States has in Afghanistan now - at the same time he decided to pull American forces out of Syria, one official said.
The announcement came hours after Jim Mattis, the secretary of defense, said that he would resign from his position at the end of February after disagreeing with the president over his approach to policy in the Middle East.
The whirlwind of troop withdrawals and the resignation of Mr. Mattis leave a murky picture for what is next in the United States’ longest war, and they come as Afghanistan has been troubled by spasms of violence afflicting the capital, Kabul, and other important areas. 
The United States has also been conducting talks with representatives of the Taliban, in what officials have described as discussions that could lead to formal talks to end the conflict.
Senior Afghan officials and Western diplomats in Kabul woke up to the shock of the news on Friday morning, and many of them braced for chaos ahead. 
Several Afghan officials, often in the loop on security planning and decision-making, said they had received no indication in recent days that the Americans would pull troops out. 
The fear that Mr. Trump might take impulsive actions, however, often loomed in the background of discussions with the United States, they said.
They saw the abrupt decision as a further sign that voices from the ground were lacking in the debate over the war and that with Mr. Mattis’s resignation, Afghanistan had lost one of the last influential voices in Washington who channeled the reality of the conflict into the White House’s deliberations.
The president long campaigned on bringing troops home, but in 2017, at the request of Mr. Mattis, he begrudgingly pledged an additional 4,000 troops to the Afghan campaign to try to hasten an end to the conflict.
Though Pentagon officials have said the influx of forces - coupled with a more aggressive air campaign - was helping the war effort, Afghan forces continued to take nearly unsustainable levels of casualties and lose ground to the Taliban.
The renewed American effort in 2017 was the first step in ensuring Afghan forces could become more independent without a set timeline for a withdrawal. 
But with plans to quickly reduce the number of American troops in the country, it is unclear if the Afghans can hold their own against an increasingly aggressive Taliban.
Currently, American airstrikes are at levels not seen since the height of the war, when tens of thousands of American troops were spread throughout the country. 
That air support, officials say, consists mostly of propping up Afghan troops while they try to hold territory from a resurgent Taliban.
```

来源:https://www.nytimes.com

下面是三行输出的总结版本。

```
The whirlwind of troop withdrawals and the resignation of Mr. Mattis leave a murky picture for what is next in the United States’ longest war, and they come as Afghanistan has been troubled by spasms of violence afflicting the capital, Kabul, and other important areas. 
They saw the abrupt decision as a further sign that voices from the ground were lacking in the debate over the war and that with Mr. Mattis’s resignation, Afghanistan had lost one of the last influential voices in Washington who channeled the reality of the conflict into the White House’s deliberations. 
Though Pentagon officials have said the influx of forces - coupled with a more aggressive air campaign - was helping the war effort, Afghan forces continued to take nearly unsustainable levels of casualties and lose ground to the Taliban.
```

输出显示了具有最大相似性得分的前 3 个句子。您可以根据需要更改汇总版本中的行数。

在生成文本摘要的核心 python 代码之上，我使用 Flask API 将其与 web 应用程序集成，并将其部署在云上。这使得从用户处获取输入文本并显示生成的摘要作为结果变得容易。

**重要链接**

*   这个项目的代码可以在我的 Github 库的这里找到。
*   [这里的](https://nlp-extractive-summary.herokuapp.com/)是我部署在云上的 web 应用程序的链接。
*   [这个](https://www.linkedin.com/in/sawan-saxena-640a4475/)是我喜欢的丁简介。

**参考文献**

*   将秩序带入文本:[https://www.linkedin.com/in/sawan-saxena-640a4475/](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)
*   了解文本摘要并使用 python 创建自己的摘要器:[https://towards data science . com/understand-Text-summary-and-create-your-own-summary zer-in-python-b 26 a9 f 09 fc 70](https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70)
*   使用 TextRank 算法的文本摘要介绍:[https://www . analyticsvidhya . com/blog/2018/11/Introduction-Text-summary-Text rank-python/](https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/)

在下一篇文章中，我将讨论文本摘要的抽象方法。请在评论中告诉我你的反馈。

感谢阅读。:)