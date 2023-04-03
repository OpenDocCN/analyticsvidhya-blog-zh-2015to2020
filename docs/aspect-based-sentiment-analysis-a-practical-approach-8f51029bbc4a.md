# 使用 Python 实现基于方面的情感分析

> 原文：<https://medium.com/analytics-vidhya/aspect-based-sentiment-analysis-a-practical-approach-8f51029bbc4a?source=collection_archive---------0----------------------->

![](img/fb0a956d72e999fec474603446a644b8.png)

Pexels.com

基于方面的情感分析也称为基于特征的情感分析，是一种从给定文本及其各自的情感中找出各种特征、属性或方面的技术。

在本文中，我们将研究如何使用 Python 和各种 NLP 工具(如 StanfordNLP 和 NLTK)实现 ABSA。

我提到的关于代码实施的论文是由 **Nachiappan Chockalingam** 发表的，他在其中非常详细地解释了关于 ABSA 的一切。我强烈建议您首先阅读这篇令人惊叹的论文，然后跳到代码上，因为它会让您非常清楚代码中发生了什么。

论文:[使用特定领域的情感分数对产品评论进行简单有效的基于特征的情感分析](https://www.polibits.gelbukh.com/2018_57/Simple%20and%20Effective%20Feature%20Based%20Sentiment%20Analysis%20on%20Product%20Reviews%20using%20Domain%20Specific%20Sentiment%20Scores.pdf)

# 我们开始吧

> 安装基本库

*   打开您的终端并安装以下库

```
pip install pandas
pip install numpy
pip install nltk
pip install stanfordnlp
```

*   PyTorch 安装请访问这个 [PyTorch 网站](https://pytorch.org/)。

> 导入库

*   打开您的 Jupyter 笔记本并导入以下库，

*   现在下载 Stanford English 模型和一些 nltk 工具，稍后将用于提取文本中的依存关系和其他文本预处理任务。

*   创建一个样本文本审查，我们将执行 ABSA。

```
txt = "The Sound Quality is great but the battery life is very bad."
```

*   将文本小写，并将句子标记出来。

*   现在，对<sentlist>中的每个句子进行标记，并执行词性标记，将其存储到标记列表中。</sentlist>

*   现在有很多情况下，一个特征由多个单词表示，所以我们需要首先通过将多个单词特征连接成一个单词特征来处理这个问题。

*   标记新句子。

*   现在我们将使用斯坦福 NLP 依存解析器来获取每个单词之间的关系。

*   现在我们将只从<dep_node>中选择那些可能包含特性的子列表。</dep_node>

*   现在使用<dep_node>列表和<featurelist>我们将确定特征列表中的这些特征与哪个单词相关。</featurelist></dep_node>

*   如你所见，我们已经有了特征词，以及每个词的相关词列表。
*   现在只从<fcluster>中选择特性名词列表。</fcluster>

*   这样，我们就有了特征列表和它们在句子中各自的情感词，现在你要做的就是检查情感词是积极的，消极的还是中性的。

> 完整代码

# 结论

*   至此，我们已经看到了使用各种 NLP 工具和技术的基于方面的情感分析的基本实现。
*   这里解释的代码可以通过添加各种 NLP 预处理技术得到很大的改进，如共指消解、俚语词去除、否定处理、讽刺检测等。
*   如果你觉得这篇文章很有用，请**鼓掌并分享**，并随时询问关于这篇文章的任何疑问。随时欢迎任何改进代码的建议或方法:)
*   在 [LinkedIn](https://www.linkedin.com/in/rohan-goel-b0a6ab160/) 或 [GitHub](https://github.com/RG2021) 上与我联系。

# 参考

*   [https://medium . com/@ nitesh 10126/aspect-based-opinion-analysis-in-product-reviews-unsupervised-way-fb0 b 38 EAD 501](/@nitesh10126/aspect-based-sentiment-analysis-in-product-reviews-unsupervised-way-fb0b38ead501)
*   [https://www . poli bits . gelbukh . com/2018 _ 57/Simple % 20 and % 20 effective % 20 feature % 20 based % 20 peption % 20 analysis % 20 on % 20 product % 20 reviews % 20 using % 20 domain % 20 specific % 20 peption % 20 scores . pdf](https://www.polibits.gelbukh.com/2018_57/Simple%20and%20Effective%20Feature%20Based%20Sentiment%20Analysis%20on%20Product%20Reviews%20using%20Domain%20Specific%20Sentiment%20Scores.pdf)
*   [https://medium . com/@ pmin 91/aspect-based-opinion-mining-NLP-with-python-a53eb 4752800](/@pmin91/aspect-based-opinion-mining-nlp-with-python-a53eb4752800)

# 更新和修复

*   如果有人面临 StanfordNLP 模块的错误，请尝试使用评论中提到的这个名为“stanza”的替代模块。

```
import stanzastanza.download('en') # download English modelnlp = stanza.Pipeline('en') # initialize English neural pipelinedoc = nlp(finaltxt)dep_node = []for dep_edge in doc.sentences[0].dependencies:dep_node.append([dep_edge[2].text, dep_edge[0].id, dep_edge[1]])for i in range(0, len(dep_node)):if (int(dep_node[i][1]) != 0):dep_node[i][1] = newwordList[(int(dep_node[i][1]) - 1)]print(dep_node)
```