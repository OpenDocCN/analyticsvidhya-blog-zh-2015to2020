# 自然语言处理:实验空间(四)

> 原文：<https://medium.com/analytics-vidhya/natural-language-processing-experimenting-spacy-part-4-smartlake-e3ad670c607a?source=collection_archive---------1----------------------->

![](img/7d0870e86889e50a10457ce9473d77a7.png)

# 介绍

在前 3 篇文章中，我们已经说明了 Google 和 AWS NLP APIs 的用法，并实验了 spacy 库从不同的文档中提取实体和名词。我们选择使用个人简介和工作描述，因为这是一个常见的用例，很容易理解。

我们想比以前的文章走得更远，并尝试更详细的空间。

# 命名实体识别

现在让我们在这个例子中使用 Python 库，因为它提供了比使用 R 库更多的特性(至少就我所理解的而言)。像往常一样，我们需要安装空间库并下载我们想要使用的相应模型(在[https://spacy.io/usage/.](https://spacy.io/usage/)下有更多关于这个的内容)

现在让我们使用 spacy 解析一个文档，并打印对应于“PRODUCT”的命名实体的摘录:

```
import spacy spacy.load('en') 
docCV=nlp(textCV) 
for ent in docCV.ents: 
# Print the entity text and its label 
  if ent.label_=='PRODUCT': 
    print(ent.text, ent.label_,) 
Agile PRODUCT 
Tier 1 PRODUCT
```

小型英国型号的结果并不令人印象深刻，因此中型型号的结果可能有所不同:

```
nlp = spacy.load('en_core_web_md')
docCV=nlp(textCV) 
for ent in docCV.ents: 
  # Print the entity text and its label if ent.label_=='PRODUCT':
    print(ent.text, ent.label_,)
```

事实上，由于没有检测到产品类型的实体，这是相当令人惊讶的。这很大程度上取决于模型是在哪个文本语料库上训练的。如果我们将相同的模型应用于另一个配置文件，我们会得到以下结果:

C++产品
C++产品
Solaris 产品
C++产品

Spacy 检测到 C++和 Solaris 是产品，但不是 Java 或 JavaScript。因此，让我们将我们自己的检测规则添加到模型中，以更接近我们想要的，即识别人们档案中的技术技能:

```
patterns = [{"label": "PROG", "pattern": [{"lower": "java"}]}, {"label": "PROG", "pattern": [{"lower": "javascript"}]}] 
ruler = EntityRuler(nlp, patterns=patterns,overwrite_ents=True) nlp.add_pipe(ruler) 
docCV=nlp(textCV) 
for ents in docCV.ents: 
# Print the entity text and its label 
if ents.label_=='PROG': 
  print(ents.text, ents.label_,) 
Java PROG 
Java PROG 
Java PROG 
Java PROG 
Java PROG
```

因此，我们能够将基于规则的实体识别添加到统计模型中。这允许以快速的方式对模型进行微调，以适应特定的领域和需求。

让我们也分析一下工作描述，看看仅使用统计模型可以识别出哪种实体:

SLQ ORG
7 年日期
计算机科学 ORG
Java GPE
JavaScript ORG
Boosting、Trees ORG
SLQ ORG
红移 ORG
S3、Spark PRODUCT
digital ocean ORG
第三序数
Google Analytics ORG
Adwords ORG
Crimson Hexagon ORG
Map/Reduce ORG
Hadoop ORG
guro bi GPE
MySQL GPE
Business

这很有趣，因为在这种情况下，模型将 Java 检测为一个地缘政治实体，将 JavaScript 检测为一个组织。这是自动完成的，不需要为这个特定的文档编写特定的规则。

总之，我们可以说，实体识别不仅依赖于所使用的统计模型，还依赖于我们正在处理的文档的结构。这比预期的更复杂。

# 记录相似之处

spacy 的一个很好的特性是能够比较标记(单词)、句子和文档之间的语言和语义相似性。为了做到这一点，我们必须使用加载了词向量的空间模型。为了做到这一点，我们需要加载中型网络英语模型或大型模型。让我们使用命令行加载中型模型:

```
python -m spacy download en_core_web_md
```

预训练模型中的每个单词都有相应的向量，并且文档向量将被默认为所有文档标记(单词)的平均向量。将使用文档向量之间的余弦相似度来计算相似度。结果将是一个介于 0 和 1 之间的数字，1 是最高分(0 的余弦值是 1，意味着空间中的向量彼此重合)

让我们从一般的数据科学家工作描述开始，看看会发生什么。

docjob . similarity(docCV)
Out[98]:0 . 48686 . 48686868661

docjob . similarity(doccv 2)
Out[99]:0 . 46867 . 48686868661

由于两份简介都是技术性的，工作描述也包含了大量的技术词汇，所以平均来看，这两份文件非常相似也就不足为奇了。因此，让我们用欧洲议会关于网络安全的另一份出版物来尝试一下:

doccv . similarity(docDoc)
Out[106]:0 . 54686 . 48686868661

毫不奇怪，相似性降低了，尽管目标文档本质上也是技术性的。

这就是现在的全部，下一次，让我们看看如何使用 spacy 进一步优化我们的文档分析。

*原载于 2019 年 5 月 1 日*[*https://smart lake . ch*](https://smartlake.ch/natural-language-processing-experimenting-spacy-part-4/)*。*