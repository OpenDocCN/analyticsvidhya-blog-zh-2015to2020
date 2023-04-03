# Apache Spark 上的缩放空间文档/文本相似性(NLP)

> 原文：<https://medium.com/analytics-vidhya/scale-spacy-text-similarity-nlp-on-apache-spark-cce95bda686c?source=collection_archive---------15----------------------->

![](img/08129a9c92091eb42eb7f3edcbb92874.png)

安妮·斯普拉特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

在本文中，我们将了解什么是[空间](https://spacy.io/)，以及我们如何利用 Apache Spark 来大规模运行空间。

随着日常业务对机器学习用例的需求不断增长，不同领域对不同类型的机器学习算法的需求也随着数据的不断增长而增长。

## 什么是空间

Spacy 是工业级自然语言处理库，用于处理机器学习的文本数据，提供了大量的预处理、标记化、词干化、命名实体解析、预训练的词向量和为深度学习准备文本等选项。Spacy 还提供了一些构建向量，例如用于 [word2vec](https://en.wikipedia.org/wiki/Word2vec) 。你可以在这里找到更多关于各种语言的模型的细节

[**Apache Spark**](https://en.wikipedia.org/wiki/Apache_Spark) 是一个开源的分布式通用集群计算框架。Spark 提供了一个接口，通过隐式数据并行和容错对整个集群进行编程。

**如何在 spacy 上运行相似度:**

*注:您可以在这里* *找到关于安装空间* [*的详细说明。(我用的是 spacy 的 2.1.8 版本)*](https://spacy.io/usage)

我将探索基于余弦相似性的空间相似性度量。它是通过比较**单词向量**或“单词嵌入”，一个单词的多维意义表示来确定的。

让我们看看来自空间的相似性 api。

我在这篇文章中使用的是 **en_core_web_lg** 模型，它是在普通爬行中训练的手套向量，有 685k 个键，685k 个唯一向量(300 个维度)

```
import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()
doc1 = nlp("i love my pet dog")
doc2 = nlp("Maggie is my lovable pet dog !")
print("output:" , doc1.similarity(doc2))
```

输出:0.9822815156578484

这意味着以上两个文档/句子文本 98.22 %相似。

现在的挑战是如何以分布式方式在 spark 上执行，其中每个执行器节点都在计算分布式数据切片的相似性。

这里的想法是在多个执行器上并行运行 similarity()作为 spark UDF(用户定义的函数)。以下是需要配置的一些重要因素，以便在 spacy on spark 上运行和扩展:

我的所有代码都打包在 MySpacyClassObject 类中。为了避免每次计算相似度都要反复下载 en_core_web_lg，我将 en_core_web_lg 存储在每个执行器上的全局变量中。这将加快计算速度，因为 en_core_web_lg (~789 MB)已经加载到 executor 内存中

```
@staticmethod
*def* get_spacy():

    *if* "nlp" *not in* globals():
        globals()["nlp"] = en_core_web_lg.load()

    *return* globals()["nlp"]
```

**创造火花 UDF:**

```
*def* text_similarity(dsc1, *dsc2*):
    '''
    Load spacy and calculate similarity measure
    '''
    *if* (dsc1 *is None or dsc2 is None*):
        *return* float(-1)
    *if* ((len(str(dsc1)) < 1 *or* str(dsc1) == "") *or* (len(str(*dsc2*)) < 1 *or* str(*dsc2*) == "")):
        *return* float(-1)

    nlp_glob = MySpacyClassObject.get_spacy()

    *return* float(nlp_glob(dsc1).similarity(nlp_glob(*dsc2*)))#define your udf 
text_similarity_UDF = udf(*lambda arr*: MySpacyClassObject.text_similarity(arr[0], arr[1]), FloatType())
```

如何使用 UDF:

```
#You can read data from any data source per your need to create #spark data frame to use UDF. I am reading table from hive which #already has two columns with textual data for similaritytext_similarity_DF = spark.read.table(
    "your_schema.table_name").repartition(500)
text_similarity_with_measure = text_similarity_UDF \
    .withColumn("similarity",
                text_similarity_UDF(array(text_similarity_DF['text1'],
                                   text_similarity_DF['text2']))) \
    .select(text_similarity_DF['text1']
            , "similarity"
            , text_similarity_DF['text2'])#Persist dataframe text_similarity_with_measure back in hive
text_similarity_with_measure.write.mode('overwrite').saveAsTable(your_output_tabble_name)
```

**提高性能的重要考虑因素和挑战:**

*①****。*** *连载* : Spacy 2。*版本支持序列化。确保您使用的是版本> 2。

2.确保 **en_core_web_lg** (或者您正在使用的其他模型)在每个执行器上都可用，以便您可以读取和加载模型以获得更好的性能。另一种方法是你仍然可以让你的 spark 程序只下载一次，然后继续重用它(请看上面的 get_spacy()静态方法)。

3.确保每个执行器都安装了空间并可供使用。