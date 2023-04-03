# 使用计数矢量器将文本转换为文档术语矩阵

> 原文：<https://medium.com/analytics-vidhya/converting-texts-to-document-term-matrix-using-count-vectorizer-c247127b61ba?source=collection_archive---------7----------------------->

在 ML 中处理文本是最刺激智力的练习之一，但是这个练习的缺点是我们的 ML 算法不能直接处理文本，所有这些 ML 算法都需要数字作为参数。这意味着我们的文本数据应该被转换成数字向量。在自然语言处理行话中，这被称为特征提取。具体来说，文本特征提取。

CountVectorizer 是一个用 sklearn 编写的类，帮助我们将文本数据转换为数字向量。我将使用 sklearn 中提供的示例。先做最重要的事情；我们需要从 sklearn 导入该类，以便访问它。

因为我们将使用 sk learn[【1】](#_ftn1)中提供的例子，我们将有一个包含四个文档的语料库。这是我们要转换成向量的文本数据。假设你还在上高二，一个向量就是一个指针，或者简单的说，一个 1*1 的数组。例如，在下面的语料库中，我们将有四个向量，在每个向量中，我们将存储一些表示文本数据的数字输入。语料库是语言片断的集合，集合可能包括列表、图表、树等。

```
corpus = [**... **    'This is the first document.',**... **    'This document is the second document.',**... **    'And this is the third one.',**... **    'Is this the first document?',**...** ]
```

现在让我们获取类的实例，这样我们就可以利用它的方法，将文本数据转换成 ML 算法所需的数字输入

```
vectorizer = CountVectorizer()
```

这个类有几个参数。其中最重要的是分析器，它有三个选项。Word、char 和 char_ws。在 analyzer 中，您可以指定 n-gram 应该如何处理您的数据，应该将其视为字符还是单词。因此，如果您将分析器设置为 word，将 n-gram_range 设置为(2，2)，它将选择两个相邻的单词并将它们连接起来。如果你的 n-gram_range 设置为(1，2)，我们的字典将取一个单词加对(相邻对)。既然我们已经初始化了这个类，我们就有了一个可以用来访问其中的方法的对象。其中有一个名为 fit_transform 的函数()，我们将把语料库作为参数，并将文本数据转换为向量。这种方法利用一种称为单词包(BoW)的方法来实现这一点。

```
X = vectorizer.fit_transform(corpus)
```

一袋单词

单词包所做的，类似于 python 中 flatten()函数所做的；

1.**它首先将数组形状折叠成一维，然后移除所有重复的数组**。因此，我们将得到如下输出:

'这个'，'是'，'这个'，'第一个'，'文件'，'第二个'，'第三个'，'一个'

**但是 get_feature_names()函数会返回这样的结果:**

和'，'文档'，'第一个'，'是'，'第一个'，'第二个'，'第三个'，'这个'

这是因为数组是按字母顺序排序的。

**2。** **它使用我们得到的字典得到文档-术语矩阵向量**

我们的字典里有 9 个元素。因此，在数组的每个向量(索引)中，我们有 9 个元素。

类似这样的事情(这是在计算之前)

```
[ 0  0  0  0  0  0  0  0  0 ][ 0  0  0  0  0  0  0  0  0 ][ 0  0  0  0  0  0  0  0  0 ][ 0  0  0  0  0  0  0  0  0 ]
```

记住所有这些数字，代表我们的字典'和'，'文件'，'第一'，'是'，'一'，'第二'，'的'，'第三'，'这'

对于第一个元素，你问它是否出现在处理的文档中，它出现了多少次。因为“这是第一份文件。”我们看到字典中的第一个术语“and”在我们的文档中没有出现(我们处理的是第一个文档)，所以我们用 0 替换它的索引。第二个术语是“文档”，它在文档中出现过一次，所以我们用 1 替换第二个索引。最后，对于第一个向量，应该是这样的。

```
[0 1 1 1 0 0 1 0 1]
```

引文

[sci kit-learn:Python 中的机器学习](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)，Pedregosa *等人*，JMLR 12，第 2825–2830 页，2011 年。

[https://sci kit-learn . org/stable/modules/generated/sk learn . feature _ extraction . text . count vectorizer . html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)