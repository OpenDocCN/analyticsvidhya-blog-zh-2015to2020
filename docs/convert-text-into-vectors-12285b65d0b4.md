# 将文本转换成矢量

> 原文：<https://medium.com/analytics-vidhya/convert-text-into-vectors-12285b65d0b4?source=collection_archive---------21----------------------->

自然语言处理是一种处理文本数据的技术，简而言之，它被称为 NLP。它用于语言翻译应用程序、文字处理器、交互式语音响应(IVR)、个人助理应用程序，如 ok google、Siri、Cortana。

有许多方法可以将文本转换成矢量。在这里，我们将讨论最受欢迎的。

# 单词袋(BOG)

这是文本处理中最简单的方法之一，将文本转换成向量。它是数据中单词出现的表示。它涉及两件事

*   所有已知单词的词汇表。
*   已知单词的存在。

它不关心数据中的信息。它只关心数据中单词的频率，这就是为什么它被称为**包。**

# 克

克也用于将文本转换成矢量。首先，它在处理之前将单词转换成组。它可以是 2、3 或 n 组。在这种技术中，单词被称为克。

让我们看看这意味着什么:

*句子:我是 medium 的作家，写关于机器学习的文章。*

**Uni-gram —** 一次只需要一个单词就可以转换成向量。

![](img/9065a10a28bc33525312b721f899dc0d.png)

双元语法 —一次需要两个单词才能转换成向量。

![](img/da5a4e95405b53e9ae5c75e09562e793.png)![](img/0325b29e4072b16ae20931bf769d2849.png)

**Tri-gram** —一次需要三个单词才能转换成向量。

**N 元语法** —一次需要 N 个单词才能转换成向量。

二元/三元/多元语法也保存了句子信息。

# 术语频率(TF)

它使用词频将文本转换成向量。数据中出现的次数除以数据中的总字数。

![](img/0154e8f539928c3eb6c9345b7917691d.png)

# 反向数据频率(IDF)

它用于衡量术语的稀有程度。对于我们正在查看的每个术语，我们获取文档集中的文档总数，并将其除以包含我们术语的文档数。这给了我们一个稀有的标准。

![](img/355a4d5b4795b71c554f16f01c68b5da.png)

# TF-IDF

正如我们上面讨论的

TF =单词在数据中出现的次数/数据中的总单词数

IDF = ln(文档数/该术语出现的文档数)

TF-IDF 得分越高，该术语越罕见，反之亦然。

# 密码

[笔记本](https://github.com/namratesh/Machine-Learning/blob/master/Convert%20text%20into%20vectors.ipynb)此处链接。

**导入库**

![](img/07dc53721a1631fb1ea1c06008013835.png)

**句子**

![](img/3d92f22de97a19e9ebda667faf1bba60.png)

**单词的标记化**

![](img/ee7ae1e99f751860d1c9d32455e8a0c3.png)

**删除停用词**

![](img/677633fe5542ad40d4e5df98f0a3744b.png)

## 克

**双字母组合**

![](img/554d7675044c8f1db9cab96786da36cc.png)

**三元组**

![](img/66b22eb80e63b7c8041360cbfb3be023.png)

**n-gram**

![](img/e8a49631092683a31f3af163b1a46cb4.png)

# TF-IDF

![](img/3f296e3db709e771812697e89cc4e7f5.png)

感谢阅读！！！

欢迎建议！！！

参考资料:

[](https://machinelearningmastery.com/natural-language-processing/) [## 什么是自然语言处理？

### 自然语言处理，简称 NLP，被广泛定义为自然语言的自动操作…

machinelearningmastery.com](https://machinelearningmastery.com/natural-language-processing/) 

[https://towards data science . com/machine-learning-text-processing-1 d5a 2d 638958](https://towardsdatascience.com/machine-learning-text-processing-1d5a2d638958)