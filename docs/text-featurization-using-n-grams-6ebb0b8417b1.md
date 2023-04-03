# 使用 n 元语法的文本特征化

> 原文：<https://medium.com/analytics-vidhya/text-featurization-using-n-grams-6ebb0b8417b1?source=collection_archive---------25----------------------->

![](img/2c4fc215141e5d91be3df203208bd2c9.png)

由[万泰传媒](https://unsplash.com/@vantaymedia?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

# 单字符

*   在这里，文档中的每个单词都是一个维度。
*   例如，假设有如下给出的文档语料库:

1.  这辆车开起来很好，但是很贵。
2.  这辆车很贵，但开起来很好。

上述文档语料库的一元语法如下所示:

`['This', 'car', 'drives', 'good', 'and', 'is', 'expensive', 'very']`

> **注 1:** 一个文本，可以是一个单词或一个句子，在自然语言处理中被称为文档。
> 
> 注 2: 这些文档的集合称为文档语料库。

# 二元语法

*   在这里，每一对连续的单词都是一个维度。
*   使用上述文档语料库的使用示例:

上述文档语料库的二元语法如下所示:

`['This car', 'car drives', 'drives good', 'good and', 'and is', 'is expensive', 'car is', 'is very', 'very expensive', 'and drives']`

# 三元语法

*   在这里，每三个连续的单词都是一个维度。
*   使用上述文档语料库的使用示例:

上述文档语料库的三元语法如下所示:

`['This car drives', 'car drives good', 'drives good and', 'good and is', 'and is expensive', 'This car is', 'car is very', 'is very expensive', 'expensive and drives', 'and drives good']`

# n 元语法

*   这里，每一组 n 个连续的单词就是一个维度。

# 通过 Sklearn 实现 n-grams

*   查看 jupyter 的笔记本文件，[此处](https://github.com/deveshSingh06/Natural_Language_Processing/blob/master/3.%20Uni-gram%2C%20Bi-gram%2C%20Tri-gram%2C%20n-gram.ipynb)。

# **结论:**

*   二字组和三字组在单词袋中非常有用。
*   Uni-gram 丢弃序列信息。
*   而二元语法、三元语法、…、n 元语法保留了一些序列信息。

> 在 GitHub 上关注我: [deveshSingh06](https://github.com/deveshSingh06)