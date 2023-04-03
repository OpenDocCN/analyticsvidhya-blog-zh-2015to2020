# 机器学习中的概率|第一部分

> 原文：<https://medium.com/analytics-vidhya/probabilities-in-machine-learning-bf9344e93b9?source=collection_archive---------27----------------------->

![](img/79baaa3d227d6eb0b51047fde3652fe4.png)

# 简介:

当人类基于有限的经验和信息做出决策时，我们会做出推论来填补空白。在机器学习中，这是通过朴素贝叶斯分类来完成的。

# 什么是朴素贝叶斯分类？

朴素贝叶斯分类是一种通过对特征的概率实施贝叶斯定理来对数据进行分类的算法。

## 朴素贝叶斯分类的优点:

*   处理小型数据集

与传统的神经网络不同，在传统的神经网络中，每个神经元与其他每个神经元直接相连，概率被认为是独立的。

*   计算不密集

与驱动神经网络的权重不同，朴素贝叶斯分类的参数不会在每次迭代中改变。这使得该算法的计算量大大降低。

## 朴素贝叶斯分类的缺点:

*   不擅长学习大数据

当数据足以优化所有参数时，神经网络的复杂映射胜过朴素贝叶斯算法的简单结构。

# 在 Python 中实现朴素贝叶斯分类；

朴素贝叶斯分类实际上非常简单。让我们以分类一串文本来自垃圾邮件还是合法邮件为例。

## **第一步|创建数据:**

```
len_normal = 500normal_dict = {
    'Hello':10,
    'Dinner':10,
    'Dear':400,
    'Please':300,
    'Money':0
}len_spam = 150spam_dict = {
    'Hello':100,
    'Dinner':0,
    'Dear':0,
    'Please':2,
    'Money':100
}
```

这是在正常电子邮件或垃圾邮件的数据集中可以找到的某个单词的数量。len_normal 和 len_spam 稍后用于计算该术语出现在电子邮件中的概率。

## 第 2 步|将计数改为概率:

```
def count_to_probability(normal_dict,spam_dict,len_normal,len_spam):
    for term in normal_dict:
        normal_dict[term] = normal_dict[term]/len_normal
    for term in spam_dict:
        spam_dict[term] = spam_dict[term]/len_spam
    return normal_dict,spam_dict
```

通过将每个计数除以所有正常或垃圾邮件中的总字数，一个单词出现在电子邮件中的概率。

## 步骤 3|计算新字符串的概率:

```
X = 'Hello Please Money'
def calculate_probability(dictionary,X):
    split = X.split()
    probability = 1
    for term in split:
        probability *= dictionary[term]
    return probability
```

根据贝叶斯定理，通过乘以相应项的概率，可以找到整个字符串是垃圾邮件还是正常的概率。我不打算深入贝叶斯定理的细节，但是你可以在这里[研究一下](https://www.youtube.com/watch?v=HZGCoVF3YvM)。计算出最终概率后，比较将该函数应用于垃圾邮件字典时的值。较高的值是网络的最终预测值。

**奖励步骤|填充:**

```
def padding(dictionary,alpha):
    for term in dictionary:
        dictionary[term] += alpha
    return dictionary
```

您会注意到，字符串 X 中的一些单词在字典中的值为 0。这使得很难做出好的预测，因为当值乘以 0 时，计算的概率也将是 0。一种常见的做法是为字典中的每个术语添加一个 alpha 值，以防止值为 0。

# 感谢您阅读本文！