# Python 元组与列表-列表和元组之间的比较

> 原文：<https://medium.com/analytics-vidhya/python-tuples-vs-lists-comparison-between-lists-and-tuples-807b5692aafb?source=collection_archive---------8----------------------->

![](img/47a55b67adece2f24671c723cfb3841e.png)

在本文中，我们将尝试解释元组和列表之间的审查差异。它们都是 python 中相似的序列类型。当考虑列表和元组时，有很大的不同。元组是不可变的列表，但列表不是这样。所以，你不能改变它们的大小以及它们的不可变对象。

元组直接引用它们的元素。此外，对象的长度和顺序也很重要。但是列表有一个额外的间接指向外部指针数组的层。这为元组提供了索引查找和解包的速度优势。

使用元组有两个好处。

1.  **清晰**:当你在代码中看到一个元组，你就知道长度信息永远不会改变。
2.  **性能**:一个元组比相同长度的链表使用的内存少。

***例题 1。*** 我们不能改变元组中的条目:

```
>>> a = (1, 'foo')
>>> a[0] = 10
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
```

***例二。*** 如果一个 tuple 中有任何 list(mutable)，我们可以改变它。

```
>>> a = (1, 'foo', [1, 2])
>>> b = (1, 'foo', [1, 2])
>>> a == b
True
>>> b[-1].append(3)
>>> a == b
False
>>> b
(1, 'foo', [1, 2, 3])
```

***例三。*** 元组的大小是固定的，不会过度分配。因此，它可以比需要过度分配以使 ***append()*** 操作高效的列表更紧凑地存储。

```
>>> import sys>>> a = (1, 2, 3)
>>> id(a)
4468514392
>>> sys.getsizeof(a)
72>>> b = [1, 2, 3]
>>> id(b)
4469068104
>>> sys.getsizeof(b)
88
```

***例 4。*** 空元组总是只将一个元组作为单体。它自己会立即返回。此外，它的长度为零。当我们创建一个空元组时，指向 python 已经预分配的元组。因此，他们有相同的地址。他们在节省内存。

```
>>> a = ()
>>> b = ()
>>> a is b
True
>>> id(a)
4390871112
>>> id(b)
4390871112>>> a = (1, 2, 3)
>>> b = tuple(a)
>>> a is b
True
```

***例 5。*** 既然元组是**不可变的**，就不一定要**复制**。列表是**可变的**对象，需要将所有数据**复制**到一个新列表中:

```
>>> a = []
>>> b = []
>>> a is b
False
>>> id(a)
4504237576
>>> id(b)
4504282440>>> a = [1, 2, 3] 
>>> b = list(a)
>>> a is b
False
```

# **tuple 一般用在哪里？**

*   字符串格式。
*   使用变量和参数。
*   迭代字典键值对。
*   从一个函数中返回两个或更多的项。

# 摘要

在本文中，我们解释了元组和列表序列类型。元组(可变)序列更紧凑，更快，更好用。它可以保存列表(可变)对象，并确保在嵌套数据结构中正确使用这些对象。虽然它不能被改变，但是如果它包含一个列表(可变的)对象，它就可以改变。

# 参考

*   [https://docs.python.org/3/library/collections.html](https://docs.python.org/3/library/collections.html)
*   [https://realpython.com/python-lists-tuples/](https://realpython.com/python-lists-tuples/)