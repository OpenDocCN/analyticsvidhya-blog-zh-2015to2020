# 关于 Python 字典的一些注释

> 原文：<https://medium.com/analytics-vidhya/some-notes-on-python-dictionaries-b4a0c1f21473?source=collection_archive---------18----------------------->

## 更好地利用 python 字典的几点建议

![](img/0ea432dde24ebfcf2df658087188d884.png)

# 介绍

Python [字典](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)是将键映射到值的灵活容器。知道如何使用字典是任何一个 pythonist 爱好者的必备技能。

这个博客涵盖了 Python 字典上的一些注释，这些注释帮助我改进了我的`dict`游戏。如果您是一名(> =)中级 python 程序员，那么其中的大部分内容您可能已经很熟悉了。但是我仍然希望你会喜欢这篇文章，也许能从中受益。

我试图让这个博客对初学者友好，但是如果你不太熟悉`dict`，你可能会发现[复习](https://realpython.com/python-dicts/)很有用。

我们开始吧。

# 注 1:使用字典的一个基本例子

对于`dict`如何工作的基本演示，让我们计算列表中项目的频率。这是一个非常明显的`dict`用例。
我们将以尽可能最基本的方式实现这一点开始，并随着我们的进展改进我们的代码。稍后，我们将使用一个专门的容器来做这件事。

# 注意 2: `setdefault '优雅地处理丢失的键

我们可以在上面的代码中避免使用`if-else`子句，并使用`setdefault`方法处理丢失的键。
如果项目/关键字在字典中，`setdefault`返回其值。否则，它插入具有给定值的项并返回它。
在`setdefault`的[文档](https://docs.python.org/3.8/library/stdtypes.html#dict.setdefault)中的解释非常直接，如下所示:

> `setdefault(key[,default])`
> 如果键在字典中，返回其值。如果不是，插入值为 default 的键并返回 default。默认为“无”。

这不仅减少了代码的行数，而且使代码更具可读性和 pythonic 性。
让我们使用`setdefault`来改进注释 1 中的示例。

## 这里有另一个“setdefault”的例子来使它更清楚，因为这是一个重要的方法。

下面的代码从一个列表中分离出所有的奇数和偶数，并将其放入自己的列表/桶中。
一个桶包含所有奇数，而另一个桶包含所有奇数。我们将使用`num%2` ( [一个散列函数](https://en.wikipedia.org/wiki/Hash_function))来创建密钥，因为`odd_num%2 == 1`和`even_num%2 == 0`。

当然，你可以争辩说，在上面的代码中，我们可以简单地将`buckets`初始化为`bucket={'odd':[],'even':[]}`。但是想想那些您事先不知道键的非平凡用例，例如，读取一个包含县及其城市的`.csv`文件，每行是`<country_name>,<city_name>`，您需要将所有的国家及其城市分组；具有任意数量桶的复杂散列函数等。如果您愿意，可以使代码更短。

```
for num in num_list:
    buckets.setdefault(num%2,[]).append(num)
```

为了便于比较，下面是同一个例子中没有使用`setdefault`的两个(故意难看的)替代方案。

# 注 3:使用“collections.Counter()”对对象进行计数

回到我们之前计算频率的例子，这是一个重复的任务，python 在`collections`模块中有一个内置的`Counter`。`Counter`附带了一些有用的方法，比如`most_common(n)`来快速找出最频繁出现的 n 个条目，非常类似于`dict` ( [docs](https://docs.python.org/3/library/collections.html#collections.Counter) )。

# 注 4:词典释义

[理解](https://docs.python.org/3/tutorial/datastructures.html?highlight=comprehension#list-comprehensions)是 Python 中最有用的工具之一，当然，它也受到字典的支持。语法与 list comprehensions 的
基本相同，不同之处在于使用了`{..}`而不是`(..)`，并且要求您定义一个`key: value`对。从下面的代码示例中应该很明显。

请注意，字典理解不应该用于任何副作用，如增加变量或打印等。

# 注 5:注 python 字典和“OrderedDict”中的插入顺序

从 Python 3.7+开始，字典保持插入顺序。但是，不建议依赖它。许多流行的库(和程序员)认为`dict`中的顺序并不重要，因为它通常并不重要。

如果您想保持插入顺序，您应该使用`OrderedDict` ( [docs](https://docs.python.org/3/library/collections.html#collections.OrderedDict) )，它会记住默认情况下插入项目的顺序。除了清楚地传达你的意图之外，它还有一个额外的好处，就是不必太担心向后兼容性。

如果你想了解更多这方面的知识，我强烈推荐格雷格·甘登伯格的文章。

# 注意 6:字典键需要是可散列的，并且事物也是可散列的。

对于作为字典中的键的对象，它需要是可哈希的。可散列对象的例子有`int`、`str`、`float`等。具体来说，它需要满足以下三个要求。

1.  它应该通过一个`__hash__()` dunder 方法支持`hash()`函数，该方法的值在对象的生命周期内不会改变。
2.  它通过`__eq__()`方法支持相等比较。
3.  如果`a == b`是`True`，那么`hash(a) == hash(b)`也必须是`True`。

基于同样的原因，一个`tuple`可以成为`dict`中的一个键，而一个`list`则不能。([参考](https://stackoverflow.com/questions/7257588/why-cant-i-use-a-list-as-a-dict-key-in-python)

关于这一点，默认情况下，用户定义的类型是可哈希的。这是因为它们的哈希值是它们的`id()`并且它们都不相等。
需要注意的一点是，`__hash__()`、
`__eq__()`的定制实现应该只考虑那些在对象生命周期内不会改变的对象属性。([进一步信息](https://stackoverflow.com/questions/4901815/object-of-custom-type-as-dictionary-key))

# 注 7:字典速度快，但用空间换时间

在内部，`dict`使用散列表。通过设计，这些哈希表是稀疏的，这意味着它们不是非常节省空间。对于大量的记录，将它们存储在紧凑的对象中可能更节省空间，比如`tuples`。

即使`dict`有很大的内存开销，只要它适合内存，就允许快速访问。([参考](https://stackoverflow.com/questions/327311/how-are-pythons-built-in-dictionaries-implemented))

# 注 8:如果唯一性是您所需要的，请使用“集合”

从一个收藏中找出所有独特的物品是常有的事。使用带有虚拟值的`dict`可能很有诱惑力，因为 dict 中的所有键在默认情况下都是惟一的。

在这种情况下，最好使用`set`来代替。Python `set`保证唯一性。它也有很好的性质，类似于数学中的集合，如并和交。与`dict`类似，`set`中的元素必须是可散列的。

但是如果情况需要一个可散列的`set`，你将不得不用`frozenset`来代替，因为`set`是不可散列的。`frozenset`是`set`的哈希版本，所以可以放在`set`里面。

因为这项工作是关于字典的，所以我会给你留一个到文档的[链接](https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset)来学习更多关于集合的知识。

# 摘要

1.  通过计算列表中项目的频率来基本演示`dict`如何工作。
2.  丢失的钥匙可以通过`setdefault`优雅的处理。
3.  `collections.Counter`是用于计数易腐烂物品的专用容器
4.  创建词典时支持理解。
5.  虽然`dict`维持了秩序，但这不是应该依赖的东西。同样最好使用`OrderedDict`。
6.  字典键需要是可哈希的。默认情况下，用户定义的对象是可哈希的。
7.  字典速度快，但空间效率低，因为它使用稀疏哈希表。
8.  当您只需要查找唯一值时，请使用集合而不是字典。

# 结论

这篇博客介绍了一些使用 Python 字典的注意事项。我希望这对你来说是一个知识性和愉快的阅读。如果你发现任何错误，请留下评论，以便我可以纠正它。也欢迎任何反馈和建议。

你可以在我的 Github 中找到这个博客的 Jupyter 笔记本版本。

感谢您的阅读。

# 进一步阅读

我强烈推荐卢西亚诺·拉马尔霍写的《流畅的 Python》这本书，它非常深入地涵盖了这个博客中的所有主题。
这个博客也很大程度上受到了他的精彩著作的启发。