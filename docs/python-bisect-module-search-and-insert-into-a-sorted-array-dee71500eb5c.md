# Python 二分模块-搜索并插入到排序数组中

> 原文：<https://medium.com/analytics-vidhya/python-bisect-module-search-and-insert-into-a-sorted-array-dee71500eb5c?source=collection_archive---------20----------------------->

![](img/8acf3e89a64ca208dd6d10d6b47ae089.png)

你有没有想过:
1。在排序后的数组中找到应该插入新元素的位置
2。向排序后的数组中插入一个元素并保持排序顺序

如果是，那么这是给你的。

等分模块允许你在一个排序的数组中搜索一个插入点，并作为它的扩展来插入元素并保持排序的顺序。关键词是数组已经按照某种顺序排序。

正式文档是[这里](https://docs.python.org/3/library/bisect.html)源代码是[这里](https://github.com/python/cpython/blob/master/Lib/bisect.py)

在中定义了以下方法:

```
bisect.bisect_left(*a*, *x*, *lo=0*, *hi=len(a)*)
bisect.bisect_right(*a*, *x*, *lo=0*, *hi=len(a))* bisect.bisect(*a*, *x*, *lo=0*, *hi=len(a)*)
bisect.insort_left(*a*, *x*, *lo=0*, *hi=len(a)*)
bisect.insort_right(*a*, *x*, *lo=0*, *hi=len(a)*)
bisect.insort(*a*, *x*, *lo=0*, *hi=len(a)*)
```

现在，让我们看一些简单的例子来说明这些方法。让我们考虑一个元素已经存在的情况:

`>>> l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> bisect.bisect_left(l, 5)
5
>>> bisect.bisect_right(l, 5)
6
>>> bisect.bisect(l, 5)
6`

正如你在上面看到的，`bisect`与`bisect_left`相似，但是`bisect`返回一个插入点，该插入点位于 *l* 中任何现有条目 5 的后面(右边)。本质上`bisect and bisect_right are equivalent`。你可以在源代码[的第 79 行看到。](https://github.com/python/cpython/blob/master/Lib/bisect.py)

现在，让我们看看如何使用这些方法在一个排序列表中插入:

`>>> l
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> bisect.insort_left(l, 5)
>>> l
[0, 1, 2, 3, 4, *5*, 5, 6, 7, 8, 9]
>>> bisect.insort_right(l, 5)
>>> l
[0, 1, 2, 3, 4, 5, 5, *5*, 6, 7, 8, 9]`

*insort_left* 相当于`a.insert(bisect.bisect_left(a, x, lo, hi), x)`假设 *a* 是 ***已经排序*** 。请记住，O(log n)搜索由缓慢的 ***O(n)插入步骤*** 控制。

上面强调了开发人员在使用它时需要记住的两件事，在进行复杂性分析时要记住它们。

现在为了让这个变得更有趣，我用 lambda 重写了文档中给出的例子:

`>>> get_grade = lambda score, breakpoints=[60, 70, 80, 90], grades=’FDCBA’ : grades[bisect.bisect(breakpoints, score)]`

`>>> [f”Score {score} gets the Grade %s” % (get_grade(score)) for score in [33, 99, 77, 70, 89, 90, 100, -1, -1.0002] if score >= 0]
[‘Score 33 gets the Grade F’, ‘Score 99 gets the Grade A’, ‘Score 77 gets the Grade C’, ‘Score 70 gets the Grade C’, ‘Score 89 gets the Grade B’, ‘Score 90 gets the Grade A’, ‘Score 100 gets the Grade A’]`

好吧，这两行只是让你接触了 *lambda，格式字符串和列表理解和数据验证【注意分数-1 和-1.0002】*；-)是时候去探索了！

现在，对于更高级的开发人员，请查看[这个](https://code.activestate.com/recipes/577197-sortedcollection/)。这是一个使用二等分构建全功能集合类的方法，该集合类具有直接的搜索方法和对键函数的支持。这些键是预先计算好的，以避免在搜索过程中对 key 函数进行不必要的调用。

如果你需要一个例子来理解以上句子，请阅读以下内容:

```
 >>> from pprint import pprint
    >>> from operator import itemgetter

    >>> s = SortedCollection(key=itemgetter(2))
    >>> for record in [
    ...         ('roger', 'young', 30),
    ...         ('angela', 'jones', 28),
    ...         ('bill', 'smith', 22),
    ...         ('david', 'thomas', 32)]:
    ...     s.insert(record)

    >>> pprint(list(s))         # show records sorted by age
    [('bill', 'smith', 22),
     ('angela', 'jones', 28),
     ('roger', 'young', 30),
     ('david', 'thomas', 32)]

    >>> s.find_le(29)           # find oldest person aged 29 or younger
    ('angela', 'jones', 28)
    >>> s.find_lt(28)           # find oldest person under 28
    ('bill', 'smith', 22)
    >>> s.find_gt(28)           # find youngest person over 28
    ('roger', 'young', 30)

    >>> r = s.find_ge(32)       # find youngest person aged 32 or older
    >>> s.index(r)              # get the index of their record
    3
    >>> s[3]                    # fetch the record at that index
    ('david', 'thomas', 32)

    >>> s.key = itemgetter(0)   # now sort by first name
    >>> pprint(list(s))
    [('angela', 'jones', 28),
     ('bill', 'smith', 22),
     ('david', 'thomas', 32),
     ('roger', 'young', 30)]Finding and indexing are O(log n) operations while iteration and insertion are O(n).  The initial sort is O(n log n). Time to impress in your next coding interview!
```

有几个问题:

1.  参数 *lo* 不接受负指数，参见下面的
    
2.  参数 *hi* **接受**负指数，但 ***不使用*** ，因为行为不正确
    `>>> l
    [0, 1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9]
    >>>
    >>> bisect.bisect_right(l, 5.1, 0, -2)
    0`

参考资料:

[https://docs.python.org/3/library/bisect.html](https://docs.python.org/3/library/bisect.html)
https://code . active state . com/recipes/577197-sorted collection/