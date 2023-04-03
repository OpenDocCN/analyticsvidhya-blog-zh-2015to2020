# 重构雨滴

> 原文：<https://medium.com/analytics-vidhya/refactoring-raindrops-39dcd4bf536e?source=collection_archive---------20----------------------->

![](img/4aaae33293f8ce18cc6a2bb278ee0da2.png)

在我关于这个问题的上一篇文章中，我们介绍了如何[重构 Python 代码，使其在雨滴(又名 Fizz-Buzz)问题中更具可伸缩性](/analytics-vidhya/revisiting-raindrops-in-python-abc4fbb23f4e)。

我用字典重写了解决方案，并在这里结束:

```
def convert(number):
    factors = []
    output = ''
    tu = {1:'Pling', 2:'Plang', 3:'Plong'}

    if number % 3 == 0:
        factors += tu[1]

    if number % 5 == 0:
        factors += tu[2]

    if number % 7 == 0:
        factors += tu[3]

    return output.join(factors) or str(number)
```

如果你发现一系列的“如果”语句有点令人畏缩，你并不孤单！肯定不是干解。你甚至可以称之为湿解(每次都写)。老实说，一开始我甚至不愿意发表它，因为它太不优雅，也不符合 pythonic 语言。所以让我们摆脱那些重复！

首先，请注意我们有一个带编号键的字典(记住:python 中的字典是无序的，这与这个问题没有特别的关系，但是如果数据需要排序，您可能希望使用不同的数据结构，比如 SortedDict)。

我们可以根据关键数字遍历字典，并将这些因素添加到列表中。这可以通过一个“for”循环来完成，或者更好的方法是，将这个“for”循环折叠成一个列表理解:

```
def convert(number):
    factors = []

    d = {3:'Pling', 5:'Plang', 7:'Plong'}

    factors += [v for (k,v) in d.items() if number % k == 0]

    return ''.join(factors) or str(number)
```

这里，我们将“if”条件写成一次，用一个变量(k)代替整数 3、5 和 7。如果有任何数字通过了测试，我们就追加字典值—在这种情况下，如果数字可以被键整除(或者，换句话说，如果商的余数为 0)，我们就遍历字典中的每个键来进行测试。最后，我们根据问题的具体情况，把这个列表连接成一个字符串。

# 在 4 行简单的代码中，我们建立了一段通用的代码，可以处理任何数量的键/值对。

像所有好的写作一样，干净的软件是一个迭代的过程。它有助于注意为什么以及何时进行重构更改，以避免将来重复模式。