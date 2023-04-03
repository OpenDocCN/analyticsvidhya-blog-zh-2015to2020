# 用 Python 漂亮的打印！

> 原文：<https://medium.com/analytics-vidhya/pretty-printing-in-python-277bc4db2847?source=collection_archive---------13----------------------->

![](img/f25433c4d0a8cdf463959f614074340e.png)

克里斯·里德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

> Python 中的 ictionaries 可能会变得很长，这通常会变得混乱，使人更难阅读。仅仅在终端上打印字典是不行的。那么，我们如何让它对程序员来说是可读的呢？幸运的是，在庞大的 python 库集合中有一个库可以做我们想要的事情。让我们看看我们如何能做到这一点。

# pprint(“漂亮打印”)模块:

`pprint`模块有一些功能有助于比`print()`提供的功能更清晰地显示字典中的条目。这两个功能是`pprint()`和`pformat()`。首先，我们将看看与程序中的`print()`函数和最后的`pformat()`函数相比，`pprint()`函数如何增强字典的可读性。

## 打印():

```
message = 'It was a bright cold day in April, and the clocks were striking thirteen.'count = {}for character in message:
  count.setdefault(character, 0)
  count[character] = count[character] + 1print(count)
```

这是一个程序，用于计算特定字符在给定的`message`中出现的次数，并以字典的形式存储结果，并通过使用`print()`函数打印出来。我们从这个程序得到的输出看起来像这样:

```
{' ': 13, ',': 1, '.': 1, 'A': 1, 'I': 1, 'a': 4, 'c': 3, 'b': 1, 'e': 5, 'd': 3, 'g': 2, 'i': 6, 'h': 3, 'k': 2, 'l': 3, 'o': 2, 'n': 4, 'p': 1, 's': 3, 'r': 5, 't': 6, 'w': 2, 'y': 1}
```

这根本不可读，但实际上是一场灾难。

## pprint():

```
**import pprint**message = 'It was a bright cold day in April, and the clocks were striking thirteen.'count = {}for character in message:
  count.setdefault(character, 0)
  count[character] = count[character] + 1**pprint.pprint**(count)
```

这是相同的程序，打印相同的字典，但是现在使用来自`pprint`模块的`pprint()`函数。让我们看看这个函数有什么魔力:

```
{' ': 13,
 ',': 1,
 '.': 1,
 'A': 1,
 'I': 1,
 'a': 4,
 'c': 3,
 'b': 1,
 'e': 5,
 'd': 3,
 'g': 2,
 'i': 6,
 'h': 3,
 'k': 2,
 'l': 3,
 'o': 2,
 'n': 4,
 'p': 1,
 's': 3,
 'r': 5,
 't': 6,
 'w': 2,
 'y': 1}
```

这一次，当程序运行时，输出看起来更加清晰，键已经排序。

当字典本身包含嵌套列表或字典时，`pprint.pprint()`函数特别有用。

## pformat():

如果您想以字符串值的形式获取美化后的文本，而不是将其显示在屏幕上，请调用`pprint.pformat()`。这两条线彼此等价:

```
pprint.pprint(someDictionaryValue) print(pprint.pformat(someDictionaryValue))
```

## 链接到 pprint 模块文档:

 [## pprint —数据漂亮打印机— Python 3.8.5 文档

### 源代码:Lib/pprint.py 该模块提供了在一个…

docs.python.org](https://docs.python.org/3/library/pprint.html#:~:text=pformat%20%28object%29&text=Return%20the%20formatted%20representation%20of%20object.,passed%20to%20the%20PrettyPrinter%20constructor.&text=Print%20the%20formatted%20representation%20of,stream%2C%20followed%20by%20a%20newline.) 

感谢阅读这篇文章。我希望这可能对你有所帮助。有什么建议别忘了拍拍这篇文章回应一下。