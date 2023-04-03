# Python 中的“yield”是做什么的？

> 原文：<https://medium.com/analytics-vidhya/what-does-the-yield-do-in-python-44dfaf978906?source=collection_archive---------8----------------------->

![](img/07f70367ca456c4741b8e312261a74ea.png)

[https://www . Forbes . com/sites/robis bitts 2/2018/08/13/its-all-about-yield/？sh=5938e0fd7393](https://www.forbes.com/sites/robisbitts2/2018/08/13/its-all-about-yield/?sh=5938e0fd7393)

**yield** 是 **Python** 中的一个关键字，用于从一个函数返回而不破坏其局部变量的状态，当调用该函数时，执行从最后一个 **yield** 语句开始。任何包含 **yield** 关键字的函数都被称为生成器

要理解`yield`是做什么的，你必须理解*发生器*是什么。在你理解生成器之前，你必须理解 *iterables* 。

# 可重复的

当您创建列表时，您可以逐个读取它的项目。**逐个读取它的条目称为迭代:**

```
>>> list = [1, 2, 3]>>> for i in list:
...    print(i)=========================Output===================1
2
3
```

`list`是一个*可迭代*。当你使用列表理解时，你创建了一个列表，因此一个 iterable:

```
>>> list = [x*x for x in range(3)]
>>> for i in list:
...    print(i)=========================Output===================0
1
4
```

一切可以用`for... in...`的东西都是可重复的；`lists`、`strings`，文件...

这些 iterables 很方便，因为你可以随心所欲地读取它们，但是你把所有的值都存储在内存中，当你有很多值时，这并不总是你想要的。

# 发电机

生成器是迭代器，一种只能迭代一次的迭代器。发生器不会将所有值存储在存储器中，**它们会动态生成值**:

```
>>> mygenerator = (x*x for x in range(3))
>>> for i in mygenerator:
...    print(i)=========================Output===================0
1
4
```

除了你用`()`代替`[]`之外，都是一样的。但是，你**不能**执行第二次`for i in mygenerator`，因为生成器只能使用一次:它们计算 0，然后忘记它，计算 1，最后一个接一个地计算 4。

# 产量

`yield`是一个关键字，用法和`return`、**一样，只是函数会返回一个生成器**。

```
>>> def createGenerator():
...    list = range(3)
...    for i in list:
...        yield i*i
...
>>> mygenerator = createGenerator() # create a generator
>>> print(mygenerator) # mygenerator is an object!=========================Output===================<generator object createGenerator at 0xb7555c34> >>> for i in mygenerator:
...     print(i)=========================Output===================0
1
4
```

这是一个没用的例子，但是当你知道你的函数将返回一大堆你只需要读一次的值时，这是很方便的。

要掌握`yield`，你必须明白**在调用函数时，你在函数体中写的代码是不运行的。**这个函数只返回生成器对象，这有点棘手:-)

然后，您的代码将从每次`for`使用生成器时停止的地方继续运行。

现在最难的部分是:

第一次`for`调用从你的函数创建的生成器对象时，它将从头开始运行你的函数中的代码，直到它到达`yield`，然后它将返回循环的第一个值。然后，每个后续调用将运行您在函数中编写的循环的另一个迭代，并返回下一个值。这将继续，直到发生器被认为是空的，这发生在函数运行时没有点击`yield`。这可能是因为循环已经结束，或者因为您不再满足某个`"if/else"`。

# 理解迭代的内在机制

迭代是一个包含可迭代程序(实现了`__iter__()`方法)和迭代器(实现了`__next__()`方法)的过程。Iterables 是任何你可以从中获得迭代器的对象。迭代器是让你在可迭代对象上迭代的对象。