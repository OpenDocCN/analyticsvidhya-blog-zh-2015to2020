# 可变和不可变对象(py)

> 原文：<https://medium.com/analytics-vidhya/mutable-and-immutable-objects-py-efe79e5d2d39?source=collection_archive---------21----------------------->

![](img/65a91164c738f07c578071807b898639.png)

在这篇博客中，我将谈论 python 中可变对象和不可变对象之间的区别，这个主题非常重要，因为如果你理解了这种区别，你就可以在修改不同类型的数据时避免意外的行为。

让我们从定义两件事开始，数据类型和变量的 id，如果你以前有编程经验，你会知道存在不同类型的数据，比如 int，float，str，你声明的每个变量都有一个唯一的 id。

在 Python 解释器中，我们可以使用函数 **type()** 检查变量的数据类型，使用函数 **id()** 检查变量的唯一 id:

```
**>>> x** = 10 
**>>> type**(x)
<class 'int'>
**>>> y** = 10 
**>>> type**(y)
<class 'float'>
**>>> id**(x)
10105376
**>>> id**(y)
13990449964
```

# 可变对象

**可变**对象属于 list、dict 和 set 类型，这意味着这些类型的数据可以通过多种方式修改，让我们以数据列表的类型为例:

```
**>>> letters = ['a', 'b', 'c']**
**>>> letters[0] = 'z'
>>> letters[1] = 'y'
>>> letters
['z', 'y', 'c']**
```

在前面的例子中，我们看到了如何使用索引修改列表，但是列表有多种方法，如 append、extend、join。如果你想了解更多关于列表和它们的方法，使用下面的[链接](https://docs.python.org/3/tutorial/datastructures.html)。

# 不可变对象

上的**不可变**相反是所有那些不允许任何类型修改的数据类型，一些不可变类型包括 int、float、bool、string、tuple。

```
**>>> message = 'Holberton'**
**>>> message[0] = 'L'** Traceback (most recent call last):File "<stdin>", line 1, in <module>TypeError: 'str' object does not support item assignment
```

Python 通过使引用同一个字符串值的两个名称引用同一个对象来优化资源。

```
**>>> message1 = 'Holberton'**
**>>> message2 = 'Holberton'
>>> message1 == message2**
True
**>>> message1 is message2**
True 
```

# 为什么对待可变和不可变对象很重要？

正如我在本文开头所解释的，处理可变和不可变对象很重要，因为我们可以避免意外行为，并理解 Python 如何在幕后优化资源

# **参数如何传递给函数，这对可变和不可变对象意味着什么**

在 Python 中，参数总是通过引用传递给函数。调用者和函数共享同一个对象或变量。当我们在函数中改变一个函数参数的值时，无论参数或变量的名称如何，该变量的值在调用程序代码块中也会改变，这意味着当我们传递一个可变对象时，它可以在另一个函数中被修改，这与不可变对象相反

```
>>> def increment(n):
...   n += 1
>>> a = 1
>>> increment(a)
>>> print(a)
a = 1  #keeps the same value
```

通过将函数的返回赋值给' **a'** ，我们正在重新赋值给 **'a'** 以引用一个具有新值的新对象

```
>>> def increment(n):
...   n += 1
...   return n
>>> a = 1
>>> a = increment(a)  
>>> print(a)
a = 2  # 'a' refers to a new object with the new value
```