# 可迭代&用 Python 创建生成器

> 原文：<https://medium.com/analytics-vidhya/being-iterable-creating-generators-in-python-20de0dc649a1?source=collection_archive---------18----------------------->

![](img/7a0519da657cffac2b0d12f9efe42fb2.png)

奥斯卡·伊尔迪兹在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

如果我们可以从一个对象中得到一个迭代器，那么这个对象就叫做 **iterable** 。

Python 中的**迭代器**只是一个可以被迭代的对象，也就是说，简单地说，它意味着——“*对象将返回数据，一次一个元素”*。

Python 迭代器对象必须实现两个方法， **__iter__()** 和 **__next__()** ，其中 __iter__()返回迭代器，__next__()返回下一个元素。

让我们从 Python 的内置数据结构开始，比如列表、字符串、字典——这些都是可迭代的。

我们将逐一举例来理解这些。

# 列表迭代器

```
*# Creating a list*
**alist = [23, 13, 67, 79]***# iter() [ it internally calls __iter__() to return an iterator]*
**iterator = iter(alist)***# Check type of iterator*
**print(type(iterator))** Output: <class 'list_iterator'>
```

当所有元素都被提取出来后，就会产生 StopIteration 异常。

```
*# Lets iterate through it and extract elements using next()***while True:
    try:
        print(next(iterator))
    except StopIteration:
        break****Output:**
23
13
67
7
```

# 字典迭代器

```
*# create a dictionary***adict = {1:'One', 2:'Two', 3:'Three'}****iterator = iter(adict)****print(type(iterator))
Output:**<class 'dict_keyiterator'>
```

使用 dictionary_keyiterator，我们可以遍历字典中的键

```
**while True:
    try:
        k = next(iterator)
        print(f"{k} = {adict[k]}")
    except StopIteration:
        break**Output:
1 = One
2 = Two
3 = Three
```

# 用 Python 构建我们自己的迭代器

我们只需要实现两个方法`__iter__()`和`__next__()`。

`__iter__()`方法返回迭代器对象本身。如果需要，可以执行一些初始化。

`__next__()`方法必须返回序列中的下一项。在到达终点时，以及在随后的调用中，它必须引发`StopIteration`。

## 让我们构建几个迭代器

```
**class M_Raise_N:
    '''
    returns no. m to the power k where k=0 to n
    '''
    def __init__(self, m, n):
        self.m = m
        self.n = n

    def __iter__(self):
        self.counter = 1
        return self

    def __next__(self):
        if self.counter > self.n:
            raise StopIteration
        else:
            result = self.m ** self.counter
            self.counter += 1
            return result****for element in M_Raise_N(3,5):
    print(element)****Output:**
3
9
27
81
243
```

## 让我们创建另一个迭代器来生成斐波那契数列

```
**class fibo:
    '''generate fibonacci numbers
    '''
    def __init__(self, n):
        self.a=0
        self.b=1
        self.n=n

    def __iter__(self):
        self.counter = 1
        return self

    def __next__(self):
        if self.counter > self.n:
            raise StopIteration
        else:
            x = self.a
            self.a = self.b
            self.b = x + self.b
            self.counter += 1
            return self.b****for element in fibo(7):
    print(element)****Output:**
1
2
3
5
8
13
21
```

# 发电机

正如我们在上面看到的，创建迭代器涉及到创建一个类的开销，实现像 __iter__()、__next__()、引发 StopIteration 异常这样的方法，生成器提供了一个创建迭代器的选项，而没有这些开销。

简单地说，生成器是一个返回对象(迭代器)的函数，我们可以迭代这个对象(一次一个值)。

# 如何用 Python 创建生成器？

这就像用 yield 语句创建函数一样简单。如果一个函数至少包含一个 yield 语句，它就成为一个生成函数。

让我们试一些例子

```
**def m_raise_n(m, n):
    '''to generate numbers m to power k where k=0 to n
    '''
    for i in range(1,n+1):
        yield m ** i****g = m_raise_n(2, 7)****type(g)**
generator# Lets iterate and generate numbers **
while True:
    try:
        print(next(g))
    except StopIteration:
        break****Output:** 4 8 16 32 64 128
```

另一种方法是使用 for 循环从生成器函数中生成数字

```
**for item in m_raise_n(2,7):
    print(item)**Output:
2
4
8
16
32
64
128
```

## 让我们创建另一个生成器函数

```
**# Generator function**
**def nextN(x, n):
    '''to generate next n successors of item x
    '''
    for  i in range(1, n+1):
        yield x + i****for element in nextN(13, 7):
    print(element)****Output:**14
15
16
17
18
19
20
```