# Python 迭代器 Vs Iterables Vs 生成器

> 原文：<https://medium.com/analytics-vidhya/python-iterators-vs-iterables-vs-generators-e15b9403292e?source=collection_archive---------1----------------------->

这个三元组是 Python 编程中最令人困惑的事情之一，也是面试中被问得最多的问题之一。让我们一个一个地详细澄清这个困惑:—

**迭代器**

迭代器是包含可数个值的对象。
从技术上讲，在 Python 中，迭代器是实现迭代器协议的对象，迭代器协议由 __iter__()和 __next__()方法组成。

```
 **iter_obj=iter([3,4,5])**
This creates an iterator.

**next(iter_obj)**

This return the first element i.e 3

**next(iter_obj)**

This returns the next element i.e 4 and so on.
```

一个如何创建你自己的迭代器的例子

让我们创建一个一直数到 10 的迭代器。

```
class MyNumbers:
 def __iter__(self):
 self.a = 1
 return selfdef __next__(self):
 if self.a <= 10:
 x = self.a
 self.a += 1
 return x
 else:
 raise StopIterationmyclass = MyNumbers()
myiter = iter(myclass)for x in myiter:
 print(x)
```

**条目**

列表、元组、字典和集合都是可迭代的对象。它们是可迭代的容器，你可以从中获得迭代器。所有这些对象都有一个 iter()方法，用于获取迭代器。
示例:-

```
mytuple = (“red”, “blue”, “green”)
myit = iter(mytuple)print(next(myit))

This will return red.

print(next(myit))

This will return blue.

print(next(myit))

This will return green.
```

这里 mytuple 是可迭代的，myit 是迭代器。

现在让我们来见见最后一位嘉宾

**生成器**
一个 python 生成器函数借给我们一个 python 迭代的值序列。
让我们实现一个生成器:-

```
def even(x):
 while(x!=0):
 if x%2==0:
 yield x
 x-=1
 for i in even(8):
 print(i)

The output is:-
8
6
4
2
```

从上面的例子中，您一定已经看到了生成器和迭代器之间的一些显著差异。让我们来讨论一下

1.  我们使用一个函数来创建生成器。但是在用 python 创建迭代器时，我们使用了 iter()和 next()函数。
2.  python 中的生成器利用了‘yield’关键字。python 迭代器不会。
3.  每次‘yield’暂停 Python 中的循环时，Python generator 都会保存局部变量的状态。迭代器不使用局部变量，它只需要 iterable 来迭代。
4.  一个生成器可以有任意数量的“yield”语句。
5.  您可以使用 python 类实现自己的迭代器；在 python 中，生成器不需要类。
6.  要编写 python 生成器，可以使用 Python 函数或 comprehension。但是对于迭代器，必须使用 iter()和 next()函数。

**现在让我们讨论生成器和迭代器的优缺点:-**

1.  生成器对于编写快速而紧凑的代码非常有效。这是 Python 迭代器的一个优势。它们也比自定义迭代器更容易编码。
2.  Python 迭代器更节省内存。

```
 **def** func():i=1**while** i>0:yield ii-=1
**for** i **in** func():
    print(i)func().__sizeof__() 
```

这将为生成器返回 32。

但是对于迭代器，比如:

```
iter([1,2]).__sizeof__() 
```

我们得到 16。

感谢阅读。如果你从博客中学到了什么，请👏。