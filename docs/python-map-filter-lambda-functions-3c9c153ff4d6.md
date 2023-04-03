# Python 地图/过滤器/lambda …函数

> 原文：<https://medium.com/analytics-vidhya/python-map-filter-lambda-functions-3c9c153ff4d6?source=collection_archive---------16----------------------->

![](img/7af598990be1c1744f51e8d6c3134b65.png)

克里斯里德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

我是查兰。这是我的第一篇文章，在这里我列出了 python 中一些有用的函数。Python 提供了几个函数来实现函数式编程。函数式编程是关于表达式的。面向表达式的函数有:

1.  地图()
2.  过滤器()
3.  减少()
4.  λ函数
5.  列表理解

## 1 .地图()

map()函数遍历迭代器的每一个元素，并执行我们需要执行的功能。

那么什么是迭代器呢🤔🤔？

迭代器是包含可数个值的对象，可以被迭代😁。

列表、元组、字典和集合都是可迭代的对象

***语法:***

映射(函数，迭代)

示例:假设我们需要对列表中的每个元素求平方

通过使用 for 循环

```
l=[]def squareFor(x): for i in x: l.append(i**2) return lres=squareFor([1,2,3,4,5])print(res)#res=[1,4,9,6,25]
```

通过使用 map()

```
def square(x):
   return x**2li=[1,2,3,4,5]
squaredList=list(map(square,li))
print(squaredList)//[1,4,9,16,25]
```

这里我们在 map()函数中提供了 function-square 和 iterator-list

每个元素都被传递给 square()函数，该函数将每个元素的平方返回给 squaredList。

***2.filter()***

filter()在函数的帮助下过滤给定的序列，该函数测试序列中的每个元素是否为真

**语法:**

过滤器(函数，可迭代)

示例:过滤列表中的偶数

使用 for 循环

```
l=[]def even(x): for i in x: if(i%2==0): l.append(i) return lfilter=even([1,2,3,4,5])print(filter)//[2,4]
```

使用过滤器()

```
def evenno(x): return x%2==0filteredList=list(filter(evenno,[1,2,3,4,5]))print(filteredList)#[2,4]
```

列表中的每个元素都被传递给一个函数。基于条件元素被返回到 filteredList。

3.reduce()

reduce()用于将特定函数应用于返回单个值的所有元素列表。

为了使用 reduce，我们需要导入它

**从 functools 导入减少**

**语法:**

reduce(函数，可迭代)

```
from functools import reducedef reduceList(acc,x): return acc+xreducedList=reduce(reduceList,[1,2,3,4,5])print(reducedList)#15
```

还原操作

acc=1，x=2 1+2=3 现在 3 被分配给 acc

acc=3，x = 3 ^ 3+3 = 6 ^ 6 被分配给 acc

acc=6，x=4 6+4=10 10 被分配给 acc

acc=10，x=5 10+5=15 它到达列表的末尾，返回 15

**4λ函数**

lambda 函数是一个匿名函数。

那么什么是匿名函数呢🤔？

匿名函数是没有函数名的函数。

**语法**

lambda 参数:表达式

lambda 函数接受更多参数，但只能有表达式

```
y=lambda z:z**2print(y(2))#4
```

y(2)这里 2 是传递给 lambda 函数的参数，表达式被求值并返回值。

lambda 函数可以在 map、filter 和 reduce 中使用。

```
#lambda function in mapmapp=list(map(lambda x:x**2,[1,2,3,4,5]))print(mapp)#[1,4,9,16,25]#instead of normal function we pass in lamba function#filterfiltered=list(filter(lambda x:x%2==0,[1,2,3,4,5]))print(filtered)#reducefrom functools import reducereduced=reduce((lambda acc,x:acc+x),[1,2,3,4,5])print(reduced)
```

**5。列表理解**

这是一种基于现有列表定义和创建列表的优雅方式。

**语法**

[列表条件中项目的表达式]

条件可选！

```
#wITHOUT CONDITIONALl=[i for i in range(5)]print(l) #[0,1,2,3,4]#WITH CONDITIONALevenList=[i for i in [1,2,3,4,5] if(i%2==0)]print(evenList)#for loop=>condition=>placing i in list if condition is true
```

我们还可以通过替换列表来返回 tuple，set in map，filter，reduce，lambda 函数😯.即元组(map(lambda x:x**2，[1，2，3，4，5]))。这将返回一个元组。

和列表理解一样，python 也有字典、集合、元组理解。

我相信这些信息会帮助你理解 python 中函数式编程的基础。谢谢你😊。

代码-[https://github.com/charanpy/lambda-map-filter-reduce](https://github.com/charanpy/lambda-map-filter-reduce)