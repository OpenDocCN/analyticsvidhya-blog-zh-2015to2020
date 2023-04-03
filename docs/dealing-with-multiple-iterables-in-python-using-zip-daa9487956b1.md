# 使用 zip()在 Python 中处理多个可迭代对象

> 原文：<https://medium.com/analytics-vidhya/dealing-with-multiple-iterables-in-python-using-zip-daa9487956b1?source=collection_archive---------4----------------------->

Python 的 zip()函数创建了一个迭代器，它将聚合两个或更多可迭代对象的元素。zip()映射多个容器的相似索引，这样它们就可以作为一个实体使用。

**zip()是可用的内置命名空间。**
`'zip' in dir(__builtins__)`
真实

## zip()接受 iterables 并返回元组的迭代器，其中第 I 个元组包含来自每个参数序列或 iterables 的第 I 个元素。

```
## First example of zip()
atomic_num = [1, 2, 3, 6, 7, 8]
chem_element = ['H', 'He', 'Li', 'C', 'N', 'O']

# both iterables passed to zip() are lists
zipped = zip(chem_element, atomic_num)

# zipped holds an iterator object
print(type(zipped));
<class 'zip'>

# extracting elements 
for tup in zipped:
    print(tup)
('H', 1)
('He', 2)
('Li', 3)
('C', 6)
('N', 7)
('O', 8)
```

这里，zip(chem_element，atomic_num)返回一个迭代器，它产生(x，y)形式的元组。x 值取自化学元素，y 值取自原子数量。

如果我们处理像列表、元组或字符串这样的序列，那么 iterables 肯定是从左到右求值的。然而，对于其他类型的可重复项，我们可能会看到奇怪的行为

```
## String and list as iterables passed to zip()

s = 'rgb'
colors = ['Red', 'Green', 'Blue']

for t in zip(s, colors):
    print(t)
------
Output:
('r', 'Red')
('g', 'Green')
('b', 'Blue')## using zip() with sets
s1 = {12, 13,11}
s2 = {'l', 'm', 'k'}

for t in zip(s1, s2):
    print(t)
--------
Output:
(11, 'k')
(12, 'm')
(13, 'l')
```

正如我们注意到的集合，数据不是从左到右提取的

## 处理长度不等的可重复项

```
## using zip() with two iterables of unequal length

A = [1, 2, 3, 4, 5]
B = [20, 25, 27]

for t in zip(A, B):
    print(t)
------
Output:
(1, 20)
(2, 25)
(3, 27)
```

使用 zip()时，注意 iterables 的长度很重要。我们作为参数传入的 iterables 可能长度不同。在这些情况下，zip()输出的元素数量将等于最短 iterable 的长度。zip()将完全忽略任何更长的可重复项中的剩余元素

## 使用 itertools.zip_longest()

```
from itertools import zip_longest
numbers = [1,2,3]
letters = ['a', 'b', 'c', 'd']
floats = [10.1, 11.2, 12.3, 13.4, 14.5, 15.6]print("Using zip()\n--------------")
for t in zip(numbers, letters, floats):
    print(t)print("\nUsing zip_longest()\n-----------------")
for t in zip_longest(numbers, letters, floats):
    print(t)

Using zip()
--------------
(1, 'a', 10.1)
(2, 'b', 11.2)
(3, 'c', 12.3)Using zip_longest()
-----------------
(1, 'a', 10.1)
(2, 'b', 11.2)
(3, 'c', 12.3)
(None, 'd', 13.4)
(None, None, 14.5)
(None, None, 15.6)
```

正如我们看到的，使用 zip_longest()迭代将继续，直到最长的可迭代次数用完:

## Python 3 和 2 中 zip()的比较

Python 的 zip()函数在该语言的两个版本中工作方式不同。
在 Python 2 中，zip()返回元组列表
在 Python 3 中，然而，zip()返回迭代器。

```
## Python 2.7X = [1,2,3]
Y = ['a','b','c']zipped = zip(X,Y);
print(type(zipped));
print(zipped);for t in zip(X,Y):
    print(t)
------
output:
<type 'list'>
[(1, 'a'), (2, 'b'), (3, 'c')]
(1, 'a')
(2, 'b')
(3, 'c')## Python 3.6
X = [1,2,3]
Y = ['a','b','c']zipped = zip(X,Y);
print(type(zipped));
print(zipped);for t in zip(X,Y):
    print(t) -------
Output:
<class 'zip'>
<zip object at 0x000002A5A9D1DC08>
(1, 'a')
(2, 'b')
(3, 'c')
```

## 在多个可迭代对象上循环

在多个可迭代对象上循环是 Python 的 zip()函数最常见的用例之一。如果您需要遍历多个列表、元组或任何其他序列，那么您很可能会求助于 zip()

```
##
import numpy as np
s = 'A'*5
n = np.arange(1,6)for letter,digit in zip(s,n):
    print("{}{}".format(letter, digit))---------
Output:
A1
A2
A3
A4
A5
```

## 并行遍历字典

在 Python 3.6 及更高版本中，字典是有序集合，这意味着它们按照引入时的顺序保存元素。如果您利用了这个特性，那么您可以使用 Python zip()函数以一种安全和一致的方式遍历多个字典

```
student = {'name': ['Arjun', 'Ram', 'John'] }stud_id = {'stud_id': [1,2,3]}for k1, k2 in zip(student, stud_id):
    for v1, v2 in zip(student[k1], stud_id[k2]):
        print(v1, v2)
--------------
Output:
Arjun 1
Ram 2
John 3
```

## 合并两个列表并并行排序

```
atomic_num = [1, 2, 3, 6, 7, 8]
chem_element = ['H', 'He', 'Li', 'C', 'N', 'O']# Sort data in alphabetic order of chemical element name
data1 = list(sorted(zip(chem_element, atomic_num), key = lambda t : t[0]))
print(data1)print("-"*60)
# Sort data in descending order of atomin number
data2 = list(sorted(zip(chem_element, atomic_num), key = lambda t : t[1], reverse=True))
print(data2)---------------
Output:
[('C', 6), ('H', 1), ('He', 2), ('Li', 3), ('N', 7), ('O', 8)]
------------------------------------------------------------
[('O', 8), ('N', 7), ('C', 6), ('Li', 3), ('He', 2), ('H', 1)]
```

## 并行处理列表并进行计算

```
sales = [50000, 56000, 62000, 70000]
cost  = [15000, 15000, 16000, 16000]
month = ['Jan', 'Feb', 'Mar', 'Apr']for m,s,c in zip(month, sales, cost):
    print(f"For month {m}, Sales : {s}, Cost : {c}, Profit : {s-c}")
-------------
Output:
For month Jan, Sales : 50000, Cost : 15000, Profit : 35000
For month Feb, Sales : 56000, Cost : 15000, Profit : 41000
For month Mar, Sales : 62000, Cost : 16000, Profit : 46000
For month Apr, Sales : 70000, Cost : 16000, Profit : 54000
```

## 使用 zip()构建词典

```
keys = ['rainbow', 'traffic', 'apple', 'banana']
values =['vibgyor', ['red', 'green', 'yellow'], ['red', 'green'], 'yellow']items_dict = dict(zip(keys, values))
print(items_dict)
-----------
Output:
{'rainbow': 'vibgyor', 'traffic': ['red', 'green', 'yellow'], 'apple': ['red', 'green'], 'banana': 'yellow'}
```