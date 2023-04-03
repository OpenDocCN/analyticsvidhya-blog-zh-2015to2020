# Python 的编程妙招。

> 原文：<https://medium.com/analytics-vidhya/pythons-cool-tricks-for-programming-e0940c555747?source=collection_archive---------9----------------------->

简单是 Python 解决问题的方式。

![](img/83ddc62d9c5487a9ac4aa5152e04ecc3.png)

[在 memegenerator 上创建](https://imgflip.com/i/4lesva)

1.  **反转字符串、列表或元组→**

每当我发现自己处于需要执行反向操作的情况时，我通常会使用一个 **for 循环**或者求助于一些类似于 **reverse()** 的函数来完成这项工作。

但是后来，令我震惊的是，我发现 python 有另一个锦囊妙计来解决这个问题——切片操作你可以使用切片语法来反转字符串、元组或列表

```
string = 'John Wick'
print(string[::-1]) # logs: 'kciW nhoJ'tup = (23, 45, 12, 56)
print(tup[::-1]) # logs: (56, 12, 45, 23)l = ['john', 'baba', 'yaga', 'wick']
print(l[::-1]) # logs: ['wick', 'yaga', 'baba', 'john']
```

2.**检查给定的字符串是否是回文→**

*很简单，对吧？当我们想到解决这个问题的方法时，我们想到了很多不同的方法。但是猜猜看，解决这个问题的最简单的方法之一可能还是**切片操作**。*

使用上面讨论的反转技巧，一行代码就足以检查给定的字符串是否是回文。首先，我们使用`string[::-1]`来反转字符串，然后我们简单地比较我们的原始字符串和反转的字符串`string == string[::-1]` 来得到结果。

```
string = 'aibohphobia'
print('Palindrome' if string == string[::-1] else 'No Plaindrome')
# print: Palindromestring = 'johnwick'
print('Palindrome' if string == string[::-1] else 'No Plaindrome')
# print: No Plaindrome
```

3.**从列表中获取“n”个最大或最小的元素→**

*今天*，我们将使用 python 的标准库 **heapify** 来将我们的列表转换成堆，然后使用`nlargest(n, iterable)`和`nsmallest(n, iterable)`方法来从我们的堆中找到“ **n** 个最大和最小的元素。

```
lis = [23, 9, 34, 22, 43, 21, 5, 40]import heapq
heapq.heapify(lis) # Convert list into min heapprint(heapq.nlargest(4, lis)) # will print 4 largest elements
# print: [43, 40, 34, 23]print(heapq.nsmallest(4, lis)) # will print 4 smallest element
# print: [5, 9, 21, 22]
```

4.**交换 2 个元素，不使用第三个变量→**

这个问题会在工作面试中频繁出现，当我使用这个技巧时，面试官*(如果不是 python 背景)*会变得不耐烦。但是作为一个 python 爱好者/开发者，当你知道这些技巧时，你总是会发现自己处于更有利的位置。

只需使用`a,b = b,a`交换两个变量的值。如果你有兴趣知道这内部是如何工作的，参考这个 [**解释**](https://stackoverflow.com/questions/21047524/how-does-swapping-of-members-in-tuples-a-b-b-a-work-internally) 。

```
a = 10
b = 25
print(f'a: {a}, b: {b}') # print: a: 10, b: 25a,b = b,a # swapping variable 
print(f'a: {a}, b: {b}') # print: a: 25, b: 10
```

*在*的采访中，你总是可以用老办法来回答→

```
a,b = 10, 25
a = a ^ b
b = a ^ b 
a = a ^ b
print(f'a: {a}, b: {b}') # print: a: 25, b: 10
```

5.**统计字符串、元组和列表中每个元素的出现次数→**

Python 有一个惊人的**集合**模块，其中包含许多有用的实用程序。*对于我们的任务，我们将使用集合模块中的* ***计数器*** *。*

首先，让我们使用 split()函数分割字符串，默认情况下，该函数返回一个按空格分割的字符串列表→

```
quoteFromGeeta = '''senses exists beyond body
           mind exists beyond senses
           intelligence exists beyond mind
           and knowing yourself exists beyond intelligence'''word_list = quoteFromGeeta.split()
print(word_list)#print: ['senses', 'exists', 'beyond', 'body', 'mind', 'exists', 'beyond', 'senses', 'intelligence', 'exists', 'beyond', 'mind', 'and', 'knowing', 'yourself', 'exists', 'beyond', 'intelligence']
```

现在使用 ***计数器，*** 让我们找出**单词列表中每个单词的计数→**

```
from collections import Counterc = Counter(word_list) # pass your list to Counter[print(f'Word: {word}, Count: {count}') for word,count in c.items()]# prints
Word: senses, Count: 2
Word: exists, Count: 4
Word: beyond, Count: 4
Word: body, Count: 1
Word: mind, Count: 2
Word: intelligence, Count: 2
Word: and, Count: 1
Word: knowing, Count: 1
Word: yourself, Count: 1
```

很简单，不是吗？

你也可以使用`most_common(n)`方法找到最常见的单词

```
common_words = c.most_common(3)
[print(word, count) for word,count in common_words]# print
Word: exists, Count: 4
Word: beyond, Count: 4
Word: senses, Count: 2
```

6.**检查两个字符串是否是变位词→**

> 你一定想知道，什么是变位词，对吗？
> 一个**串**的一个**变位词**是一个**串**包含相同的字符，只是字符的顺序可以不同。例如，“abcd”和“dabc”是彼此的一个**变位词**。

使用 ***计数器*** *从* **集合*，*** 我们可以很容易地检查两个字符串是否是变位词→

```
from collections import Counter
str1 = 'abcd'
str2 = 'dabc'c1 = Counter(name1)
c2 = Counter(name2)
print(c1 == c2) # print True
```

7.**使用运算符链接检查两个以上的条件→**

检查多个条件的值是我们每天都要执行的任务，让我们考虑一个需要检查`**5 < a < 20**`的例子

*在其他编程语言*中，你会这样检查它→

```
if **a > 5 and a < 20**:
    pass
```

*在 python 中，*你可以简单地使用**操作符链接→**

```
if **5 < a < 20**:
    pass
```

8.**从列表中取值并分配给变量→**

为了解决这个问题，我们将使用一个对许多开发人员来说仍然难以理解的概念— **解包**

虽然我们大多数人可能在某个时候用过它，但是这个概念本身会引起你的兴趣。如果你是这个概念的新手，那么你今天肯定会学到一些很棒的东西。

> **Python 中的解包**指的是在一个赋值语句中将一个可迭代的值赋给一个变量元组(或列表)的操作。你会喜欢 Python 中的**解包**特性。

举例说明的时间到了→

```
l = ['john', '40', 'assassin']
#with out unpacking feature
name = l[0]
age = age[1]
profession = l[2]#with unpacking feature
name, age, profession = l 
```

要深入探索它的工作原理，请参考这篇来自 Stackabuse 的 [**文章**](https://stackabuse.com/unpacking-in-python-beyond-parallel-assignment/#:~:text=Unpacking%20in%20Python%20refers%20to,the%20iterable%20unpacking%20operator%2C%20*%20.) 以获得惊人的解释。

9.**获取列表、字符串和元组的最后一个元素→**

让我们来看看一个解决方案，它将使解决这个问题看起来像是小菜一碟。*使用* `*iterable[-1]*` *查找最后一个元素。*

```
l = ['john', '40', 'assassin']
print(l[-1]) # 'assassin'
```

10.**检查列表或元组是否为空**

*如果你一直觉得检查一个列表是否为空很麻烦。*如果你一直在使用如下所示的这种繁琐而又混乱的方法，那么我们将会见证一些让我们的生活变得更加轻松的事情。首先，让我们看看更常用的方法→

```
temp = []
if len(temp) == 0:
   passif len(temp):
   pass
```

*在 python 中，空列表被认为是错误的，所以你可以像下面这样直接检查它→*

```
if temp:
   pass
```

11.**连接两个列表，元组→**

您可以使用`**+**`操作符来连接元组和列表，在执行连接后，它将返回新的元组或列表。

```
# On list
list1 = [1,2,3,4,5]
list2 = [6,7,8,9,10]
list_new = list1 + list2
print(list_new) # print [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]# On tuple
tuple1 = (1,2,3,4,5)
tuple2 = (6,7,8,9,10)
tuple3 = (11,12,13,14,15)
tuple_new = tuple1 + tuple2 + tuple3
print(tuple_new)
```

12.**从现有列表创建新列表→**

让我们用一个例子来理解我们的问题陈述。

```
l = ['John Wick', 45, 'assassin']new_l =  l # this will copy the reference of l in new_lprint(new_l) # print  ['John Wick', 45, 'assassin']l[0] = 'Baba Yaga' # any change in l will reflect to new_l alsoprint(new_l) # print ['Baba Yaga', 45, 'assassin']
```

*从上面的快照*可以看出，新的列表变量只存储了现有列表的引用地址，现在指向它，但是根据我们的要求，没有创建新的列表。

要创建一个全新的列表*，您只需在现有列表上使用*`***list[:]***`**，通过一个简单的操作您将拥有一个新的列表。**

```
*l = ['John Wick', 45, 'assassin']new_l =  l[:] # this will create the new listprint(new_l) # print  ['John Wick', 45, 'assassin']l[0] = 'Baba Yaga' # any change in l will reflect to new_l alsoprint(new_l) # print ['John Wick', 45, 'assassin']*
```

> *注意:list[:]将创建一个 list 的浅层副本，要创建深层副本，可以使用`copy`模块中的`deepcopy`方法*

# *未完待续…*

*python 中有更多很酷的库，如 **functools** 、 **itertools** 、 **pandas** ，我们将在后续文章中探讨。*

*希望你喜欢这篇文章。请在评论中告诉我，你最喜欢哪个技巧。*

*祝你好运，编码愉快😃*