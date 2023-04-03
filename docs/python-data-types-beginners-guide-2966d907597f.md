# Python 数据类型:初学者指南

> 原文：<https://medium.com/analytics-vidhya/python-data-types-beginners-guide-2966d907597f?source=collection_archive---------8----------------------->

![](img/fc8c0f89303c8c9114d425164587d3c2.png)

## Python 编程简介

Python 是一种高级通用编程语言。这一切都始于 1989 年，当时 Guido Van Rossum 开始实现它，并由 python 软件基金会进一步开发。使用 python，代码可读性得到保证，程序员可以轻松地表达概念。不像一些编程语言，学习曲线是我称之为的平滑曲线。

> 蟒蛇是以电视喜剧节目“巨蟒剧团的飞行马戏团”命名的，而不是蟒蛇。

如果您想在 python 之旅中迈出大胆的下一步，那么这篇文章是您的最佳选择。不久前，我写了一篇关于如何[建立你的 python 开发环境](/analytics-vidhya/setting-up-your-python-3-development-environment-on-windows-26d912da9d2f?source=friends_link&sk=e53c5da824c0b618e0881a6e9cbd219a)的文章，你应该去看看。

这是本系列的第一篇，旨在向您介绍 python 编程，从而让您更容易成为 Python 程序员。

在我们深入研究数据类型之前，我想先简单介绍一下变量，因为在阅读过程中你会看到它们的使用。此外，它们将帮助你更好地理解我们正在讨论的一些概念。

> **变量**是存储器中用于存储数据的命名位置。你可以把变量看作计算机程序中保存数据的容器。使用 python，变量的声明和赋值非常简单，允许您随意命名变量并更改其值:

```
age = 15      #declaring and assignment
print(age)    #output: 15
name = ‘Jane’ #declaring and assignment
print(name)   #output: Jane
```

## **数据类型**

数据类型代表数据项的分类。它决定了可以对数据项执行什么类型的操作。每一种编程语言都有自己的哲学。我们的重点是`Numbers`、`Strings`、`Lists`、`Dictionaries`、`Boolean`、`Tuples`和`Set` 类型。

*   **数字:**这包括用数值表示的数据。Python 支持三种数值类型；`int`、`float`和`complex`。

> **整数**–包括正负整数。**浮动**点数有小数点。
> Python**complex**type 有两个参数——一个实数值和一个虚数值。

```
print(type(4))                            #output: <class 'int'>print(type(3.0))                          #output: <class 'float'>a = (3 + 2j) 
print(type(a))                            #output: <class 'complex'>
```

*   字符串:它们是引号中的字符序列。使用 python 有几种定义字符串的方法:

```
greetings = ‘Hello’                          #single quotesprint(greetings)                             #output: Hellogreetings = “Holla”                          #double quotesprint(greetings)                             #output: Hollagreetings = ‘’’namaskaar’’’                  #triple quotesprint(greetings)                             #output: Namaskaar
```

*   **布尔:**布尔有两个可能的值`True` 或`False` ，表示逻辑的真值。

> T 和 F 都必须大写，以便 python 解释器正确识别。

```
print(type(True))                           #output: <class 'bool'>
```

*   **列表:**我喜欢将列表定义为一系列用逗号分隔并放在方括号`[]`中的项目。它可以接受任何类型(int、float、string 等)和数量的项。

```
favorite_colors = [‘Blue’, ‘Purple’, ‘Black’]  #list of stringsfavorite_number = [1, 3, 9.5]                  #list of numbersfavorite_food = []                             #empty list
```

Python 还支持用内置的`list()`函数定义列表。使用 Python，您可以选择使用正数和负数索引来访问列表中的项目:

```
country = [‘Nigeria’, ‘India’, ‘Germany’, ‘France’]print(country[0])                            #Accessing first item
#output: Nigeriaprint(country[-1])                           #Accessing last item
#output: France
```

> Python 索引从 0 开始

*   **字典:**这些类型被定义为`key: value`对，条目不按顺序收集:

```
my_dict = {}                                  #empty dictionarymy_dict = {1: ‘English’, 2: ‘French’, 3: ‘Hausa’}
```

Python 还支持用内置的`dict()`函数定义字典。使用 python，您可以选择使用键访问字典中的项目:

```
friend = {‘name’: ‘Jones’, ‘age’: 19, ‘school’: ‘Stanford’}print(friend[‘school’])                  #output: Stanford
```

*   **元组:**元组除了不是可变的之外，与列表相似；一旦定义，内容就不能更改。与 list 不同，它们包含在括号`()`中:

```
school = (‘Harvard’, ‘Stanford’, ‘Bradford’)print(school)           #output: ('Harvard', 'Stanford', 'Bradford')
```

Python 还支持用内置的`tuple()`函数定义元组。使用 python，您可以选择通过索引从元组中访问项目，就像`list`:

```
country = (‘Nigeria’, ‘India’, ‘Germany’, ‘France’)print(country[-1])                 #output: Franceprint(country[2])                  #output: Germany
```

*   **集合:**用花括号`{}`括起来的无序的独特项目的集合。像 list 一样，它们也是可变的。Python 允许我们进行`intersection`、`union`等数学集合运算。Python 还支持用内置的`set()`函数定义集合。

```
fav_number = {1, 3, 6, 9.5}                 #set of numbersprint(fav_number)                           #output: {1, 3, 6, 9.5}A = set(‘a’)B = set(‘b’)A.union(B)                                  #output: {'a', 'b'}
```

好哇！！！你已经掌握了 Pythons 的基本语法。现在不要松口，因为还有更多事情要做:)

请随意从这个资源库下载用于演示的整个 python 笔记本[Python 编程简介](https://github.com/ezekielolugbami/Introduction-to-Python-Programming/blob/master/python%20data%20types.ipynb)。