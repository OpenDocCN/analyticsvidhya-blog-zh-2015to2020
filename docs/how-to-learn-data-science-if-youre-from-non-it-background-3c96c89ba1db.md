# 如果你来自非 IT 背景，如何学习数据科学

> 原文：<https://medium.com/analytics-vidhya/how-to-learn-data-science-if-youre-from-non-it-background-3c96c89ba1db?source=collection_archive---------16----------------------->

“**数据科学**这个术语是在 21 世纪初创造的。这归功于 William S. Cleveland，他在 2001 年写了“**数据科学**:扩展统计领域技术领域的行动计划。”

数据科学家需要多学科技能的结合。

让我来分类一下数据分析和数据可视化之间的区别:

***数据分析库*** 如 NumPy、Pandas 等。

***数据可视化库*** 如 Seaborn、Matplotlib 等。

![](img/19ef8112b8cd7de0fafab839a935eacc.png)

# 如何安装 Python？

[Anaconda 发行版](https://www.anaconda.com/distribution/)用于安装(适用于 Linux、Windows 和 Mac OS X)。这是因为它包含了数据分析所需的所有库。

一旦安装了 Anaconda，Jupyter 笔记本是最好用的

有许多方法可以打开 Jupyter 笔记本，其中一些方法如下:

1.  打开 Anaconda 提示符，输入 Jupyter Notebook
2.  打开 Anaconda Navigator 并启动 Jupyter

Jupyter 笔记本用扩展名保存文件。ipynb

# **i.pynb-互动 Python 笔记本**

要用 Python 打印一个字符串，只需写:

**“打印”**命令打印行**“你好，世界！”** —学习一门新的编程语言时，通常要编写的第一个不可避免的程序:

```
print(“Hello, World!”)
```

## 示例编号

【类型】1，2，-5，1000 整数

1.2，-0.5，2e2，3E2 浮点数

现在让我们从一些基本的算术开始。

# 基础算术

## #注释:-添加一个有意义的完整行

## -、+等是运算符

## / :-除法，如果需要，用小数给出答案

## // :- Float 用整数和小数给出答案

## * :-乘法

## **:电源

# Python 缩进

缩进是指代码行开头的空格。

在其他编程语言中，代码中的缩进只是为了可读性，而 Python 中的缩进非常重要。

Python 使用缩进来表示代码块。

# Python 语法——您成功的基石

好吧，老实说，有许多关于语法的具体细节需要学习，这篇文章不足以充分完成这项任务。但我会真心实意地试着减轻你学习上的痛苦。

以下是基本概念:

*   Python 是一种强类型语言(在强类型语言中，每种数据类型都被预定义为语言的一部分，并且为给定程序定义的所有常量或变量都必须用其中一种数据类型来描述)，但同时，它是动态类型的(没有变量声明，只有赋值语句)。
*   Python 是区分大小写的语言(var 和 VAR 是两个不同的变量)和面向对象的语言(Python 中的一切都是对象:数字、字典、用户定义的和内置的类)。
*   Python 没有强制的运算符完成字符，块边界由缩进定义。缩进开始一个新的块，缺少缩进结束它。等待新缩进的表达式以冒号(:)结束。单行注释以井号(#)开始；对于多行注释，使用字符串文字，用三撇号或三引号括起来。
*   使用等号(" = ")赋值(实际上，对象与值的名称相关联)，使用两个等号(" == ")检查相等性。您可以分别使用+ =和— =运算符，根据运算符右侧指定的值来增加/减少这些值。这适用于许多数据类型，包括和字符串。
*   **数据类型。**在 Python 中，有以下几种数据结构:列表、元组、字典。集合也可用，但仅在 Python 2.5 和更高版本中可用。列表就像一维数组(但你也可以创建其他列表的列表，得到一个多维数组)，字典是关联数组(所谓的哈希表，可以是任何数据类型)，元组是不可改变的一维数组(在 Python 中，“数组”可以是任何类型，所以你可以混合，例如，整数，字符串等。在列表/字典/元组中)。所有类型数组中第一个元素的索引为 0，最后一个元素可以通过 index -1 得到。
*   使用冒号(:)只能处理部分数组元素。在这种情况下，冒号前的索引指示数组的已用部分的第一个元素，冒号后的索引指示数组的已用部分的最后一个元素之后的元素(它不包括在子数组中)。如果未指定第一个索引，则使用数组的第一个元素；如果未指定第二个元素，则最后一个元素将是数组的最后一个元素。计算负值确定元素从末端开始的位置。

打印(“印度是一个国家”)

# 印度是一个国家的产出

1.  **，**用于字符串的情况

**2** 。int，float，string 不允许容纳 1 行以上的内容

**3** 。该列表包含不止一行

这些是可以在 Jupyter 笔记本上尝试的代码行

```
# Modulo
7%4# Floor Division
7//4# Multiplication
2*2# Powers
2**3# Can also do roots this way
4**0.5# Order of Operations followed in Python
2 + 10 * 10 + 3# Can use parentheses to specify orders
(2+10) * (10+3)
```

# 变量赋值

创建这些标签时使用的名称和不能使用的名称需要遵循一些规则:

```
1\. Names can not start with a number.
2\. There can be no spaces in the name, use _ instead.
3\. Can't use any of these symbols :'",<>/?|\()!@#$%^&*~-+
4\. It's considered best practice (PEP8) that names are lowercase.
5\. Avoid using the characters 'l' (lowercase letter el), 'O' (uppercase letter oh), 
   or 'I' (uppercase letter eye) as single character variable names.
6\. Avoid using words that have special meaning in Python like "list" and "str"
```

Python 使用*动态类型*，这意味着您可以将变量重新分配给不同的数据类型。这使得 Python 在分配数据类型时非常灵活；它不同于其他静态类型化的语言。

# 动态类型的利弊

## 动态类型的优点

*   非常容易使用
*   更快的开发时间

## 动态类型的缺点

*   可能会导致意外的错误！
*   你需要注意`type()`