# Python Bootcamp:运算符 I

> 原文：<https://medium.com/analytics-vidhya/python-bootcamp-operators-i-d3f9d795fac?source=collection_archive---------24----------------------->

# 变量

变量是内存中用来存储数据的指定位置。简单地说，它就像一个有名字的盒子，你可以在里面存储一个值。

您可以简单地通过编写以下代码来定义变量:

```
<variable_name> = <value>
```

这里有一个例子:

```
a = 9
```

这里，a 是变量。我们已经把右边的值，即 9 赋给了左边的变量 a。

# 如何命名一个变量

你可以为你的变量取任何名字，但是有一些规则。

*   变量名不应以数字开头。
*   当变量名包含两个以上的单词时，应该用下划线隔开(例如:用户名)。

一些有效变量名和赋值的例子:

```
username = “@sonnymommom”user_id = 100verified = False
```

注意:Python 是一种区分大小写的编程语言。因此，在 Python 中，A 和 A 是两个不同的变量名。

一定要记住给变量起一个容易理解的名字，并遵循命名惯例！

# 更新变量的值

为了改变它们的值，变量可以被重新分配任意多次。

```
a = 9
print(a)
a = 100 // Overwriting the value by assigning 100 to the variable x.
print(a)
```

这会产生以下结果:

```
9
100
```

# 从用户处获取输入

在 Python 中，可以使用 input()函数从用户那里获取输入。例如:

```
x = input(‘Enter a sentence:’)
print(‘The inputted string is:’, x)
```

当您运行该程序时，输出将是:

```
Enter a sentence: Learning Python is fun.
The inputted string is: Learning Python is fun.
```

# 经营者

这部分先说算术运算符。

Python 支持不同类型的算术运算，这些运算可以在文字数字、变量或某种组合上执行。主要算术运算符有:

*   添加

```
x + y
```

*   减法

```
x — y
```

增加

```
x * y
```

*   分开

```
x / y
```

*   系数

```
x % y
```

*   指数运算

```
x ** y
```

示例:

```
x=7, y=3print(x*y)print(x%y)
```

这将产生以下结果:

```
21
1
```