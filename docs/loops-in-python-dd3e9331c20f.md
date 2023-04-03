# Python 中的循环

> 原文：<https://medium.com/analytics-vidhya/loops-in-python-dd3e9331c20f?source=collection_archive---------18----------------------->

循环在编程中用于重复特定的代码块。在 Python 中我们有两种类型的循环:for 和 while。

# while 循环

Python 中的 while 循环用于迭代一段代码，只要测试表达式(条件)为真。

语法:

```
while test_expression:
   (body of while loop)
```

在 while 循环中，首先检查测试表达式。只有当 test_expression 的计算结果为 True 时，才会执行循环体。在第一次迭代之后，再次检查 test_expression，并且该过程继续，直到 test_expression 评估为 False。

while 循环的主体是通过缩进来确定的。正文以缩进开始，第一个非缩进的行表示结束。

示例:

```
i = 1
while i < 6:
   print(i)
   i += 1
```

输出将是:

```
1
2
3
4
5
```

在上面的程序中，只要变量 I 小于 6，测试表达式就为真。

我们需要增加循环体中计数器变量的值。否则将导致永无止境的循环。

## while 用 else 循环

while 循环也可以有一个可选的 else 块。如果 while 循环中的 test_expression 计算结果为 False，则执行 else 部分。

举个例子，

```
i = 1
while i < 6:
   print(i)
   i += 1
else:
   print(“i is no longer less than 6”)
```

输出将是:

```
1
2
3
4
5
```

我不再小于 6

这里，在第六次迭代时，test_expression 变为 False。因此，执行 else 部分。

# for 循环

for 循环用于迭代序列或其他迭代对象。

语法:

```
for <variable_name> in <sequence>:
   (body of for loop)
```

循环继续，直到到达序列中的最后一项。与 while 循环一样，for 循环的主体也是通过缩进来确定的。

查看以下示例:

```
for x in “programming”:
   print(x)
```

该程序给出了输出:

```
p
r
o
g
r
a
m
m
i
n
g
```

## range()函数

要在一组代码中循环指定的次数，我们可以使用 range()函数。

range()函数返回一个数字序列，默认情况下从 0 开始，按 1 递增(默认情况下)，到指定的数字结束。

示例:

```
for x in range(7):
   print(x)
```

这给出了输出:

```
0
1
2
3
4
5
6
```

注意:范围(7)不是从 0 到 7 的值，而是从 0 到 6 的值。

range()函数默认将 0 作为起始值，但是可以通过添加一个参数来指定起始值:range(3，7)，表示从 3 到 7(但不包括 7)的值:

示例:

```
for x in range(3, 7):
   print(x)
```

输出将是:

```
3
4
5
6
```

range()函数默认将序列递增 1，但是可以通过添加第三个参数来指定增量值:range(1，10，2):

```
for x in range(1, 10, 2):
   print(x)
```

这给出了输出:

```
1
3
5
7
9
```

## 否则在 for 循环中

for 循环中的 else 关键字指定循环结束时要执行的代码块:

```
for x in range(5):
   print(x)
else:
   print(“Finished Counting!”)
```

输出:

```
0
1
2
3
4
```

数完了！

该程序打印从 0 到 4 的所有数字，然后在循环结束时执行 else 部分，打印消息“完成计数”。