# 操作员 II &控制流程

> 原文：<https://medium.com/analytics-vidhya/operators-ii-control-flow-adb6185a3d4a?source=collection_archive---------32----------------------->

# 比较运算符

比较运算符用于比较两个值:

**操作员及姓名**

==相等

！=不相等

>大于

< Less than

> =大于或等于

<= Less than or equal to

The boolean values True and False are returned when an expression is compared or evaluated. For example:

```
x = 2
print(x == 2) # prints out True
print(x >= 3) # prints out True
print(x < 3) # prints out True
```

# Logical Operators

Logical operators are used to combine conditional statements:

**操作员及描述**

and →如果两个语句都为真，则返回真

or →如果其中一个语句为真，则返回真

not→反转结果，如果结果为真，则返回 False

查看以下示例:

```
x = 9
print(x > 3 and x < 10)
# returns True because 9 is greater than 3 AND 9 is less than 10 x = 9
print(x > 3 or x < 4)
# returns True because one of the conditions are true (9 is greater than 3, but 9 is not less than 4) x = 5
print(not(x > 3 and x < 10))
# returns False because not is used to reverse the result
```

# 控制流

在编程中，我们经常希望根据代码是否满足某个条件来运行不同的代码。例如，你可能想说‘太棒了！’只有分数 100%的时候。我们将从这里了解这一点。

## 如果语句

通过使用 if 语句，您可以编写仅在特定条件下执行的代码。您可以通过编写 if，后跟一个条件表达式和一个冒号(:)来创建 if 语句。

举个例子，

```
x = 100
y = 300
if y > x:
   print(“y is greater than x”)
```

在本例中，我们使用两个变量 x 和 y，它们作为 if 语句的一部分来测试 y 是否大于 x。由于 x 是 100，y 是 300，我们知道 300 大于 100，因此我们打印到屏幕上“y 大于 x”。

Python 依靠缩进(行首的空格)来定义代码中的范围。

If 语句，不带缩进(会引发错误):

```
x = 100
y = 300
if y > x:
print(“y is greater than x”) # you will get an error
```

## else 语句

使用 else 语句，您可以添加一些当 if 语句的条件为 False 时要运行的代码。

```
x = 300
y = 100
if y > x:
   print(“y is greater than x”)
else:
   print(“x is greater than y”)
```

在本例中，x 大于 y，因此第一个条件不成立，然后我们转到 else 条件，并打印屏幕显示“x 大于 y”。

## elif 语句

elif 关键字是 Python 的说法“如果前面的条件不成立，那么尝试这个条件”。

```
x = 33
y = 33
if y > x:
   print(“y is greater than x”)
elif y == x:
   print(“y and x are equal”)
else:
   print(“y is greater than x”)
```

在这个例子中，x 等于 y，所以第一个条件不为真，但是 elif 条件为真，所以我们打印到屏幕上“x 和 y 相等”。

您可以根据需要多次添加 elif。但是，请记住，只有第一次返回 True 的代码才会被执行。