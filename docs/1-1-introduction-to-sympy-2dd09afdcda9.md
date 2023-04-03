# 1.1 SymPy 简介

> 原文：<https://medium.com/analytics-vidhya/1-1-introduction-to-sympy-2dd09afdcda9?source=collection_archive---------0----------------------->

SymPy 是 python 库对符号数学的支持。https://www.sympy.org/是 SymPy 的官方网页。SymPy 的主要目标是为 Python 语言中的**算法计算**提供支持。SymPy 是“免费使用且非常轻量级的库”。SymPy 完全用 Python 编码，因此它有与 Python 相同的限制。

![](img/7ab89c26ec8d6f33c020d43f58538e5f.png)

[https://upload . wikimedia . org/Wikipedia/commons/d/de/Sympy-160 px . png](https://upload.wikimedia.org/wikipedia/commons/d/de/Sympy-160px.png)

SymPy 的最新版本是 1.3(2018 年 9 月发布)。Anaconda 自带预装的 SymPy。您可以更新如下:

```
$conda update sympy
```

SymPy 对 mpmath 有额外的依赖性，mpmath 也附带了 anaconda。您可以按如下方式安装它:

```
$conda install mpmath
```

SymPy 的第一个例子:

```
from sympy import * 
x, y = symbols('x y')
expr = x + 2*y
expr
```

在上面的代码片段中，我们首先在内存中导入 sympy。接下来，我们创建两个符号:x 和 y，它们将用于形成我们的多项式方程。在下一行，我们创建了一个示例方程:x+2y，然后我们将它打印在控制台上。以下是输出:

```
x + 2*y
```

如果你想把它作为格式化输出 x+2y 打印出来。必须执行以下代码语句。

```
init_printing(use_unicode=True) 
```

执行上述代码后，如果我们运行完第一个示例，我们会得到以下输出:

```
x+2y
```

现在，让我们创建另一个等式:x + x-3，并从等式中减去 2x。这将创造更多关于 SymPy 的理解。

```
x, y = symbols(‘x y’)
expr = x**2 + x — 3
expr-2*x
```

输出:

```
x²-x-3
```

我们可以求出给定方程的平方根，如下所示:

```
solve(x*x-2*x-15) output: [−3,5]
```

现在，让我们用实际数字代替符号值来求解方程:

```
a,b,c = symbols(‘a b c’)
expr = a**3 + 4*a*b — c
expr.subs([(a, 5), (b, 8), (c, 3)]) output: 282
```

如果我们用 numpy 和 sympy 来计算一个数的平方根，这是有区别的。逐一执行以下代码语句，以检查差异:

```
numpy.sqrt(4)
numpy.sqrt(6)
sympy.sqrt(4)
sympy.sqrt(6)output:
2.0
2.449489742783178
2
√6
```

如果它不是一个完美的正方形，Sympy 不会计算 sqrt。

在 SymPy 中，我们可以找到一个给定方程的因子，利用这些因子展开也是可能的。看看这个例子:

```
eq = x**2 + 4*x — 45
factor(eq)output:
(x−5)(x+9)expand((x + 9)**3)output:
x³ + 27x² + 243x + 729
```

如果你正在阅读这一部分，意味着你已经浏览了整篇文章。非常感谢你的支持。在这篇文章中，我们简单介绍了 SymPy。这对研究社区的人们和其他想不断学习新事物的人来说是非常有用的。再次感谢。