# Python 函数

> 原文：<https://medium.com/analytics-vidhya/python-functions-46853ec5ee4b?source=collection_archive---------11----------------------->

![](img/7b1305e7604ec5f48e973c107b1ce382.png)

## 在 Python 中定义和使用函数

如果您一直在关注我的 python 初学者系列，您一定已经注意到我们的脚本都是一次性代码块。此外，我强调你的代码需要有良好的结构，因为这使它对其他人可读。在这篇文章中，我将向你展示如何定义和使用 Python 函数。

获得良好可读性和可重用性的一个方法是将 Python 代码组织成可重用的程序片段，称为**函数**。使用函数，您可以给语句块命名，并在需要时在程序中的任何地方调用它们。这称为 ***调用*** 功能。

我们将讨论在 Python 中创建函数的两种方法:使用`**def**` 语句和`**lambda**` 语句。

> **注:** `def` —定义任意函数，`lambda` —定义匿名函数。

Python 打包了几个内置函数，我们可以在应用程序中随时随地使用它们。你猜对了，你以前见过他们；包括`print`、`len`、`range` 等等:

```
len(‘hello’) #check the length**output: 5**
```

> 这里`**len**` 是函数名，‘hello’是函数的自变量。

有趣吧？当你开始通过将功能组织成块来定义你的个人功能时，这变得更加令人兴奋。

## **定义功能**

Python 函数可以用关键字`**def**` 定义，后面是函数名和括号`()`。输入参数/自变量可以在括号内传递:

**语法**

```
def function_name(arguments):
    “function_doc_string”
    statements
    return expression
```

提供一个数字时返回斐波纳契数列的函数示例:

> **注:**斐波纳契数列，是这样的，每一个数都是前面两个数的和，从 0 和 1 开始。即 F0 = 0，F1 = 1，Fn = Fn-1 + Fn-2
> 
> 对于 n > 1

```
def fibonacci(N):
    f = []
    x, y = 0, 1
    while len(f) < N:
        x, y = y, x + y
        f.append(x)
    return ffibonacci(15) #call function and pass argument
**output:** [1,1,2,3,5,8,13,21,34,55,89, 144, 233, 377, 610]
```

## **功能参数**

函数参数是我们传递给函数进行操作的值。它们就像变量；声明和赋值是在函数运行时完成的。通常当我们定义一个函数时，我们希望函数大部分时间使用某些值；这些被称为 ***默认*** 值，但我们也想给用户一些灵活性。

> **注意:**如果需要，可以在传递值时指定参数名

```
#function to sum two numbers
def sum(x,y):
    result = x + y
    return resultsum(x=1, y=4) #call function, specify parameter name and value**output: 5**
```

## **λ函数**

它们通常被称为 ***匿名函数*。**与我们使用`def` 关键字定义函数不同，`lambda` 函数适用于定义简短的一次性函数:

```
sqr = lambda x: x**2
sqr(2)**output: 4**
```

这相当于使用了`def` 这样的关键字:

```
def sqr(x):
    return x**2sqr(2) #call function **output: 4**
```

现在，你会问为什么你会想要使用匿名函数。因为在 Python 中事物被组织成对象，包括函数。这意味着我们可以提供函数作为其他函数的参数，如下所示:

```
#function to convert text to upper case
def yell(text):
    return text.upper()#pass the yell function as parameter into the Python built-in print function
print(yell(‘Hello’))
```

函数为我们的应用程序提供了改进的模块化，并使代码重用成为可能。

我们已经介绍了开始定义和使用函数所需要知道的一切，接下来就看你自己了。请记住:

> “成功并不总是意味着伟大。这是一致性的问题。持续的努力工作会带来成功。伟大终将到来。”—道恩·强森

我希望你喜欢这篇文章，它对你有帮助。一定要和别人分享。

你可以在这里查看我其他有趣的话题。

关注我的 [*中型*](/@ezekiel.olugbami) *，* [*Twitter*](https://twitter.com/OlugbamiEzekiel) 或*[*LinkedIn*](http://www.linkedin.com/in/ezekiel-olugbami)获取关于 Python、数据科学和相关主题的提示和学习。*

***快乐编码:)***