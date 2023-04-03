# Python decorator:它如何改变其他类和函数的行为

> 原文：<https://medium.com/analytics-vidhya/python-decorator-how-it-changes-behaviour-of-other-classes-and-functions-a5cac813f79d?source=collection_archive---------22----------------------->

## 本指南通过描述性示例提供了关于 Python decorator 的简单易懂的知识

![](img/1aacbf358c34741f7f4e52f6595dfc2e.png)

如今，Python 已经成为世界上最流行的编程语言之一。它不仅简单易学，而且提供了许多有趣和有用的功能。Decorator ( **@** )，它允许我们修改其他函数和类的行为，必须是其中之一。它的主要目的是封装复杂的实现，但仍然让您的代码清晰美观。对于初学者来说，Decorator 可能有点混乱，但是本文将证明它比您想象的要简单得多。

# **语法**

```
@my_decorator
def hello_world():
    print('Hello world!')
```

在上面的例子中，我们可以看到一个非常简单的函数，它打印字符串" *Hello world* ！"。但是不同的是，它是用装饰器定义的，这导致了函数的一些变化。但是如果我们不知道 **@** ，不使用这个特性怎么重写上面的代码呢？这是答案:

```
def hello_world():
    print('Hello world!') hello_world = my_decorator(hello_world)
```

`my_decorator` 是一个可调用函数。在这个函数中，有另一个函数执行一些操作，包括调用作为参数传递的函数，在本例中是`hello_world`。如果这对于你来说仍然难以理解，让我们转到文章的下一部分。

# 创建您自己的装饰

现在我们定义上面提到的`my_decorator`函数:

```
def my_decorator(func):
    def add_star():
        print('***')
        func()
        print('***')
    return add_star @my_decorator
def hello_world():
    print('Hello world!') hello_world()
```

当像上面这样添加装饰器时，`hello_world`被传递给可调用的`my_decorator`。在内部，`add_star`被定义为在调用作为参数传递的`hello_world`之前和之后打印星形线的动作。最后，`add_star`被返回并赋给`hello_world`。从这一点来说，`hello_world`不再是原来的功能，而是一个新装饰的功能。毕竟，输出将是:

```
***
Hello world!
***
```

# 装饰函数包含参数

上面的例子是一段简单的代码，它只是将一些字符串输出到控制台。如果定义一个函数需要一些参数，为了接受它们，我们只需在定义装饰器返回的函数时添加`*args`和`**kwargs`。

```
def print_result_decorator(func):
    def print_result(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f'Your result is {result}')
        return result
    return print_result @print_result_decorator
def square(a):
    return a * a square(2)
```

输出:

```
Your result is 4
```

# 作为装饰者的类

本文展示了一些 function decorator 的例子，但是我们可以使用 class 来提供相同的功能。现在我将使用类风格重写`print_result_decorator`函数。

```
class ClassResultDecorator:
    def __init__(self, func):
        self.func = func def __call__(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        print(f'Your result is {result}')
        return result @ClassResultDecorator
def square(a):
    return a * a square(4)
```

输出:

```
Your result is 16
```

如上定义函数时，首先调用`__init__`方法初始化实例，其中`square`被传递并设置为实例的属性。然后实例被返回并分配给`square`。当调用`square(4)`时，意味着我们正在调用`ClassResultDecorator`实例和方法`__call__`被调用并执行其行为。

# 装饰器作为回调

Decorator 不仅仅是用来修改行为，它还有一个重要的应用，使得一个函数成为类中其他函数的回调函数。马上开始研究样品。

```
class Calculator:
    def connect(self, func):
        self.printer = func def sum(self, a, b):
        result = a + b
        if hasattr(self, 'printer') and self.printer is not None:
            self.printer(result)
        return result cal = Calculator() @cal.connect
def print_result(result):
    print(f'Your result is {result}') cal.sum(1, 2)
cal.sum(10, 10)
```

输出:

```
Your result is 3
Your result is 20
```

为了传递一个要被回调的函数，我们需要定义一个要被装饰的类函数，其目的是将传递的函数保存为类实例的一个属性。在这种情况下，`connect`就是那个，它将`print_result`保存为计算器实例`cal`的属性`printer`。每次我们调用`cal.sum()`，在这个方法里面，`printer()`会被调用，并把汇总结果打印到控制台。

# 结论

Python decorator 非常棒，是保持代码整洁和可读性的一个非常有用的特性。虽然开始可能会有些混乱，但我希望你读完这篇文章后能有一个清晰的理解。

最重要的是，感谢您抽出时间。