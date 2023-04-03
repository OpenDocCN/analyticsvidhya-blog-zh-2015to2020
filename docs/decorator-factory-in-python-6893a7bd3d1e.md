# Python 中的装饰工厂

> 原文：<https://medium.com/analytics-vidhya/decorator-factory-in-python-6893a7bd3d1e?source=collection_archive---------4----------------------->

![](img/112610d411e889aa289d45cb52a6cf81.png)

罗伯特·马修斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

装饰器是强大而有用的，但是你需要知道如何正确地使用它们来充分利用这个奇妙的特性。在本文中，我将讨论如何使用带有附加参数的装饰器(不仅仅是函数)。我们将看到两种这样做的技术，一种是用函数，另一种是用类。

我们为什么需要这个？让我们从一个关于装饰者的简短介绍开始，然后看看我们受到限制的一个方面。*如果你已经了解装饰者，请随意跳到“装饰者之后”标题。

装饰器是 python 的一个特性，它为一些函数增加了一些功能。在实践中，这意味着我可以编写一个装饰器，它做一些事情，通过装饰另一个函数，我将这个功能添加到一个新的函数中，而无需编写两次代码。举个例子就容易理解了:

```
def typing(num):
    print('typing the number', num)typing(4)
---> typing the number, 4
```

这是一个相当简单的函数。现在，假设我有 4 个函数，我想给它们都加上计时，我想知道它运行了多长时间。一个解决方案是给每个函数增加计时功能，但是你可以想象这不是很枯燥(不要重复)。相反，我们所做的是编写这个计时函数一次，并将这个函数添加(修饰)到任何其他需要“它的服务”的函数中。

```
# Timing functionality from Python's built-in module
from time import perf_counterdef decorating(fn):
    def inner(*args):      
        start = perf_counter()
        result = fn(*args)
        end = perf_counter()
        elapsed = end - start
        print(result)
        print('elapsed', elapsed)
    return inner
```

因此，我们有一个函数，它开始计算时间，运行原始函数(使用计时服务的函数)，对一个新变量再次计算时间，并返回差值(例如经过的时间)。让我们定义一个计算阶乘的函数，并向它添加计时功能。

```
@decorating
def calc_factorial(num):
    if num < 0:
        raise ValueError('Please use a number not smaller than 0')
    product = 1
    for i in range(num):
        product = product * (i+1)
    return productcalc_factorial(4)
---> 
24
elapsed 5.719979526475072e-06calc_factorial(10)
---> 
3628800
elapsed 6.7150103859603405e-06
```

就在那里！我们免费获得计时功能，只需添加这个@装饰符号。你可以想象为什么这比一遍又一遍地写所有的东西更具伸缩性。那些是装饰者，这是主要的想法。

# 装修工之后

我们已经看到了 decoratorss 是如何非常有用的，也就是说，以这种方式编写 decorator 有一个重要的限制，我们将讨论这个限制。如果我想给我的 decoration**ing**函数添加一些参数，这些参数稍后会被 decora **ted** 函数使用，会发生什么呢？。也就是说，如果我不仅想增加功能，还想传输数据，那该怎么办？例如，假设我正在测试不同的方法来编写相同的函数以达到优化的目的。我想多次运行同一个函数，并得到它所用的平均时间。我能把循环次数作为参数传递吗？不是这样的。为了能够传输数据，我们需要创建一个所谓的**装饰工厂**

*像任何可以用许多方式完成的事情一样，我们将使用一种方式来演示从装饰工厂一路传输数据。

# **装饰工厂**

装饰工厂是一个返回装饰者的函数。我们将返回一个装饰器(名为*“dec”*)，而不是返回一个函数(在我们的例子中是内部的)。让我们看看这个:

```
from time import perf_counterdef decorator_factory(loops_num):
    def decorating(fn):
        def inner(num):   
            total_elapsed = 0
            for i in range(loops_num):
                start = perf_counter()
                result = fn(num)
                end = perf_counter()
                total_elapsed += end - start
            avg_run_time = total_elapsed/loops_num
            print('result is', result)    
            print('num of loops is', loops_num)
            print('avg time elapsed', avg_run_time)
        return inner
    return decorating
```

请注意，这里我们同时返回了内部函数和装饰函数。我们返回的是一个室内装潢师。这使得内部函数能够访问一些额外的参数(例如 a，b)并使用它们。
**对于这个特殊的例子，我们将使用 num 而不是*args 来使它更容易理解。通常*args 可能更好，因为它更灵活。*

现在，我们不仅拥有计时功能，还可以访问更多参数，这是我们以前无法做到的。

```
[@decorator_factory](http://twitter.com/decorator_factory)(500)
def calc_factorial2(num):
    if num < 0:
        raise ValueError('Please use a number not smaller than 0')
    product = 1
    for i in range(num):
        product = product * (i+1)
    return productcalc_factorial2(4)
--->
result is 24
num of loops is 500
avg time elapsed 1.5613697469234467e-06
```

现在，我们也可以将循环功能用于其他函数。

```
[@decorator_factory](http://twitter.com/decorator_factory)(5)
def string(s):
 print(s)string('this is working'
--->
this is working
this is working
this is working
this is working
this is working
num of loops is 5
```

写一次，到处跑。

**类装饰工厂**

现在让我们看看生成相同行为的另一种方法，但是使用一个类。一个类有时更容易进行更复杂的操作，所以它是一个重要的工具。让我们完全像装饰工厂函数那样做，这次是用一个类。

```
class Decorator_Factory_Class:
    def __init__(self, num_loops):
        self.num_loops = num_loops
    def __call__(self, fn):
          def inner(num):   
            total_elapsed = 0
            for i in range(self.num_loops):
                start = perf_counter()
                result = fn(num)
                end = perf_counter()
                total_elapsed += end - start
            avg_run_time = total_elapsed/self.num_loops
            print('num of loops is', self.num_loops)
            return result
          return inner
```

因此，我们现在可以像使用函数一样使用该类

```
[@Decorator_Factory_Cl](http://twitter.com/Decorator_Factory_Cl)ass(5)
def calc_factorial2(num):
    if num < 0:
        raise ValueError('Please use a number not smaller than 0')
    product = 1
    for i in range(num):
        product = product * (i+1)
    return productcalc_factorial2(4)
--->
num of loops is 5
avg_run_time is 2.301810309290886e-06
24
```

所以…我们已经看到了装饰者是如何工作的，他们的好处是什么，什么是有意义的限制。我们还看到了如何通过定义一个装饰工厂来克服这个限制，这个工厂返回一个装饰器。定义几个这样的装饰器，可以减少代码的重复，加快开发速度，减少 bug。

希望你得到了重要的东西！

对于任何问题/建议/建设性的仇恨，请随时联系 Bakugan@gmail.com 或 Linkedin