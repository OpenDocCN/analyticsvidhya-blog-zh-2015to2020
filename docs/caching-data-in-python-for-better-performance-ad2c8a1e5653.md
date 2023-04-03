# 在 python 中缓存数据以获得更好的性能

> 原文：<https://medium.com/analytics-vidhya/caching-data-in-python-for-better-performance-ad2c8a1e5653?source=collection_archive---------10----------------------->

![](img/c730ec1f24c1a5d122ad77e398ab1b9e.png)

萨法尔·萨法罗夫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

**要讨论的事情**:优化、装饰、阶乘、缓存

作为一名开发人员，你花了很多时间来解决问题。如你所知，问题可以用不同的方法解决，最佳方法取决于具体的环境，通常没有一个正确的答案。也就是说，对于某些特定的情况，可能有一个最佳的解决方案，用不同的方法测试问题的能力对于开发人员来说是非常有益的。

在本文中，我们将研究执行同一个(阶乘)函数的两种方法，并分析其优缺点。作为奖励，我们将使用 decorators，只是为了引入一个非常有用的 Python 特性。

对于那些忘记阶乘是什么的人来说，阶乘基本上是一个数和它下面所有数的乘积之和，直到 0。例如:4 阶乘(4！)就是 4*3*2*1 = 24。7!就是 7*6*5*4*3*2*1 = 5040。相当直接。

我们如何编写一个计算阶乘的函数？这这一种方式:

```
def calc_factorial(num):
    if num < 0:
        raise ValueError('Please use a number not smaller than 0')
    product = 1
    for i in range(num):
        product = product * (i+1)
    return productcalc_factorial(4)
---> 24calc_factorial(7)
---> 5040
```

虽然这可能以合理的速度运行(至少数量很少)，但您可以想象这里有些东西有点浪费。如果我要求(50)的阶乘，然后一次又一次，Python 每次都必须重新计算。虽然这是同样的结果，但还得重新计算。

现在让我们用一种不同的方法来做这件事，我们很快就会看到它的好处。我们将创建一个类来缓存任何预先计算的结果，这意味着它将存储结果，所以下一次询问相同的数字时，就不需要计算了。那会是什么样子呢？

```
class FactClass:
    def __init__(self):
        self.cache = {} def __call__(self, num):
        if num < 0:
            raise ValueError('Please use a number not smaller than 0')
        if not num in self.cache:
            product = 1
            for i in range(num):
                product = product * (i+1)
            self.cache[num] = product
            return(self.cache[num])
        return(self.cache[num])
```

但是在我们运行这个之前，让我们添加一些东西。让我们添加一个装饰。装饰器将在另一篇文章中详细讨论，但是为了我们的目的，您可以将它看作是一个向另一个函数添加功能的函数。比方说，对于许多不同的用户请求，我需要确保用户的年龄大于 18 岁。我可以一次定义一个函数，然后用简洁明了的方式添加它，而不是为每个检查相同的操作编写一个函数。*正如我们上面讨论的，这当然可以通过许多其他方式来实现。

我们要在这里添加的功能是时间计算。我们写一次并把它修饰(添加)到其他函数中，而不是写两次(如果我们要扩展这个实验的话，还要写更多次)。这是一种选择:

```
# time is a function that takes a function as an argument.
def time(fn):
    from time import perf_counter

    def inner(*args):
        start = perf_counter()
        result = fn(*args)
        end = perf_counter()
        elapsed = end - start
        print('time elapsed', elapsed)        
        return result
    return inner
```

这个装饰功能是怎么搭建的？我们定义一个函数，它接受一个函数(修饰过的，我们“辅助”的原始函数)并返回另一个执行操作的函数，在本例中是计时。我们如何使用它？很简单，我们只需要在修饰函数上加上:@ <name of="" decorating="" function="">。时间函数是用额外的功能来修饰另一个函数(修饰函数)。</name>

因此，让我们用这个小的(但有用的)附加功能重写前面的函数。这是前一个函数的复制粘贴，只是在顶部添加了@time:

```
# @time is the only difference
@time
def calc_factorial(num):
    if num < 0:
        raise ValueError('Please use a number not smaller than 0')
    product = 1
    for i in range(num):
        product = product * (i+1)
    return productcalc_factorial(4)
---> time that passed 4.125002305954695e-06
---> 24
```

这和以前一样，只是我们增加了计算时间的功能。除了@time 之外，我们不需要写任何东西，我们得到的是“过去的时间”。这将允许我们区分两种方法之间的差异(在时间上)。它在这个类中完全一样，所以我们可以继续。

当计算非常小的数字时，可能没有有意义的差别。但是假设这个服务一天服务一百万人，10%的人需要做，这些都是很多计算。让我们用一个循环来演示许多不同的用户做同样的计算，并看看它们的区别。

```
# calculate 5 times the factorial of 100,000
for i in range(5):
    calc_factorial(100000)time elapsed 3.298186706000706
time elapsed 3.3299793360056356
time elapsed 3.354474993000622
time elapsed 3.271242368995445
time elapsed 3.288305276000756
```

正如我们所看到的，在常规(非缓存)函数中，时间几乎相同。循环在计算方面并没有变得更好，因为它基本上一次又一次地做同样的操作。缓存类呢？

```
# create an instance of FactClass
caching_class = FactClass()# calculate 5 times the factorial of 100,000
for i in range(5):
    caching_class(100000)time elapsed 3.3387860439979704
time elapsed 1.1549011105671525e-05
time elapsed 8.960050763562322e-07
time elapsed 6.070040399208665e-07
time elapsed 6.949994713068008e-07
```

如你所见，这是完全不同的。该函数第一次运行时，花费的时间也差不多。但是第二次呢？它跑了 289，097 倍！*用 calc_factoria 的第一次运行时间除以 caching_class 的第二次运行时间。另外三次呢？嗯，差别不大，因为函数只是从字典中获取它(详细说明将很快出现)。这是缓存的主要好处。我们不必重新计算，因为它已经存储了。想象一下这样做 10k 次而不是 5 次，你就会明白为什么这很重要。

当我写乐谱是简单地从字典中获取时，我是什么意思？嗯，caching_class 有一个名为 dict 的属性，在第一次计算后存储分数，然后提取而不是重新计算。看起来是这样的:

**我将使用比 100，000 小的(100)阶乘，因为它会变得非常大，在屏幕上看起来不太好。

```
caching_class.__dict__['cache']{100: 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000}
```

正如我们所看到的，这是一个普通的字典，键是 100，值是这个大数字。这就是我们如何获得这种超级速度。

好的，我们现在看到它更快了，快得多，但是…你注意到问题了吗？缓存需要内存。当场计算时，我们不存储，因此我们不需要存储。通过缓存，我们做到了。缓存许多不同的数字，当每个数字可能非常大时，可以使这个有问题的内存明智。这是一个应该根据手头的问题来考虑的问题。

我们已经看到了解决一个问题的两种不同方法，希望你能学到一些有用的东西！

对于任何问题/建议/建设性的仇恨，请随时联系 Bakugan@gmail.com 或 Linkedin