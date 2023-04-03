# 对于范围内的 I(n):

> 原文：<https://medium.com/analytics-vidhya/for-i-in-range-n-45e5d55ce2f4?source=collection_archive---------17----------------------->

## 我的 Python 数据科学工作流中的迭代器和可迭代对象

![](img/174d66d3b772a64fb38a8d0730e1a0d0.png)

照片由 [engin akyurt](https://unsplash.com/@enginakyurt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/fractal?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

在我从事数据科学的大部分职业生涯中，我没有充分利用 Python 编程语言。老实说，我从未正式学习过 Python，因此我最初的项目与其说是结构化思维，不如说是黑客。在过去的几天里，我决定重新审视我在 Python 中的一些老项目，并将更好的编程思想应用到这些项目中。我一直注意到的一个错误是代码中对可重复项的误用。这篇博客是对这两个基本 Python 概念——迭代器和可迭代对象——的论述。

## 阶乘和

作为一个运行的例子，让我们考虑一个将第一个`n`整数的阶乘相加的程序。这是一个略显做作的例子，但这种模式在我见过的许多数据科学应用程序中非常常见。如果你像我一样，你会写一个类似下面的程序。

您会立即注意到这个程序的一个问题。我们正在重新计算`factorial(1) to factorial(n-1)`来计算`factorial(n)`。因此，智能阅读器将重写一个高效的专用函数。

然而，在函数风格的编程中，应该尽可能避免专门化的函数。一个专门化的`fact_sum(n)`函数不能在任何其他需要阶乘的程序中使用。作为一名数据科学家，我写的大部分代码都是函数风格的，我希望避免专门化函数。要更系统地了解函数式编程，请查看[本指南](https://docs.python.org/3/howto/functional.html)。

## 迭代器和迭代器

另一种方法是开发一个 Iterable 并使用迭代器来获得阶乘。我们大多数人在用 Python 编写的每个程序中都使用过迭代器。这篇文章的标题`for i in range(n):`就是这样一个例子。范围函数返回一个迭代器。下面的代码显示了阶乘问题的迭代器模式。在这个例子中，我们通过实现一个`__iter__` &和一个`__next__`方法来创建一个 Iterable。

在你们中的任何人抱怨这是实现 Iterable 和迭代器本身的可怕模式之前，让我公开我同意。我在这里挥了一下手，因为这是数据科学工作流中很少使用的模式。尽管如此，这是一个重要的想法。因此，在继续讨论生成器之前，让我来解构一下上面的代码。通过实现这些特殊的方法，我们为 Python 创建了一种机制，可以在 for 循环中使用迭代器`facts`，如下面的示例`for f in facts`所示。还要注意使用`StopIteration`作为迭代器没有更多信息要发送的信号。所有 Python 内置的都有一个优雅的处理`StopIteration`的方式。因此，for 循环确切地知道在`StopIteration`上做什么。我会参考 Python 文档来获得更多关于 Iterables 和 Iterators 的信息。

## 发电机

生成器是数据科学工作流中最常用的迭代机制。生成器被定义为使用 def 关键字的函数。但是它没有返回一个值！让我们看看我们的阶乘例子。生成器函数构建了一个包装函数体的生成器对象`my_iter`。每次我们调用生成器对象上的`next(...)`(下一个由 for 循环调用)时，调用都会评估函数，直到`yield`为止，然后 ***函数被挂起。*** 这允许**状态持续**，我们得到一个又一个阶乘，而无需重新计算任何先前计算的值。生成器对象遵循与迭代器相同的协议——一旦函数体`returns`出现，它们就引发`StopIteration` 。

关于使用`def`产生的一个有趣的注意事项— `factorial2`还不是一个函数，我们使用`def`关键字。似乎有一个关于是否使用 def 关键字或使用新关键字的争论。在讨论利弊之后，BDFL(圭多)做了最后的决定。

> 定义它保持。任何一方的论点都不能完全令人信服，所以我咨询了我的语言设计师的直觉。它告诉我，PEP 中提出的语法是完全正确的——不冷不热。但是，就像希腊神话中特尔斐的神谕一样，它没有告诉我为什么，所以我没有反驳反对 PEP 语法的论据。我能想到的最好的词(除了同意已经做出的反驳之外)是“FUD”。如果这是语言的一部分，我非常怀疑它会出现在安德鲁·库奇林的“Python 疣”页面上。

## Itertools 和 Functools

在本博客的结尾部分，我们将讨论 Itertools。对于大多数应用程序，我们通常不必创建自己的迭代器甚至生成器。Itertools 和 Functools 有很多很好的函数可以为我们完成所有这些工作。该模块中最常用的两个函数是`zip`和`enumerate`。但是还有更多最著名的——`count(), accumulate(), chain(), filter()`和[还有更多](https://docs.python.org/3/library/itertools.html#itertools-recipes)。

我们的最后一段代码是使用 itertools 实现阶乘代码。

就是这样！

## 结论

我希望这能对 Python 中迭代的工作原理有一点概述。大多数有经验的大蟒蛇都知道这种模式。但是如果你和我一样，不知道所有的细节，我希望我能够对迭代器、可迭代对象和生成器的世界有所了解。