# 使用 Numba 加速 Python 中的余弦相似性计算

> 原文：<https://medium.com/analytics-vidhya/speed-up-cosine-similarity-computations-in-python-using-numba-c04bc0741750?source=collection_archive---------6----------------------->

![](img/6c13c72fe167a2868dc9e7502c76d1e0.png)

> 对机器学习主题感兴趣或需要一些帮助？
> 
> **取得联系—**[**【https://linktr.ee/pranaychandekar】**](https://linktr.ee/pranaychandekar)

根据 IEEE 标准，Python 在过去的三年里一直是顶级编程语言。它也是用于构建机器学习应用程序的语言。作为一种解释语言，它加快了开发速度。然而，这同样会使它在运行时变慢，因为它每次都必须编译和执行每条语句。这在缩放过程中成为一个问题。

这就给我们带来了一个问题——“为了加快速度，我们能不能像其他编译器语言一样编译 python 代码一次，或者至少编译它的一部分？”，“这样会不会更快？”

答案是**——是的，我们可以**！

在本文中，我们将借助一个实验来了解如何使用 Numba 加速 Python 中的数值计算。

# 解决方案— Numba

根据网站介绍， [Numba](https://numba.pydata.org/) 是一个开源 JIT(Just In Time)编译器，它将 Python 和 NumPy 代码的子集翻译成快速机器代码。它设计用于 NumPy 数组和函数。它优化了面向数组和数学密集型的 python 代码。

为了验证 numba 的说法，我用机器学习中最常用的功能之一尝试了 Numba，看看有什么不同。
**余弦相似度计算**。

# 实验

在这个实验中，我在两个有和没有 numba 的 50 维 numpy 数组之间进行了余弦相似性计算。

余弦相似性 python 函数。

余弦相似函数

numba 也有同样的功能。

带数字装饰器的余弦相似函数

我针对不同的计算次数运行了这两个函数，以观察计算时间的差异。

# 结果

结果

差别是明显的。numba 的使用使我们的计算速度提高了数倍。

# 外卖食品

在这篇文章中，我们只是触及了表面。我们可以用 Numba 做更多的事情。但我会把这种探索留给你。您可以在下面的资源库中找到 jupyter-notebook 的完整实验。

[](https://github.com/pranaychandekar/numba_cosine) [## pranaychandekar/numba _ 余弦

### …

github.com](https://github.com/pranaychandekar/numba_cosine) 

> 对此类话题感兴趣或需要一些帮助？
> 
> **取得联系—**[**https://linktr.ee/pranaychandekar**](https://linktr.ee/pranaychandekar)

# 引文

1.  [https://numba.pydata.org/](https://numba.pydata.org/)
2.  [https://spectrum . IEEE . org/computing/software/the-top-programming-languages-2019](https://spectrum.ieee.org/computing/software/the-top-programming-languages-2019)