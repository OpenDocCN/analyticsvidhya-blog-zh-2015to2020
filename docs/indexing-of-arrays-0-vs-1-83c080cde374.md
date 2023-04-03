# 数组索引:0 对 1

> 原文：<https://medium.com/analytics-vidhya/indexing-of-arrays-0-vs-1-83c080cde374?source=collection_archive---------20----------------------->

![](img/99fb877b408132c953e2c902bc976c92.png)

照片由来自[佩克斯](https://www.pexels.com/photo/alberta-amazing-attraction-banff-417074/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)的[詹姆斯·惠勒](https://www.pexels.com/@souvenirpixels?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)拍摄

# 1.1:为什么是零？

大多数编程语言使用基于 0 的索引，即该语言中的数组从索引 0 开始。其中一个主要原因是公约。早在 1966 年，Martin Richards——IBM BPCL 的创造者——使用 0 作为初始索引，这样指针 p 就可以访问 p[T5 的内存位置。由于它的广泛使用和可移植性，C 建立了这个基于 0 的索引。从那时起，大多数编程语言，其中许多是 C 的衍生物，开始使用基于 0 的索引。](https://en.wikipedia.org/wiki/Zero-based_numbering#Origin)

后来，著名的计算机科学家 Edsgar W. Dijkstra 接着写了一篇名为《为什么编号应该从 0 开始》的笔记，列举并辩护了这种“奇怪”约定 [](http://www.cs.utexas.edu/users/EWD/transcriptions/EWD08xx/EWD831.html) 的所有理由。

根据 Dijkstra，为了表示数字 1，2，3 … 5 的集合，我们可以使用以下 4 个约定:

A.0 < i < 6

B. 0 ≤ i < 6

C. 0 < i ≤ 5

D. 0 ≤ i ≤ 5

Dijkstra laid out a rule for the *理想约定*。

*   约定应该能够显示一个空集

同样，Dijkstra 排除了 A 和 C，因为要表示从 0 开始，我们必须在符号中使用-1。Dijkstra 称这是“丑陋的”。

我们现在只剩下选项 B 和 d。根据 Dijkstra 提出的规则，为了表示一个空集，我们可以用下面的方式写它:

B.0 ≤ i < 0

D.0 ≤ i ≤ -1

同样，Dijkstra 称选项 D“丑陋”，不仅因为它使用了负数，还因为它使用了较小的数作为上限，较低的数作为下限。自然，我们倾向于选择 b。

除了这种主观的“丑陋”，许多递归算法使用 0 作为基本情况，如果没有基于 0 的索引，将更难实现。在许多组合算法中——比如得到一个集合的所有子集——空集经常出现，简单的符号表示更好。

此外，对于循环列表，可以使用模操作符在最后一个元素后将迭代器返回 0。

对于具有 n 个元素和不断迭代自身的变量 k 的循环列表，表示被访问的元素的索引的迭代器 I 可以以如下方式迭代:

i = (k % n)

注意，在访问第(n-1)个元素之后，我们退回到第‘零’个元素。

同样，C 和许多其他低级语言使用 0，以便可以以简单的公式方式访问数据 [](https://en.wikipedia.org/wiki/Zero-based_numbering#Numerical_properties) :

如果 a 是数组中第一个(初始的，不是位置 1 的)元素的内存地址，其中 s 是数组中每个元素的大小，I 是迭代器，则第 I 个位置的元素的内存地址可以表示为:

a + (i * s)

# 1.2:为什么一个？

一些语言如 Lua、Fortran、COBOL 和 Julia 使用基于 1 的索引。这种与惯例相反的索引实践的共同论点是“从一开始计数是很自然的”，并且在一定程度上是正确的。自从数学诞生以来，人类就学会了从 1 开始计数，因为不可能什么也“数”不出。

Julia 主要用于科学和数学研究领域，对科学家来说，从 1 开始计数更直观，因此更受欢迎。

Julia 最初也是为数学领域更好的编程语言而设计的，用于迭代 MATLAB，并使用类似语言中没有的通用编程功能。

然而，需要注意的是，在朱莉娅 [⁴](https://docs.julialang.org/en/v1/devdocs/offset-arrays/) 中改变索引是可能的。

另一个原因是新手友好。当编程新手学习索引一个列表时，将 1:3 想象成 1，2，3 比通常的 0:3 更直观，0:3 表示数字 0，1，2。

像 Pascal 和 Perl 这样的一些语言允许程序员定义数组的索引到他们选择的 [⁵](https://en.wikipedia.org/wiki/Zero-based_numbering#Usage_in_programming_languages) 。这个选项通常可以使解决问题变得容易得多(例如，算术级数的 3 个连续项之和取为(n-d)+(n)+(n+d)= 3n，而不是(n) + (n + d) + (n + 2d) = (3n + 3d)

高级语言应该使用基于 1 的索引的另一个原因是，它们不允许程序员单独访问内存位置，也不需要基于 0 的索引，这反而会使代码变得复杂。

此外，基于 0 的索引提供的“索引优化”不是真正的优化 [⁶](https://en.wikipedia.org/wiki/Zero-based_numbering#Numerical_properties) ，因为内存地址的访问可以通过用以下公式定义术语“b”来简化:

b = a-s

这将把存储器地址公式简化为:

b + (i * s)

这不需要如该公式所示计算每次迭代(I-1 ),该公式通常用作反对基于 1 的索引的论据:

a+(s *(I-1))

# 1.3:结论:哪个更好？

因为这是从 Julia 的角度写的，我们将检查每个系统相对于 Julia 的优点和缺点。

因为 Julia 的社区深深植根于纯数学和科学，所以基于 1 的索引是首选。此外，如果 Julia 进行切换，使其遵循一般的编程规范，许多脚本将停止工作，并导致关键研究项目的失败。

此外，将 1:3 索引为 1，2，3 的概念比将标准的 0:3 索引为 0，1，2 更直观。虽然 Dijkstra 已经展示了半开音程的内在美，但使用 1 作为基数是自然和直观的。我们还看到，基于 0 的索引并没有像通常认为的那样提供真正的内存访问优化。

基于 1 的索引的唯一缺点是循环列表和 1 索引语言中这些列表所需的笨拙设置。

总之，这可以归结为个人偏好、语言应用和索引，这样不仅可以解决问题，还可以更容易地调试和编写代码。

在我看来，语言应该像 Perl 和 Pascal 那样为程序员提供索引选项的灵活性。然而，如果没有正确地编写代码，或者没有正确地实现灵活的索引，这会使代码难以阅读。

对 Julia 来说，同样重要的是注意到指数是可以改变的。

考虑到 Julia 的社区、使用情况、当前标准以及对初学者的友好程度，基于 1 的索引是有意义的，并且会一直存在下去，除非 Julia 在类似于 Python 2 和 3 之间的重大更新中得到彻底改革。

# 1.4:引用和参考文献

[1]-引用自[https://en.wikipedia.org/wiki/Zero-based_numbering#Origin](https://en.wikipedia.org/wiki/Zero-based_numbering#Origin)的关于 BPCL 的信息

[2]-迪杰斯特拉的笔记可以在[http://www . cs . ut exas . edu/users/EWD/transcriptions/ewd 08 xx/ewd 831 . html](http://www.cs.utexas.edu/users/EWD/transcriptions/EWD08xx/EWD831.html)找到

[3]-内存访问的数值属性，引用自[https://en . Wikipedia . org/wiki/Zero-based _ numbering # numeric _ Properties](https://en.wikipedia.org/wiki/Zero-based_numbering#Numerical_properties)

[4]-关于改变 Julia 数组索引的更多信息:[https://docs.julialang.org/en/v1/devdocs/offset-arrays/](https://docs.julialang.org/en/v1/devdocs/offset-arrays/)

[5]-帕斯卡中索引的自定义性质:

[https://en . Wikipedia . org/wiki/Zero-base _ numbering # Usage _ in _ programming _ languages](https://en.wikipedia.org/wiki/Zero-based_numbering#Usage_in_programming_languages)

[6]-证明基于 0 的索引不提供优化:[https://en . Wikipedia . org/wiki/Zero-based _ numbering # numeric _ properties](https://en.wikipedia.org/wiki/Zero-based_numbering#Numerical_properties)

封面图片“湖光山色”由 Pexels 上的“James Wheeler”提供免费使用许可:[https://www . Pexels . com/photo/Alberta-amazing-attraction-banff-417074/](https://www.pexels.com/photo/alberta-amazing-attraction-banff-417074/)