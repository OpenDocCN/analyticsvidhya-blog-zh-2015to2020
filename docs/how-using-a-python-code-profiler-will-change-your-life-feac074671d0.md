# 使用“代码分析器”是多么强大的技能啊！！

> 原文：<https://medium.com/analytics-vidhya/how-using-a-python-code-profiler-will-change-your-life-feac074671d0?source=collection_archive---------36----------------------->

对我来说幸运的是，在作为计算机科学专业人员开始工作的前几周，我有机会与一位非常资深的开发人员坐在一起，探讨如何调试某个应用程序的性能问题，在那里我目睹了“魔术”——代码剖析的简单技巧。

![](img/b9673d6916240cdd85251a07ced1ee94.png)

# **使用代码分析器的基本原理**

> “过早的**优化**是万恶之源”

软件行业的每个人(希望)都知道 Donald Knuth 的这句话。这只是总结了为什么单纯的启发和直觉不应该强迫进行大量的优化迭代，也就是说，即使在调查原因之前也不要进行重构，一些看起来非常琐碎的事情可能会损害性能，反之亦然。

这是一件非常明显和琐碎的事情，大多数人实际上都意识到了这一点，但是由于我们在如何优化代码、寻找一些我们认为有问题的东西时存在一些固有的偏见，我们没有利用工具来提供更清晰的画面，一些小问题只是在雷达下滑动。

# **何时进行优化**——需要记住的事情

1.  预感—“***这可能会更快*** ”，当你有这种感觉，资源可能会被不必要的浪费，使用一个分析器来找出。
2.  在你想要花多少时间写/重复一段代码和你想要它有多高的性能之间的权衡。
3.  对于小脚本来说，这可能非常方便，有时我们在开发脚本来完成一些琐碎的任务(如报告或抓取)时，并不会仔细查看，但这些事情可以优化，并有助于提高您作为计算机科学专业人员的技能。

## **TL；博士**

*在看一个非常琐碎的例子之前，建议你看一下 Jake VanderPlas* *关于* [*优化你的数字代码的七个策略*](https://www.youtube.com/watch?v=zQeYx87mfyw) *的 PyCon 演讲，这实际上会让你感受到这个东西有多强大，以及每个人都必须至少知道的一些正确的策略/实践。*

# **> >导入线条轮廓图**

幸运的是，在 Python 中进行代码剖析相当容易，因为有一堆工具，其中两个最受欢迎

1.  [**c profile**](https://docs.python.org/3/library/profile.html)**—这是 Python 标准库中内置的代码分析器**
2.  **这是另一个实用程序，我个人认为它非常方便，下面的阅读涵盖了一个简单的例子。**

**下面的脚本只是将格式化的日期写入一个文件，要使用这个工具，您需要导入 line_profiler(当然首先下载它),然后我们可以使用 *@profile* decorator 和我们想要查看的方法。**

**我通常从 main 方法开始，然后单步执行每个报告大部分时间消耗的调用。**

**要运行这个，有几个选项，首先我们将使用 line_profiler 提供的“kernprof”实用程序**

```
$kernprof -v -l temp/main.py
```

**这将运行脚本并在终端中显示结果，为了将输出保存到文件中并在以后进行可视化/分析，您也可以使用-o 选项。**

**从上面的结果可以看出， *main()* 方法似乎没什么奇怪的，但是对于 *formatted_dates()* 方法，在 L:10 我们花费了大部分时间，70%的时间在*out . append(date . strftime(" % Y-% m-% d))***

**现在我们能改进这个吗？—“将日期格式化为字符串的最快方法”，让我们看看**

**通过删除方法调用 *strftime* ，我们看到从 ***145.0 下降到 33.0！！*****

***[* ***注意*** *:每次运行时都会有变化，因为使用@profile 会有一些开销，所以最好是平均一下，然后再做一个推断】***

**现在让我们试着看看我们是否能进一步改进主要功能—**

**通过改变 strftime 和我们写行的方式，通过在列表中添加条目时添加' \n '，我们看到调用减少了——**

***dates = formatted _ dates(start)*从**调用 *169.0 到*74.0****

## **总结—**

1.  **了解你的工具并利用它们，而不是仅仅依靠你的直觉。**
2.  **从根方法开始，然后分别深入每个函数，实际查看可以改进的地方。**
3.  **永远记住，*在优化一个片段所需的努力和它给你的整个脚本带来的性能提升比率之间的权衡。***
4.  ******line_profiler*** 还有一些更多的特性供你探索，比如忽略加载一个模块所花费的时间等等，还有输出文件生成的方式，有一些不同的方式可以可视化以做出更好的推断。***
5.  ***有一种方法可以使用魔术方法，并使用 line_profiler 模块提供的魔术方法在 jupyter 笔记本中进行分析。***
6.  ***此外，您可以使用 [atexit](https://docs.python.org/3.7/library/atexit.html) 向 line_profiler 实例注册一个函数并使脚本工作，而不必使用 ***kernprof*** ，更多信息请参见[https://lothiral Dan . github . io/2018-02-18-python-line-profiler-without-magic/](https://lothiraldan.github.io/2018-02-18-python-line-profiler-without-magic/)***

## ***参考文献—***

*   ***[Jake Vander plas——Performance Python:优化数字代码的七种策略](https://www.youtube.com/watch?v=zQeYx87mfyw)***
*   ***[马特·戴维斯— Python 性能示例调查— PyCon 2018](https://www.youtube.com/watch?v=yrRqNzJTBjk)***
*   ***Github — [line_profiler](https://github.com/pyutils/line_profiler)***
*   ***[像老板一样剖析 Python](https://zapier.com/engineering/profiling-python-boss/)***
*   ***[利用 astexit 注册一个个人资料](https://lothiraldan.github.io/2018-02-18-python-line-profiler-without-magic/)***