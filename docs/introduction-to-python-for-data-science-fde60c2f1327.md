# 面向数据科学的 Python 简介

> 原文：<https://medium.com/analytics-vidhya/introduction-to-python-for-data-science-fde60c2f1327?source=collection_archive---------13----------------------->

Jupyter 笔记本的集合，旨在在几个小时内教你 Python 的基础知识。

![](img/ea752949a39e270b5af1b8a5cd27fc3d.png)

我的电脑屏幕显示了 python 库的源代码[隐写术](https://github.com/computationalcore/cryptosteganography)

# Python 是什么？

Python 是一种解释型的高级通用编程语言。它由[吉多·范·罗苏姆](https://en.wikipedia.org/wiki/Guido_van_Rossum)创作，并于 1991 年发行。

它的设计理念通过强制代码缩进来强调代码的可读性。它的语言结构和面向对象的方法旨在帮助程序员为小型和大型项目编写清晰的逻辑代码。

这种语言的名字是为了向英国喜剧组合 [Monty Python](https://en.wikipedia.org/wiki/Monty_Python) 致敬——偶尔也用有趣的方式来介绍教程和参考资料，例如引用垃圾邮件和鸡蛋的例子(来自[著名的 Monty Python 小品](https://en.wikipedia.org/wiki/Spam_(Monty_Python))，而不是标准的 [foo 和 bar](https://en.wikipedia.org/wiki/Foobar) 。

# 为什么是 Python？

作为一种[解释型语言](https://en.wikipedia.org/wiki/Interpreted_language)和一种[动态类型语言](https://en.wikipedia.org/wiki/Dynamic_programming_language)，python operations 与[编译的](https://en.wikipedia.org/wiki/Compiled_language) — [静态类型语言](https://en.wikipedia.org/wiki/Type_system) —如 C 或 C++相比，执行速度要慢得多。

尽管如此，Python 还是被广泛使用，即使它比其他语言慢一些，因为:

*   **很容易学**

任何渴望学习这种语言的人都能轻松快速地学会。Python 的学习曲线更短，并且通过提供易于理解的语法而优于其他语言。

*   **Python 更有生产力**

与其他几种编程语言相比，它是一种更简洁、更有表现力的语言，执行相同的操作需要更少的时间、精力和代码行。

*   **公司可以优化员工的时间**

执行速度没有业务速度重要。如果开发人员编写解决方案的速度比使用另一种语言快几倍，公司就可以节省时间和资源。而[员工时间往往是最昂贵的资源](https://books.google.com.br/books?id=QUoJBAAAQBAJ&pg=PA142&dq=Human%20resource%20management&hl=en&sa=X&ved=0ahUKEwjRm_TIn6foAhVJI7kGHWLbDJ8Q6AEINDAB#v=onepage&q&f=false)。

*   **通过快速创新提升竞争力**

由于使用 python 学习和编码解决方案通常更快，因此可以更快地创建新的库和代码贡献，这使得生态系统更容易创新。

*   **庞大的社区**

Python 的惊人崛起的一个主要原因要归功于它的生态系统。例如，随着 Python 将其触角延伸到数据科学社区，越来越多的志愿者正在创建数据科学库。这反过来又引领了用 Python 创建最现代的工具和处理的方式。

*   **用 C 或 C++扩展 Python 很容易**

如果你知道如何用 C 编程，给 Python 添加新的内置模块是相当容易的。这样的扩展模块可以做两件在 Python 中无法直接完成的事情:它们可以实现新的内置对象类型，它们可以调用 C 库函数和系统调用。

通过这种方式，对于执行速度至关重要的任务可以用 C 编写，并在 python 中作为内置模块公开，在 python 程序中调用，就好像它是一个纯 python 模块一样。

事实上，数学、科学计算、数据科学和其他领域的几个 python 库——需要速度性能——都是用 C/C++编写的，并作为一个模块向 python 公开。

出于所有这些原因，并且可能比我在这里列出的更多，Python 被财富 500 强公司和世界顶级大学广泛采用，并且在数据科学界也是一种非常流行和广泛使用的语言，这是本文的主要受众。

# Python 的版本

在本文发表时，有两个流行的 Python 编程语言版本在使用，分别是 Python 2 和 Python 3。

对 python 2 的支持于 2020 年 1 月 1 日结束。Python Foundation 想要传达的[信息是，开发者应该尽快过渡到 Python 3，不要再等了:](https://www.python.org/doc/sunset-python-2/)

> *“我们已经决定，2020 年 1 月 1 日，将是我们日落 Python 2 的日子。也就是说，从那天以后，即使有人发现它有安全问题，我们也不会再改进它了。你应该尽快升级到 Python 3。”*

Python 3 于 2008 年底发布，从一开始就意味着脱离过去，因为[这是修复影响 Python 2](http://www.artima.com/weblogs/viewpost.jsp?thread=208549) 的许多缺陷并推动语言发展的唯一途径。

# 朱庇特笔记本

我们将使用 Jupyter Notebook，一个交互式环境来跟踪课程，而不是在您的本地环境中安装和管理 python。

Jupyter 是一个开源的 web 应用程序，允许你创建和共享包含实时代码、等式、可视化和叙述性文本的文档。笔记本文档是包含分析描述和结果(图、表等)的人类可读文档..)以及可以运行以执行数据分析的可执行文档。

Jupyter 支持 40 多种编程语言，一次一种(它不允许在同一个文档上有多个运行时)。你可以在 https://jupyter.org/[官网](https://jupyter.org/)了解更多

本文中提供的课程集是用 Jupyter 编写的，旨在通过 Google collaboratory(Colab)在云上运行。

# Google Colab

合作实验室是谷歌的一个研究项目，旨在帮助传播机器学习教育和研究。这是一个 Jupyter 笔记本环境，不需要任何设置，完全在云中运行。它的一个主要优势是提供了对 GPU 运行时的支持，这对机器学习很有帮助。

它的另一个最有趣的功能是 Github 集成，允许从 Github 公共存储库加载笔记本和将笔记本保存到 GitHub。这也是我使用它来部署笔记本电脑的主要原因。每个打开的笔记本都会从公共的 Github 存储库创建一个副本到你的 google drive 账户，允许你修改、交互和保存你的笔记本版本。

# 笔记本电脑

Jupyter 笔记本集旨在提供 Python 编程语言的介绍。

虽然这个集合是针对初学数据科学的学生，但我发现它对任何初学 python 编程的人都非常有用。

所有笔记本都是由 [IBM 认知类](https://cognitiveclass.ai/)开发发布的，有一些改动、代码更新和其他定制都是我做的。

笔记本按以下主题划分，每个主题包含一节课，预计完成该课需要的时间。

## Python 基础

本节涵盖 python 基础知识:打印、导入、类型、表达式和字符串。

*   你的第一个节目 — 10 分钟
*   [类型](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/1-basics/PY0101EN-1-2-Types.ipynb) — 10 分钟
*   [表情和变量](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/1-basics/PY0101EN-1-3-Expressions.ipynb) — 10 分钟
*   [串操作](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/1-basics/PY0101EN-1-4-Strings.ipynb) — 15 分钟

**预计所需总时间** : 45 分钟

## Python 数据结构

本节涵盖了主要的 Python 数据结构。

*   [元组](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/2-data-structures/PY0101EN-2-1-Tuples.ipynb) — 15 分钟
*   [列表](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/2-data-structures/PY0101EN-2-2-Lists.ipynb) — 15 分钟
*   [字典](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/2-data-structures/PY0101EN-2-3-Dictionaries.ipynb) — 20 分钟
*   [设定](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/2-data-structures/PY0101EN-2-4-Sets.ipynb) — 20 分钟

**预计所需总时间** : 75 分钟

## Python 编程基础

本节涵盖 Python 语言、逻辑和控制结构、函数以及 Python 中面向对象编程的基础知识。

*   [条件和分支](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/3-fundamentals/PY0101EN-3-1-Conditions.ipynb) — 15 分钟
*   [循环](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/3-fundamentals/PY0101EN-3-2-Loops.ipynb) — 20 分钟
*   [功能](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/3-fundamentals/PY0101EN-3-3-Functions.ipynb) — 40 分钟
*   [类别和对象](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/3-fundamentals/PY0101EN-3-4-Classes.ipynb) — 40 分钟

**预计总时间** : 120 分钟

## 文件

本节涵盖了 Python 中文件处理的基础知识。

*   [打开读取文件](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/4-files/PY0101EN-4-1-ReadFile.ipynb) — 40 分钟
*   [打开](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/4-files/PY0101EN-4-2-WriteFile.ipynb)写文件— 15 分钟

**预计所需总时间**:55 分钟

## Python 数据分析库(Pandas)

本节介绍了 [pandas](https://pandas.pydata.org/) ，这是一个开源库，为 Python 编程语言提供了高性能、易于使用的数据结构和数据分析工具。

*   [熊猫蟒蛇介绍](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/5-pandas/PY0101EN-5-1-LoadData.ipynb) — 15 分钟

## NumPy

本节介绍了使用 Python 进行科学计算的基础包 [NumPy](https://numpy.org/) 。

NumPy 使得在数据科学中经常执行的许多操作变得更加容易。与常规 Python 相比，NumPy 中同样的操作通常计算速度更快，需要的内存更少。

*   [蟒蛇皮 1D NumPy](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/6-numpy/PY0101EN-6-1-Numpy1D.ipynb)—30 分钟
*   [蟒蛇皮 2D NumPy](https://colab.research.google.com/github/computationalcore/introduction-to-python/blob/master/notebooks/6-numpy/PY0101EN-6-2-Numpy2D.ipynb)—20 分钟

**预计所需总时间** : 50 分钟

我希望这些资源可以帮助你成为一名更好的 python 程序员。

# 参考

*   [笔记本源代码](https://github.com/computationalcore/introduction-to-python)