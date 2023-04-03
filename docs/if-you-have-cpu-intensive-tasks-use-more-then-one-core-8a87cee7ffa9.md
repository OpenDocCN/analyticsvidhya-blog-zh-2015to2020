# 如果您有 CPU 密集型任务，请考虑使用多个内核

> 原文：<https://medium.com/analytics-vidhya/if-you-have-cpu-intensive-tasks-use-more-then-one-core-8a87cee7ffa9?source=collection_archive---------5----------------------->

# 快速 Python vs C insight

![](img/386ebd43e86f0b679721e438831e7a9d.png)

马修·施瓦茨在 [Unsplash](https://unsplash.com/s/photos/horse-race?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

ython 非常漂亮，它的 acyncio 是 io 密集型任务的优秀加速器。当你做 web 应用程序时，大多数情况下你应该感到满意，但并不是所有的问题都是如此。当您不得不进行大规模计算时，知道如何划分任务并由多个可用内核同时完成任务是很重要的，并行**。你的笔记本电脑可能有 8 个。也许更多。**

**我最近面临着这样的挑战，所以这是一个很好的时间来回忆并行性和快速研究如何管理它。因为我对我选择的语言 Python 所带来的加速效果不满意，所以我也决定回忆一下什么是 C 语言。是的，是的，25 年前我会说 C 是美丽的。我喜欢我在大学的作业，在 Linux box 上编程，内核 0.98。**

**解决手头问题的不同方法的时间结果。**

**这是我最初任务的结果。由于给出的数字，我写的是*使用多个内核*而不是*使用所有内核。*正如你所看到的，仅仅用 C 替换 Python 就给了我们惊人的进步。Python 最好的`multiprocessing`比最坏的 C 情况慢得多。不幸的是。**

**如果您立即担心 8 个内核无法以 8 倍的速度完成工作，或者有时 4 个内核比 8 个内核做得更好，请阅读这些堆栈上的解释:[一个](https://stackoverflow.com/questions/59138942/why-doesnt-multiprocessing-pool-increase-computational-speed-linearly)和[两个](https://stackoverflow.com/questions/50221512/python-multiprocessing-performance-only-improves-with-the-square-root-of-the-num)。**

# **拼命**简化任务，只为演示****

**假设我们有一个由 n 个元素组成的数组，我们必须将它们相加。对于纯 Python 来说，这是小学第一节编程课的初学者任务。使用电子表格非常简单，因此它相当于 Python 的`numpy`解决方案:**

**有了`numpy`真的超级简单。**

**但是如果数组不仅仅只有 19 个元素，比如上面的例子，而是几十亿个元素，那会怎么样呢？Excel 加载时肯定会冻结。Numpy 将显示它无法为形状为 y 的数组分配 X GiB。**

> **我们所能做的，是把我们的问题划分成或多或少相等的子问题(子数组)给同等的工作者(cpu 核心)。**

**当然，有很多可能性。在接下来的两节中，我将展示如何用 Python 和 C 来实现这一点，参考这个最简单的例子并讨论两个相当简单的代码片段，它们不会产生太多的输出。如果你感兴趣，请从 [GitHub 库](https://github.com/rganowski/multitaskig-trials)获取更详细的程序版本。**

# **Python 解决方案:多重处理**

**我从 Python 开始，因为我的整个应用程序都是 Python 的。使用`multiprocessing`标准库可以利用多个内核。请看看清单:**

*   **(3)要处理的数据与之前由`numpy.sum()`解决的问题相同。**
*   **(6–10)整个数组(`subtotals`)的子区域总数的计算将由`calc_subtotal`函数进行，该函数需要三个参数:`DATA`数组的上下边界和返回计算结果的特殊`multiprocessing.Queue`对象。**
*   **(13)我将吸收我所有的核心，`multiprocessing.cpu_count()`告诉我我还有多少可用的核心。**
*   **(17–26)我必须计算最好是绝对公平的数据分布。我在这里解决了几乎相同的任务，C 示例中的`OpenMP`为我自动完成。这里，所有的并行进程应该在至少 2 个元素上工作，但是其中 3 个必须考虑一个额外的元素(8 核的 19 元素阵列:5 核应该得到 2 个元素，3 核应该得到 3 个元素)。**
*   **(28–34)然后，做好一切准备后，我生成所有进程，并保留它们的引用，以便以后与主流程合并…**
*   **(36–37)…所以它是在这里完成的，通过在每个记忆的进程上调用的`join()`方法。事实上，这是一个非常重要的手术。因此，主流程知道它必须等待更早产生的进程完成。如果它们没有被加入(有人想说等待，但当然不是这样)，来自`subtotals`的`total`的最终计算将在仍未完成的结果的基础上进行。在上面的循环中，将`join()`放在`start()`之后，会阻止程序产生下一个进程，直到刚刚启动的进程完成，实际上是八个操作系统进程的顺序处理。因为并行解决方案的原因。那将是我能做的最愚蠢的事情。**

**该表显示了刚刚讨论的程序的[详细版本的计算结果。8 个过程应该并行工作，但是它们以严格的顺序出现。如果数组更大，你最终会看到交错的动作。在这样一个小例子中，Python `multiprocessing`产生新流程的成本显然比特定小计的计算成本高得多。](https://github.com/rganowski/multitaskig-trials/blob/main/utilize-all-cores-verbose.py)**

# **c 解决方案:OpenMP**

**我们可以用`OpenMP`(开放多处理 API)用 C 语言解决同样的任务。我在下面展示的程序似乎出乎意料地简洁，尤其是与 Python 版本相比。只需放下`omp.h` include 指令，以及所有的 pragma……你就会得到第一堂课的简单 C 源代码。嗯，我觉得印象很深刻！**

**要运行代码，首先必须编译源代码，使用一个特殊选项来处理`OpenMP`指令:**

```
% gcc **-fopenmp** utilize-all-cores.c -o utilize-all-cores
```

*   **(6)要处理的数据与已经解决了两次的问题中的数据完全相同，由`numpy.sum()`和 python 的`multiprocessing`处理。**
*   **(11–20)因此，技巧在于正确使用`#pragma omp …`。第一个表示程序的并行部分，指定`subtotal`变量为每个产生的并行进程的私有变量。因为变量`total`不在私有列表中，默认情况下它将被共享。为了更明确地做到这一点，我可以在指令中添加`shared (total)`子句。**
*   **(14–16)神奇的是！这个`for`循环由并行进程以一种公平的方式执行，这是我自己在 Python 版本中准备的，这个决定是在程序启动时做出的，取决于您拥有的内核数量(默认情况下所有内核都投入工作)或您的请求，因为您可以像这样调用程序:**

```
% **OMP_NUM_THREADS=4** ./utilize-all-cores
```

*   **(18–19)最后一个`#pragma`强制对共享变量进行顺序访问。在这里，它和 Python 版本中的`join()`一样重要。**

**该表显示了由刚刚讨论的程序的[详细版本提供的计算结果。8 个进程并行工作，现在，即使在如此小的数据样本中，您也可以看到它们交错。](https://github.com/rganowski/multitaskig-trials/blob/main/utilize-all-cores-verbose.c)**

# **摘要**

**![](img/6c5c751bd5b01d9e974b280339307927.png)**

**照片由 [Gene Devine](https://unsplash.com/@devine_images?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/horse-race?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄**

**如今，互联网上有大量关于使用并发解决方案的指南，但是并发并不等同于并行。各有各的用处。Async/Await 结构在 CPU 密集型任务中没有帮助，因为它们在 IO 密集型任务中很有用。**

**Python 的`multiprocessing`可以做并行任务，可惜不是它的强项。是啊！我知道，我应该试试 Go 语言…哦，我的天哪，又是那些花括号。但是，肯·汤普森…好吧，也许其他人会告诉我如何面对它，勾勒出这个刚刚讨论过的简化的问题。**