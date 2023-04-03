# 多处理—让您的计算为您服务

> 原文：<https://medium.com/analytics-vidhya/a-dummys-guide-to-multiprocessing-in-python-1d93c1943b9e?source=collection_archive---------22----------------------->

![](img/21f9ddc6121c66e6651d6736eafd2a20.png)

基特·苏曼在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

*当我开始尝试在 python 代码中使用多核的旅程时，我发现没有个人博客可以帮助我实现目标。我希望这篇博客能够填补我发现的一些空白，让读者能够快速了解嵌入多处理的过程，从而充分利用您的所有计算能力！*

这个博客由几个部分组成，我试图根据对我来说似乎合理的顺序来分解它们。内容细分如下:

*   包装
*   核心
*   存储您的结果
*   功能
*   运行您的功能
*   需要考虑的事项
*   例子

# **套餐**

首先，要创建你自己的多重处理程序，你需要一个特定的包来做这件事。在本文中，我将使用来自 [*多进程*](https://pypi.org/project/multiprocess/) 的*多进程*来构建一个例子

```
import multiprocessing
```

# **内核**

现在，您已经将多处理导入到您的笔记本电脑/环境中，您将能够看到您有多少内核可用。只需使用下面的代码就可以做到这一点。

```
multiprocessing.cpu_count()
```

这将输出一个整数，这是您可以运行多处理的最大内核数。

> 提示:为了确保您总是使用最大计算，我通常在脚本中将它设置为一个输入—processors = multi processing . CPU _ count()

# **存储您的结果**

这可能是从现有文档中理解的最具挑战性的事情，因为不清楚在运行多处理时如何从多核中提取结果。

要提取结果，您需要创建一个特定的字典，您的结果可以传递到这个字典中，如下所示。

```
multiprocessing_dict = multiprocessing.Manager().dict()
```

这将在脚本中将 *multiprocessing_dict_* 设置为一个空的 DictProxy 对象。在本文的后面，您将看到如何存储结果。

# 功能

因此，您可以定制部分来满足您的多处理需求。

要跨内核执行任何任务，您需要创建一个可以被进程调用的函数，您可以像平常一样编写这个函数。然而，要使多重处理能够使用它并存储结果，您需要向函数中添加两个特定的参数。

这些是；

1.  运行它的处理器；
2.  我们在上一步中创建的字典。

然后，不是像通常那样通过*返回*来返回函数输出，而是将它传递给 multi_dict_ argument，用 processor_ 作为它的键。

```
def your_func(your args, processor_, multi_dict_):

    your function 

    multi_dict_[processor_] = your output
```

> 注意:我已经尝试编写了一个通用的函数 shell 来演示函数的参数需求。如果不清楚这一点，在继续之前参考示例代码可能是值得的。

# 运行您的功能

现在我们已经创建了一个函数，我们可以看看如何在之前找到的内核数量上运行这个函数。这里有几个步骤，所以我会试着分解它们。

1.  首先，您需要创建一个空列表，用于存储函数输出。
2.  然后，您需要创建一个关于处理器数量的循环。当您调用多重处理函数时，您可以将您的函数传递给它，并带有相关的参数。然后将它存储在步骤 1 中创建的空列表中。
3.  现在你已经在你的列表中存储了多重处理命令，你可以遍历这个列表并开始多重处理。
4.  然后，您可以将列表中的输出连接在一起。

很多内容在这一点上可能没有意义，所以让我们看看它在代码中是什么样子的，这些步骤被标记为注释。

```
1#
multiprocessing_loop_ = []2#
for processor_ in range(processors):

    multi_process_ = multiprocessing.Process(
        target = your_func, 
        args = (your args, processor_, multiprocessing_dict)
    ) multiprocessing_loop_.append(multi_process_)#3
for process_ in multiprocessing_loop_:
    process_.start()#4
for process_ in multiprocessing_loop_:
    process_.join()
```

> 注意:同样，我试图保持这部分代码的通用性，以演示运行需求。如果不清楚这一点，在继续之前参考一下示例代码可能是值得的。

# 需要考虑的事项

**拆分您的输入—** 在上面的示例中，我们将在您指定的任意数量的内核上，对相同的输入运行相同的功能。

如果您想要运行特定的输入子集，在多重处理之前构建这些步骤是很重要的。

我已经在文章结尾的例子中演示了如何做到这一点。

**理解您的输出—** 在上面的例子中，您的输出将作为一个字典存储在一个列表中。因此，仍然需要将输出提取为可用的格式。

我已经在文章结尾的例子中演示了如何做到这一点。

**Windows 操作系统—** 不幸的是，在 Windows 操作系统上部署多处理并不容易。据我所知，这是由于 fork()的需求。

因此，我不会在本文中尝试解决这个问题。

# 例子

最后，让我们将我们所学的内容合并到一个脚本中。

为此，我创建了一个包含两列的 DataFrame。一个是随机数，另一个是随机整数(1 到 100 之间)。我为这些列中的每一列模拟了 10，000，000 行。

我的函数将我的随机数提升到我的随机整数的幂。

这将作为列“RandomPower”添加到数据帧中。

```
import pandas as pd
import numpy as np
import multiprocessing as mpprocessors = mp.cpu_count()
mp_dict = mp.Manager().dict()
mp_loop = []size = 10000000
input_step = int(size / processors)df = pd.DataFrame(data = {
    'RandomNumber' : np.random.random(size = size),
    'RandomInt' : np.random.randint(0, high = 100, size = size)
})def func(df_, indexlist_, processor_, multi_dict_):
    '''
    This function will raise the RandomNumber in the dataframe by the power of the RandomInt in the dataframe.
    ''' 
    series_ = pd.Series(
        data = df_.loc[indexlist_, 'RandomNumber'].values ** df_.loc[indexlist_, 'RandomInt'].values,
        index = indexlist_
    )multi_dict_[processor_] = series_.to_dict()for processor_ in range(processors):    
    if processor_ == 0:       
        index_ = list(range(0, input_step, 1))
    elif processor_ == processors - 1:
        index_ = list(range(processor_ * input_step, size, 1))
    else:
        index_ = list(range(input_step * processor_, input_step * (processor_ + 1), 1))mp_ = mp.Process(
        target = func,
        args = (df, index_, processor_, mp_dict)
    )    
    mp_loop.append(mp_)for l_ in mp_loop:
    l_.start()

for l_ in mp_loop:
    l_.join()

df_results = pd.DataFrame([])

for key_ in mp_dict.keys():
    df_results = df_results.append(
        pd.DataFrame.from_dict(
            mp_dict[key_], orient = 'index', columns = ['RandomPower']
        )
    )

df = df.merge(
    df_results, 
    how = 'left',
    left_index = True, 
    right_index = True
)
```

> 注意:这个特殊的例子实际上比使用 numpy 在多个内核上运行需要更长的时间。它纯粹是为了演示在多个内核上运行时如何操作和提取函数的结果而创建的。

我希望这篇博客对你有所帮助。

写和发表你的第一篇文章是令人畏惧的，所以任何反馈都将是非常有帮助的——希望能有一些掌声！

此外，如果您注意到我的代码中有任何错误，请让我知道，以便我可以相应地纠正它。