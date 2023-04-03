# 在 Python itertools 模块中漫游—第一部分

> 原文：<https://medium.com/analytics-vidhya/wandering-in-the-python-itertools-module-part-i-278b449d0f29?source=collection_archive---------27----------------------->

迭代是编程中的关键概念之一。最直接的方法显然是使用 for 或 while 循环；然而，当可迭代变量变大时，我们需要更有效的方法。好消息是 Python 的标准库中已经有了它，在名为`itertools`的模块中。

`itertools`是一个很棒的模块，它包含一组内存优化的迭代器构建块(你可以在 [Python itertools 文档](https://docs.python.org/3.8/library/itertools.html)中了解更多)。该模块包含许多有用的函数，您可能已经尝试在代码中的某个地方实现了这些函数；例如，这个名为`product`的函数相当于一个嵌套的 for 循环。我们将在这个由三部分组成的系列文章中讨论这个问题。

我决定写一个由三部分组成的系列有两个原因:
- `itertools`模块有三种类型的迭代器，因此每篇文章将涵盖其中一种。我希望每篇文章都易于阅读，而不是让读者不知所措。

最后，我想解释一下(因为你可能已经在问了，已经有相关的文档了，为什么我们需要一篇文章)为什么我决定写这篇文章。我编码已经快四年了，开始时阅读 Python 的文档是非常困难的(至少对我来说)。现在我更加精通编码，我认为它写得非常好。因此，在这篇文章中，我的目标是让那些在当前编码水平下觉得太复杂的人更容易过渡到 Python 文档(不管怎样，这是我的第一篇博客文章)。但我也想强调，理解和参考文档是最重要的技能之一，所以我强烈建议新程序员花时间阅读 Python 的文档。

# 先决条件

我使用了 Python 3.8 文档，所以唯一的先决条件是拥有 Python 3.8 和 Python 的基础知识。

我还使用了 [Google Colab](https://colab.research.google.com) ，在我看来这是一个很棒的工具，但是欢迎你使用任何你喜欢的平台，因为我们只需要 Python 标准库中的一个模块。

# 导入 itertools

我们要做的第一件事是从 Python 标准库中导入 itertools 模块，如下所示:

```
import itertools as it
```

通过这种方式，我们可以通过简单的`it.<function_name>`来访问 itertools 函数。

现在我们已经解决了这个问题，让我们开始吧！

# Itertools —无限迭代器

我们将介绍的 itertools 模块的第一部分叫做*无限迭代器*，它有三个功能:

# 数数

count 函数有两个参数:

```
it.count(start=0, step=1)
```

它基本上是一个无限迭代器，通过增加*步骤*的值(默认为 1)，从*开始*的值(默认为 0)开始创建均匀间隔的数字。

**注意:**我只想声明，使用 count 时要格外小心，因为你很容易陷入无限循环。

我们将看到的一些例子包括我们可能还没有涉及到的函数，但是释放`itertools`威力的最好方式是当你一起使用它们来生成更复杂的迭代器时。因此，让我们来看一些如何使用*计数*的例子:

## 生成数字序列

`count`最基本的功能是生成数字，从 Python 3.1 开始，`step`参数允许非整数值。

比如说；让我们写一个函数，它将在我们想要的范围内产生一个数的倍数`n`。让我们编写一个不使用 itertools 的函数和一个使用 itertools 的函数，然后比较两个函数的运行时:

```
# without using itertools
# desired_range is a list [start, stop]
def generate_without_itertools(n, desired_range): 
    return [i 
            for i in range(desired_range[0], desired_range[1]) 
            if i % n == 0] # with using itertools 
def generate_with_itertools(n, stop): 
    return it.takewhile(lambda x: x <= stop, it.count(step=n))
```

现在让我们比较一下两个函数在[0，100000]之间生成 8 的倍数需要多长时间:

```
>> %%timeit 
   generate_without_itertools(8, [0,100000]) 100 loops, best of 3: 6.54 ms per loop >> %%timeit 
   list(generate_with_itertools(8, 100000)) 1000 loops, best of 3: 1.55 ms per loop
```

函数使用`itertools`是 x4 倍快，这是一个很大的差异，在我看来，加上`itertools`有优势。注意`generate_with_itertools`函数被转换为`list`。这是因为当我运行`generate_with_itertools`时，它不会立即返回任何东西，这被称为[惰性求值](https://en.wikipedia.org/wiki/Lazy_evaluation)。当我要的时候，它会给我下一个号码。这种方法可以大大减少函数的运行时间，并且消耗更少的内存。

## 模拟枚举

假设您有一个字符串`s='python'`，那么您可以使用 *count* 来模拟枚举的行为，如下所示:

```
>> s = "python" 
>> list(zip(it.count(), s)) [(0, 'p'), (1, 'y'), (2, 't'), (3, 'h'), (4, 'o'), (5, 'n')]
```

这个例子的好处是，在不知道输入字符串长度的情况下，我们枚举了它。

这是一个简单的例子，也可以通过`list(enumerate(s))`来实现，但是根据我的经验，知道不同的做事方法是很好的，因为你永远不知道什么时候它会派上用场。

# 循环

`cycle`函数将任何 iterable 作为参数，并无限重复内容。下面是几个例子:

```
>> s = "python is great" 
>> c = it.cycle(s) # this will print the letters one by one indefinitely
>> [print(letter) for letter in c]
```

如果我把`s`放在一个列表中，比如`["python is great"]`，它将循环列表中的元素:

```
>> s = ["python is great"] 
>> c = cycle(s) 
>> i = 0 
   while i < 5: # cycle it 5 times
      # next(c) asks for the next element from the cycle object 
      print(next(c)) 
      i += 1 python is great 
python is great 
python is great 
python is great 
python is great
```

在玩循环函数的时候，我发现了一个有趣的事情，如果你把一个字典作为一个 iterable 传入，它会循环这些键:

```
>> d = {"dog": 5, "cat": 3} 
>> d_cycle = it.cycle(d) 
>> i = 0 
   while i < 5: 
       next_key = next(d_cycle) 
       print(f"{next_key}: {d[next_key]}") 
       i += 1 
dog: 5 
cat: 3 
dog: 5 
cat: 3 
dog: 5
```

# 重复

顾名思义，这个函数无限重复一个对象，除非你指定可选的`times`参数。但它仍然不会开始重复，直到我们要求它:

```
# this returns repeat object 
>> repeat(10) 
repeat(10) # this will actually call the numbers from repeat and exhaust the generator 
>> [i for i in repeat(10,3)] [10, 10, 10]
```

让我们为幂[1，10]找到 8 的幂:

```
>> list(map(pow, range(10), repeat(8))) 
[0, 1, 256, 6561, 65536, 390625, 1679616, 5764801, 16777216, 43046721]
```

# 结论

因此，在本文中，我们介绍了 Python 标准库中`itertools`模块的无限迭代器，它们是:

*   数数
*   循环
*   重复

本文中的例子很简单，只是为了演示的目的，然而`itertools`函数在相互结合构建更复杂的迭代器时运行得最好。`itertools` module 绝对是一个重要的 Python 模块，应该学习并保存在您的工具集中。

**附言**这是我的第一篇博文，所以我感谢任何**建设性的**反馈！感谢阅读！