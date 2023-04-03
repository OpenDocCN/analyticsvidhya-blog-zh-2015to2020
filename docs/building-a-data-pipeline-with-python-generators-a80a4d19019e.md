# 用 Python 生成器构建数据管道

> 原文：<https://medium.com/analytics-vidhya/building-a-data-pipeline-with-python-generators-a80a4d19019e?source=collection_archive---------3----------------------->

在这篇文章中，你将了解到我们如何使用 Python 的生成器特性来创建数据流管道。对于生产级管道，我们可能会使用合适的框架，比如 Apache Beam，但是这个特性是构建 Apache Beam 的定制组件所必需的。

# 问题是

让我们看看对数据管道的以下需求:

> 写一个小框架，处理一个无穷无尽的整数事件流。假设处理将是静态和顺序的，其中每个处理单元将输出传递到下一个处理单元，除非有不同的定义(例如，过滤器、固定事件窗口)。该框架应包括以下 6 个“构件”:
> 
> 1.stdin-source:从 stdin 中读取一个数字，打印'> '和后面的数字。例如，如果用户输入 1，它将打印" > 1"
> 2。filter:只传递与谓词匹配的事件(给定一个数字，返回 true 或 false 的函数)。谓词是在过滤器初始化期间给出的。
> 3。fixed-event-window:将事件聚集到一个固定大小的数组中，当数组满时将其向前传递。固定数组的大小是在固定事件窗口初始化期间定义的。
> 4。fold-sum:对数组中事件的值求和，并将和向前传递。
> 5。fold-median:计算数组中事件的中值，并将中值向前传递。
> 6。stdout-sink:将数字打印到 stdout 并向前传递。
> 
> 从这 6 个构建块中，可以构建任何管道，这里有一个例子:
> stdin-source->filter(I =>I>0)->固定事件窗口(2) - > fold-sum - >固定事件窗口(3)->fold-median->stdout-sink

你将如何解决这样的问题？我首先尝试的是为每一个“构建模块”编写一个函数，然后在用户输入的无限循环中运行每一个函数，并按照管道所需的顺序重新组织这些函数。

这种方法有两个明显的问题:

1.如果谓词返回 *false* ，则 *filter* 函数必须返回 null，这要么迫使其他函数优雅地处理 null，要么迫使管道自己处理 null——这意味着管道或其他函数必须知道 *filter* 函数的实现——如果我们改变管道中 filter 函数的顺序，我们也必须移动相关的 null 检查。

2.*固定事件窗口*函数必须每 X 次批量输入元素并输出它们——这意味着函数必须在多次执行中**保持其状态**。这要求管道不仅要知道 *filter* 函数的实现，而且它实际上必须是实现的一部分(使用函数范围之外的变量来处理批处理)。

示例中编写管道的代码很容易编写，但是要以一种允许我们随心所欲地摆弄构建块的方式编写它(将它编写为一个实际的管道，而不仅仅是以特定方式放在一起的一堆代码)，这需要我们使用生成器。

# 发电机简介

生成器函数是返回称为生成器(迭代器的子类)的对象的函数。我们通过使用关键字 *yield* 而不是关键字 *return* 来创建一个生成器函数。让我们先创建一个:

下面是一个返回 0-4 整数的生成器示例:

```
>>> def gen(): 
…     for i in range(5): 
…       yield i 
```

但是这里发生了什么？与 *return* 关键字不同，该函数在第一个 *yield* 语句之后并没有结束。只有当没有更多的值要产生时，函数才会结束执行——这里是指循环结束的时候。

让我们运行这个函数，看看会发生什么:

```
>>> gen() 
<generator object gen at 0x7f58682b7820>
```

我说过一个生成器函数返回一个生成器对象，我也说过它是一个迭代器，所以如果我们想计算结果，我们必须这样做:

```
# returns a generator object, the function's code isn't executed yet
>>> res = gen()# here the code inside the generator is actually being run
>>> for j in res:
...   print(j) 
0 
1 
2 
3 
4
```

关键字的*隐式调用迭代器上的*下一个*函数，直到它的值用完为止(我们也可以只使用对象上的*列表*函数——这将隐式调用所有元素上的*下一个*)。*

如果你仍然好奇发电机到底是如何在幕后工作的(你应该好奇，这很有趣)，这里[有一篇关于这个主题的非常酷的博客文章。](https://hackernoon.com/the-magic-behind-python-generator-functions-bc8eeea54220)

# 建设管道

现在我们已经非常了解什么是生成器以及我们如何创建它们，让我们回到解决手头的问题。任务是编写一个流框架，其中每个构建块对当前值进行一些处理，并在需要时向前传递，同时仍然处理更多的值。对于固定事件窗口步骤，我们需要在多次执行中保持函数的状态——这正是生成器能为我们做的。

让我们从框架的构建模块开始:

1.  从用户处读取整数输入流，并将其向前传递:

```
def stdin_source():
  def try_to_int(val):
    try:
      return int(val)
    except ValueError:
      return None for input in sys.stdin:
    if input.strip() == 'exit':
      exit() val = try_to_int(input)
    if val is not None:
      print(‘> %d’ % val)
      yield val
```

2.向前传递与谓词匹配的值:

```
def filter_numbers(numbers, predicate):
  for val in numbers:
    if predicate(val):
      yield val
```

3.将用户输入批处理到在本地定义的数组**(数组的状态在执行中被保留，因为它是一个生成器)，为每个 *batch_size* 元素传递一个固定大小的列表:**

```
def fixed_event_window(numbers, batch_size):
  arr = []
  for val in numbers:
    arr.append(val) if len(arr) == batch_size:
      res = arr.copy()
      arr = []
      yield res
```

**4.对数组元素求和:**

```
def fold_sum(arrs):
  for arr in arrs:
    yield sum(arr)
```

**5.计算数组元素的中值:**

```
# 5\. fold-median
def fold_median(arrs):
  for arr in arrs:
    yield median(arr)
```

**6.打印并向前传递值:**

```
# 6\. stdout-sink
def stdout_sink(numbers):
  for val in numbers:
    print(val)
    yield val
```

**然后，让我们按照要求将所有这些构建模块放在一起:**

```
numbers = stdin_source()
filtered = filter_numbers(numbers, lambda x: x > 0)
windowed_for_sum = fixed_event_window(filtered, 2)
folded_sum = fold_sum(windowed_for_sum)
windowed_for_median = fixed_event_window(folded_sum, 3)
folded_median = fold_median(windowed_for_median)
res = stdout_sink(folded_median)
```

**很漂亮，对吧？这里发生的事情非常清楚，而且也很容易改变函数的顺序，创建一个完全不同的管道，而不必改变管道中的其他任何东西。**

**但是，这条管道实际上不会开始运行——这只是管道的**定义**,如果我们让它保持原样，什么都不会发生——我们创建了一堆生成器对象并将它们放在一起放入管道——但是没有数据被发送到管道，因为没有东西在生成器上迭代，所以用户甚至不会被提示输入。为了运行它，我们只需迭代管道的结果:**

```
....
res = stdout_sink(folded_median)# implicitly calling next() on the generator object until there are # no more values
list(res)
```

**如果我们愿意，我们可以只运行管道的一部分——如果我们选择迭代 *folded_sum* 生成器，那么管道将只运行到那个步骤。**

# **摘要**

**帖子中的所有代码都可以在[这里](https://github.com/SockworkOrange/blog-posts/tree/main/data-pipeline-via-generators)找到——希望你对这个帖子感兴趣，并且了解了一些关于生成器的知识以及它们的用处。在下一篇文章中，我将讨论如何使用这个特性在 Apache Beam 管道中编写定制组件。**