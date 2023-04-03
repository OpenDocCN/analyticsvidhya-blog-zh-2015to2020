# 从输入调用函数

> 原文：<https://medium.com/analytics-vidhya/calling-function-from-input-5f2982ca1df5?source=collection_archive---------40----------------------->

嗨，距离我上次发帖已经很久了。借此机会，我想解释一下如何从用户输入中调用函数。是的，这将是技术性的。

![](img/3151a3337e7f4b02fe87337b37dfed7a.png)

照片由 [Sangga Rima Roman Selia](https://unsplash.com/@sxy_selia?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# 情况

*为了说明的目的，这里使用的情况被简化。*

想象你在一个非常特殊的实验室工作，专注于脱氧核糖核酸或 DNA。除此之外，现在您必须创建一个定制的队列系统。

该系统需要接受以腺嘌呤(符号为 A)、鸟嘌呤(符号为 G)、胞嘧啶(符号为 C)和胸腺嘧啶(符号为 T)形式进入的化学碱基。在接受一个输入时，它有一个**高级算法来判断**该输入是否将形成一个*正常*序列并应该被存储或者是否应该被丢弃。

它还应该能够通过类似出列的机制丢弃存储的化学物质。如果需要，它还应该能够打印当前存储的化学基础。

# 例子

在这篇文章中，我用 Python 来帮助解释。有些部分我们可以改进，但让我们忽略它，专注于主要问题，从输入中调用一个函数。

下面是自定义队列的类的示例:

```
**class CustomQueue:
  def __init__(self):
   ** self.__queued_base = []
    return **def queue(self, list_of_base):** for base in list_of_base:
      if base in ['a', 'g', 'c', 't']:
        self.__queued_base.append(base)
    return **def dequeue(self):** if len(self.__queued_base) == 0:
      return
    letter = self.__queued_base[0]
    self.__queued_base = self.__queued_base[1:]
    return letter **def get_queue(self):
**    return self.__queued_base
```

*   `__init__`将准备用于储存碱基的队列列表。
*   `queue`将一个传入的输入进行排队。为了简单起见，上面提到的*高级算法*将只存储值，如果它是‘a’、‘g’、‘c’或‘t’。
*   `dequeue`将最早出列的输入。
*   `get_queue`会以列表的形式得到所有的化学碱基。

下面是使用上面定义的类的示例:

```
**if __name__ == '__main__':**
  custom_queue = CustomQueue()
  **while True:**
    user_input = input('$ ').split(' ')
    command, param = user_input[0], user_input[1:] ** try:**
      msg = custom_queue.__getattribute__(command)(param)
    **except TypeError:**
      msg = custom_queue.__getattribute__(command)()
    **except Exception as e:**
      msg = e **if msg != None:**
      print(msg)
```

上面的实现将提示用户输入由`command`和`param`(空格分隔)组成的输入。`command`将与`CustomQueue`中的方法相同。

下面将更详细地解释提示给用户的输入。(`$`后的行是供用户发送输入的。)

*   `$ queue f t a g k c`将调用`queue`方法，并将`f t a g k c`作为参数传递。但是由于我们的*“高级算法”*，队列中只会保存`t a g c`。
*   `$ dequeue`将调用`dequeue`方法并忽略任何传递的参数。如果按顺序完成了前一点，它将从列表中取消`t`的队列。
*   `$ get_queue`将调用`get_queue`方法，该方法将返回基地列表。

# 说明

让我们举一个流程的例子:

```
$ queue a f t c g k l
$ get_queue
['a', 't', 'c', 'g']
$ dequeue
a
$ get_queue
['t', 'c', 'g']
```

乍一看，好像没什么问题。但是，在现实中，它有一个安全问题。简而言之，进入上面的流程后，继续下面的流程:

```
$ queue a
$ get_queue
['t', 'c', 'g', 'a']
$ __init__
$ get_queue
**[]**
```

现在，你应该得到输出`[]`。罪魁祸首当然是当我们输入`__init__`作为输入时。为什么会是罪魁祸首？因为输入它，系统会调用方法`__init__`。如果调用这个方法会发生什么？你猜对了，它会重新初始化队列列表。(可以尝试输入`__getattribute__`作为命令，看看结果。)

# 精炼

现在，我们明白允许用户直接从输入中调用函数或方法是危险的。来解决这个问题？很简单。我们只需要限制他们获取信息的方式。

下面是一个如何优化我们代码的例子:

```
if __name__ == '__main__':
  custom_queue = CustomQueue()
  while True:
    user_input = input('$ ').split(' ')
    command, param = user_input[0], user_input[1:] msg = None
    if command == 'queue':
      custom_queue.queue(param)
    elif command == 'dequeue':
      msg = custom_queue.dequeue()
    elif command == 'get_queue':
      msg = custom_queue.get_queue()
    else:
      msg = 'unknown command' if msg != None:
      print(msg)
```

如果我们试图遵循前面的流程:

```
$ queue a f t c g k l
$ get_queue
['a', 't', 'c', 'g']
$ dequeue
a
$ get_queue
['t', 'c', 'g']
```

并输入附加流，我们将得到:

```
$ queue a
$ get_queue
['t', 'c', 'g', 'a']
$ __init__
**unknown command**
$ get_queue
**['t', 'c', 'g', 'a']**
```

上面的例子提供了一种洞察力，通过限制用户可以做什么，我们的系统使用起来更安全。(还是那句话，你可以尝试输入`__getattribute__`作为命令，看看结果。)

# 结论

从输入用户直接调用函数或方法可能是危险的。这将允许用户执行不必要的程序。与其这样做，不如使用`if else`语句或类似的语句来限制用户可以做什么。

# 参考

[https://ghr.nlm.nih.gov/primer/basics/dna](https://ghr.nlm.nih.gov/primer/basics/dna)