# Python 中可变的默认参数 Mess

> 原文：<https://medium.com/analytics-vidhya/the-mutable-default-argument-mess-in-python-c0692c3032de?source=collection_archive---------32----------------------->

![](img/aa820996364ba03cfd800d177a1fb84c.png)

见见凯文，🙋‍♂️·凯文正在学习 Python。一天，他被要求解决如下问题:

> 设计一个函数，将' # '追加到作为参数提供的列表中，然后打印出来。如果没有提供参数，那么函数应该使用一个空列表作为缺省值。

凯文很快想出了以下解决方案:

```
def append(l = []):
    l.append('#')
    print(l)
```

看起来不错，该测试解决方案了:

```
append([1, 2, 3])
# OUTPUT: [1, 2, 3, '#'] | OKappend()
# OUTPUT: ['#']    | OKappend()
# OUTPUT: ['#', '#']  | Strange!!
```

*   对`append`的第一次调用运行良好。它将#添加到列表`[1, 2, 3]`中，然后打印它。
*   第二个也像预期的那样起作用。这一次没有提供列表作为参数，所以它使用默认的空列表并附加了一个#号。
*   现在，第三个电话导致了意想不到的事情。
*   当我们再次无争论地调用`append`时，它打印出`['#', '#']`而不是我们在上面的调用中得到的`['#']`。

# 为什么会这样？

发生这种情况的原因是 Python 只在第一次定义函数时定义了一次默认参数*。*

这是因为 python 是逐行解析的，当解析器遇到`def`时，它会将默认参数设置为一个值，该值将在以后的每次调用中使用。

当默认参数是**可变的**时，Python 的这种行为就变得特别值得关注。

由于 **immutables** 的值不可更改，如果您更新函数中的自变量变量，它将创建一个新的对象并开始指向该对象，而不是更改原来的默认对象。

但是在可变默认参数的情况下，解析函数时创建的对象被更新，而不是为该函数调用创建不同的对象。

# 解决办法

这个问题的解决方案是使用不可变的默认参数，而不是可变的。首选是`None`(尽管您可以选择任何不可变的值)。

```
def append(l = None):
    if l is None:
        l = []
    l.append('#')
    print(l)
```

让我们测试这个解决方案-

```
append([1, 2, 3])
# OUTPUT: [1, 2, 3, '#']    | OKappend()
# OUTPUT: ['#']     | OKappend()
# OUTPUT: ['#']        | Works fine!
```

太好了！这个解决方案如预期的那样有效。

但是为什么呢？让我们看看里面。

# 为什么该解决方案有效？

观看此视频，了解错误版本的代码中发生了什么-

解决方案第一版的工作

如您所见，在这种情况下，原始的`l`被修改，而不是为每个函数调用创建一个新的`l`，因为`list`是一个可变值。

现在，看看代码的正确版本-

解决方案第二版的工作

这里因为`None`是一个不可变的值，所以它不能被改变，并且为每个函数调用创建一个新的列表对象。

> 注意:默认参数是函数对象的一个属性，因此，最初它对所有函数调用都是一样的。

# 感谢阅读😊

如果你觉得这篇文章有帮助，请点赞并分享！欢迎在评论中反馈意见。

## 您可能还喜欢:

*   [现代 C++特性](https://blog.yuvv.xyz/modern-cpp-features)
*   [当你运行一个计算机程序时会发生什么？](https://blog.yuvv.xyz/what-happens-when-you-run-a-computer-program)
*   [Python 中的理解:解释](https://blog.yuvv.xyz/comprehensions-in-python-explained)
*   [Linux 命令参考示例](https://blog.yuvv.xyz/linux-commands-reference-with-examples)

这篇文章最初发表在 [Yuvraj 的 CS 博客](https://blog.yuvv.xyz/the-mutable-default-argument-mess-in-python)

在 [Twitter](https://twitter.com/yuvraajsj18) 、 [GitHub](https://github.com/yuvraajsj18) 和 [LinkedIn](https://www.linkedin.com/in/yuvraajsj18/) 上与我联系。