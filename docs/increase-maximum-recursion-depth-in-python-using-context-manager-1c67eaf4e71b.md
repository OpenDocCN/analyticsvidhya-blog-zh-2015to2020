# 使用上下文管理器增加 Python 中的最大递归深度

> 原文：<https://medium.com/analytics-vidhya/increase-maximum-recursion-depth-in-python-using-context-manager-1c67eaf4e71b?source=collection_archive---------36----------------------->

![](img/cb5b2646a697a75bbe35dc62dbb65d28.png)

由[马库斯·斯皮斯克](https://unsplash.com/@markusspiske?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/think-twice?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

上下文管理器是一个对象，它定义了在执行带有语句的**时要建立的运行时上下文。上下文管理器处理代码块执行所需的运行时上下文的进入和退出。**

让我们看看下面的例子，假设我想计算一个数的斐波纳契数-

```
def fib_cal(fib_num, memo):
    if memo[fib_num] is not None:
        return memo[fib_num]
    elif fib_num == 1 or fib_num == 2:
        result = 1
    else:
        result = fib_cal(fib_num-1, memo) + fib_cal(fib_num-2, memo)
    memo[fib_num] = result
    return result def get_fibonacci(fib_num):
    memo = [None] * (fib_num+1)
    return fib_cal(fib_num, memo)print(fibonacci(100))
```

***输出-***354224848179261915075

*   假设我的数字更大，比如 3000

```
print(fibonacci(3000))
```

*   在上面的例子中，它会抛出一个错误-

```
RecursionError: maximum recursion depth exceeded in comparison
```

因为已经超过了操作系统的最大递归限制。您可以按如下方式检查递归限制

```
import sys
sys.getrecursionlimit()
```

因此，在这种情况下，我们可以使用上下文管理器，它允许我们在需要时精确地分配和释放资源。

```
import sys class RecursionLimit:
    def __init__(self, limit):
        self.limit = limit
        self.cur_limit = sys.getrecursionlimit() def __enter__(self):
        sys.setrecursionlimit(self.limit) def __exit__(self, exc_type, exc_value, exc_traceback):
        sys.setrecursionlimit(self.cur_limit) MAX_LIMIT = 10000with RecursionLimit(MAX_LIMIT):
    print(fibonacci(3000))
```

**输出**-4106158863079712603335683787192671052201251086373692524……..6000

`__enter__()`返回需要管理的资源，而`__exit__()`不返回任何东西，而是执行清理操作。

上下文管理器还可以用于其他目的，如简单的文件 I/O，如打开和关闭套接字，在测试期间实现设置和拆除功能。

我希望你喜欢读这篇文章，如果你有什么建议，请在评论中告诉我。