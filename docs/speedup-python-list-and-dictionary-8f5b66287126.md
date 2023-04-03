# 加速 Python 列表和字典

> 原文：<https://medium.com/analytics-vidhya/speedup-python-list-and-dictionary-8f5b66287126?source=collection_archive---------37----------------------->

![](img/d4bb42659aa1548e7fc9a78347bbe791.png)

照片由[丘特尔斯纳普](https://unsplash.com/@chuttersnap?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/speed?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

今天，我们将讨论 Python 中的优化技术。在本文中，您将了解如何通过避免列表和字典中的重求值来加速代码。

这里我编写了装饰函数来计算函数的执行时间。

```
import functools
import timedef timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function - {}, took {} ms to complete'.format(func.__name__, int(elapsedTime * 1000)))
    return newfunc
```

让我们转向实际的功能

# 避免列表中的重新评估

**在循环**内评估 `**nums.append**`

```
@timeit
def append_inside_loop(limit):
    nums = []
    for num in limit:
        nums.append(num) append_inside_loop(list(range(1, 9999999)))
```

**在上面的函数`nums.append`中，函数引用在每次循环中都被重新评估。执行后，上述函数所用的总时间**

```
o/p - function - append_inside_loop, took 529 ms to complete
```

****评估** `**nums.append**` **外循环****

```
@timeit
def append_outside_loop(limit):
    nums = []
    append = nums.append
    for num in limit:
        append(num) append_outside_loop(list(range(1, 9999999)))
```

**在上面的函数中，我在循环外对`nums.append`求值，并在循环内使用`append`作为变量。以上函数占用了总时间**

```
o/p - function - append_outside_loop, took 328 ms to complete
```

**正如你所看到的，当我将`for`循环外的`append = nums.append`作为一个局部变量进行求值时，它花费了更少的时间，并通过`201 ms`加速了代码。**

**同样的技术我们也可以应用于字典的情况，请看下面的例子**

# **避免在字典中重新评估**

****评估** `**data.get**` **循环内的每次****

```
@timeit
def inside_evaluation(limit):
    data = {}
    for num in limit:
        data[num] = data.get(num, 0) + 1 inside_evaluation(list(range(1, 9999999)))
```

**以上函数占用的总时间-**

```
o/p - function - inside_evaluation, took 1400 ms to complete
```

****评估** `**data.get**` **循环外****

```
@timeit
def outside_evaluation(limit):
    data = {}
    get = data.get
    for num in limit:
        data[num] = get(num, 0) + 1 outside_evaluation(list(range(1, 9999999)))
```

**以上函数占用的总时间-**

```
o/p - function - outside_evaluation, took 1189 ms to complete
```

**如你所见，我们通过`211 ms`加速了代码。**

**我希望你喜欢 Python 中对列表和字典的优化技术的解释。不过，如果有任何疑问或改进，请在评论区提问。另外，不要忘记分享你的优化技巧。**