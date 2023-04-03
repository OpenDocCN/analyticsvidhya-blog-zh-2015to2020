# Python decorator 可以并行处理任何 IO 密集型函数

> 原文：<https://medium.com/analytics-vidhya/python-decorator-to-parallelize-any-function-23e5036fb6a?source=collection_archive---------4----------------------->

![](img/d5998ba1e898574cd587aef8f0c13981.png)

如果只需在函数中添加一个装饰器就可以加快程序的速度，这不是很酷吗？如果您不必担心以并行方式运行列表中的数据，这不是很酷吗？

今天我们要写一个 python 装饰器，它会自动为你做这些，这样你就可以更专注于代码的逻辑，而不是担心多线程问题。

在我们开始之前，关于 python 多线程的一些基础知识。

1.  实现它的最佳位置是我们试图并行化的函数是 **IO 繁重的**(睡眠时间对线程来说非常重要)。一些例子是 **API 调用、数据库调用、打开文件、等待数据流、从互联网下载文件**。
2.  通常的做法是保持产生的线程的**数量等于系统中可用的 CPU**数量。(**重要**:只是标准，**不是强制**。将解释我们如何能够获得比同等数量的线程更多的线程，并从我们的系统中提取更多的线程)

好的，我们将从装饰者的代码开始。如果你不能理解或者你不知道也不想理解编写装饰器的逻辑，你可以直接复制粘贴装饰器代码。在这个 [git repo](https://github.com/varungv/MultiThreadingSample) 中有一个示例项目。

下面是我们将要使用的装饰代码。

```
import concurrent.futures
import os
from functools import wraps

def make_parallel(func):
    *"""
        Decorator used to decorate any function which needs to be parallized.
        After the input of the function should be a list in which each element is a instance of input fot the normal function.
        You can also pass in keyword arguements seperatley.* ***:param*** *func: function
            The instance of the function that needs to be parallelized.* ***:return****: function
    """* @wraps(func)
    def wrapper(lst):
        *"""* ***:param*** *lst:
            The inputs of the function in a list.* ***:return****:
        """* # the number of threads that can be max-spawned.
        # If the number of threads are too high, then the overhead of creating the threads will be significant.
        # Here we are choosing the number of CPUs available in the system and then multiplying it with a constant.
        # In my system, i have a total of 8 CPUs so i will be generating a maximum of 16 threads in my system.
        number_of_threads_multiple = 2 # You can change this multiple according to you requirement
        number_of_workers = int(os.cpu_count() * number_of_threads_multiple)
        if len(lst) < number_of_workers:
            # If the length of the list is low, we would only require those many number of threads.
            # Here we are avoiding creating unnecessary threads
            number_of_workers = len(lst)

        if number_of_workers:
            if number_of_workers == 1:
                # If the length of the list that needs to be parallelized is 1, there is no point in
                # parallelizing the function.
                # So we run it serially.
                result = [func(lst[0])]
            else:
                # Core Code, where we are creating max number of threads and running the decorated function in parallel.
                result = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executer:
                    bag = {executer.submit(func, i): i for i in lst}
                    for future in concurrent.futures.as_completed(bag):
                        result.append(future.result())
        else:
            result = []
        return result
    return wrapper
```

我们构建了一个样本虚拟函数，它将对 JSON 占位符 API 进行 HTTPS 调用。下面是示例代码，请注意，这只是为了演示 IO 密集型调用是什么样子，您可以用您想要并行化的任何函数来替换这个函数。

```
import requests
def sample_function(post_id):
    *"""
        Just a sample function which would make dummy API calls
    """* url = f"https://jsonplaceholder.typicode.com/comments?postId={post_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {}
```

当我们尝试对此函数进行串行调用时，我们的代码看起来会像这样:

```
list_of_post_ids = list(range(1, 20))

# Serial way of calling the function
results = []
for post_id in list_of_post_ids:
    res = sample_function(post_id)
    results.append(res)
```

但是，当我们使用装饰器时，代码简化为:

```
# Paralleized way of calling the function
results = make_parallel(sample_function)(list_of_post_ids)
```

您可以观察调用函数的这两种方法之间的时间差，亲自看看多线程如何帮助我们加快 IO 繁重的调用。

如果你不喜欢本文中的代码分割，你可以去我的 [git 库](https://github.com/varungv/MultiThreadingSample)那里，我已经托管了一个完整的演示项目。

另外，注意这个装饰器只对有一个输入参数的函数有效。我将在下一篇文章中改进这个装饰器，并添加基于函数运行时自动选择线程数量的功能。