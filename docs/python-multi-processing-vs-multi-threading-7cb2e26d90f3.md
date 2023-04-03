# Python 多处理与多线程

> 原文：<https://medium.com/analytics-vidhya/python-multi-processing-vs-multi-threading-7cb2e26d90f3?source=collection_archive---------3----------------------->

我一直在努力加速我的程序，以便更快地执行任务，并且我开始了解 Pythons 的多处理和多线程。起初，他们都希望完成相同的任务，但在现实中，他们确实有相似之处，但我要说，他们更多的是不同于相同。

**多线程-**
考虑一下你去一家只有一个厨师的餐馆。当你点餐时，厨师会开始准备你点的菜，当另一位顾客来时，他会开始准备他点的菜，直到你点的菜准备好。他可能不能保证你的菜会在第二个顾客之前先准备好。因为这取决于厨师决定的优先顺序。

**多处理-** 现在我们假设你去了一家有两个厨师的餐厅。当你和第二位顾客点餐时，你们很可能会同时准备好食物，因为一位厨师正在准备你的食物，而另一位厨师正在准备第二位顾客的食物。这就是多重处理。其中两个任务并行运行，而不像多线程那样是并发的。

让我们看一个多处理池的基本例子

```
import time
import multiprocessing
def multi_process(val):
    time.sleep(5)
    return valalphabets = ['a','b','c','d','e']
with Pool() as pool:
    pool = multiprocessing.Pool(processes=5)
    start = time.time()
    output = pool.map(multi_process, alphabets)
    stop = time.time()
    pool.close()
    print('Execution Time :',stop-start)==========================================================
processes = 5 :Execution Time : 5.009518384933472
processes = 1 :Execution Time : 25.030145168304443
```

使用上面的代码，所有的字母被同时传递给多进程函数，总时间是 5 秒。如果没有多重处理，常规 for 循环的时间将是 5*5 = 25 秒

因为在多处理中，如果我们想像在 for 循环中那样增加计数器的值，所有的进程都会同时运行

```
counter = 0
for i in range(10):
    counter += 1=======================================================
Output - counter = 10
```

如果我们使用多处理或线程，上述过程将无法复制。必须进行一些更改，以确保计数器增加到 10，而不只是任何低于 10 的值。下面我们来看一个例子。

```
import time
import multiprocessing
from multiprocessing import Pool, cpu_count,Valuedef init(counter):
    global counter1
    counter1 = counterdef multi_process(val):
    time.sleep(5))
    counter1.value = counter1.value + 1
    return valif __name__ == '__main__':
    alphabets = ['a','b','c','d','e']
    counter = multiprocessing.Value('i',0)
    with Pool() as pool:
        pool = multiprocessing.Pool(processes=5, initializer=init, initargs=(counter,))
        start = time.time()
        output = pool.map(multi_process, alphabets)
        stop = time.time()
        pool.close()
        print('Execution Time :',stop-start)
    print(counter.value)=======================================================
Expected Output - counter = 5 (counter should increment to 5 for 5 items in list)
Got Ouptut Random numbers - 5,4,2,1,3 etc.
```

这种行为的原因是因为多处理多个进程可以同时访问变量，这使得很难将计数器递增 5。为了克服这个问题，多处理模块具有解决上述问题的锁定特征

## 锁

当一个进程想要访问一个共享变量时，它首先获取一个锁，然后增加变量并释放锁。这样做可以确保循环像在 for 循环中一样递增。如果锁被获取，而另一个进程必须获取该锁，那么它就等待，直到现有的锁被拥有该锁的当前进程释放。下面给出了上述问题的解决方案

```
import time
import multiprocessing
from multiprocessing import Pool, cpu_count,Valuedef init(lock,counter):
    global glock,counter1
    glock = lock
    counter1 = counterdef multi_process(val):
    time.sleep(1)
    glock.acquire()
    counter1.value = counter1.value + 1
    glock.release()
    return valif __name__ == '__main__':
    alphabets = ['a','b','c','d','e']
    counter = multiprocessing.Value('i',0)
    lock = multiprocessing.Manager().Lock()
    with Pool() as pool:
        pool = multiprocessing.Pool(processes=5, initializer=init, initargs=(lock,counter,))
        start = time.time()
        output = pool.map(multi_process, alphabets)
        stop = time.time()
        pool.close()
        print('Execution Time :',stop-start)
    print(counter.value)=============================================================
Output - counter = 5 (No matter how many times we run the above code the output will of counter will be 5 )
```

> 有时间的话，我会不断更新内容。谢谢