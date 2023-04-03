# 同步和临界区问题

> 原文：<https://medium.com/analytics-vidhya/synchronization-and-critical-section-problem-7b214cad4c87?source=collection_archive---------14----------------------->

![](img/b8db98c81f357348eb0d162ff8528102.png)

# 什么是同步？

在多编程或多线程编程环境中的同步是协调共享数据的访问和操作的过程。如果你读过我以前的一些博客，你会注意到我们偶尔会遇到两个线程试图同时访问相同共享数据的问题。为了处理这个常见问题，我们必须了解如何同步访问这些共享数据。

# 临界截面问题

**临界**段位于代码中一个线程正在访问可能被另一个线程访问的共享数据的任何地方。同步将试图防止两个线程同时访问临界区/共享数据。

## 实现同步

实现同步有许多方法，但它们都遵循以下通用伪代码:

```
// Some code
.
.
.
acquire_access 
    // critical section 
    // access and update shared data
release_access
.
.
.
// some code
```

如果另一个线程正在使用访问权限，获取访问权限将导致该线程等待；如果没有其他线程正在使用访问权限，它将成功获取访问权限。一旦线程完成，它就释放访问，以便另一个线程可以使用它。

最简单的实现是使用所谓的`Semaphore`

```
sem_t mutex;
sem_init(&mutex, 0, 1);
int critical_section = 0;//thread 1sem_wait(&mutex);
critical_section ++;
sem_post(&mutex);//thread 2sem_wait(&mutex);
critical_section ++;
sem_post(&mutex);
```

## 近距离观察

我们从声明一个互斥变量开始。我们用`sem_init(&mutex, 0, 1)`初始化它。第一个参数是我们试图初始化的信号量，`0`允许信号量在线程间共享，`1`是我们想要用来初始化`sem_t mutex`的值。[了解更多](http://man7.org/linux/man-pages/man3/sem_init.3.html)。总是将信号量初始化为正数，稍后我会解释为什么。

`thread 1`和`thread 2`的片段非常相似。我们调用`sem_wait(&mutex)`，这相当于获取访问权。下面是这个函数如何工作的伪代码

```
sem_wait(semaphore){
    if semaphore greater than 0
        semaphore -1
        return;
    else
        wait/sleep
}
```

根据上面的代码，即使是第一个访问临界区的线程，信号量也必须初始化为正数。当第一个线程访问临界区时，`mutex = 1`，调用`sem_wait`将减少互斥量和`mutex = 0`并返回。允许第一个线程进入临界区。

当第一个线程正在访问临界区时，另一个线程试图进入临界区，它调用`sem_wait(&mutex)`，但是`mutex= 0`因此它被置于睡眠或被迫等待。

第一个使用完临界区，调用`sem_post(&mutex)`，相当于释放访问。下面是`sem_post`的伪代码。

```
sem_post(semaphore){
    semaphore + 1
    wake up the first thread in line to access the critical section  
}
```

在第一个线程调用`sem_post(&mutext)`后，第二个线程醒来并进入临界区，过程重复。

# 最后一个音符

如果你想进入多线程编程，理解这个概念是至关重要的。

*最初发布于*[https://www . dev survival . com/synchron ization-and-critical-section/](https://www.devsurvival.com/synchronization-and-critical-section/)