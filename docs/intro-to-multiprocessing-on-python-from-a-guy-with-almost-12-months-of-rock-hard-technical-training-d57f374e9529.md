# 来自一个受过近 12 个月艰苦技术培训的人的 Python 多处理介绍。

> 原文：<https://medium.com/analytics-vidhya/intro-to-multiprocessing-on-python-from-a-guy-with-almost-12-months-of-rock-hard-technical-training-d57f374e9529?source=collection_archive---------16----------------------->

你好，谢谢你阅读我的第一篇文章。在本教程中，我将介绍一些多重处理的方法，并重点介绍我遇到的一些用例。

假设我们希望从各种来源收集和清理数据，我们可以按顺序完成所有的任务，或者我们可以同时运行它们。实际上，你应该问自己这个问题，你是愿意一个接一个地吃小熊软糖让你的手疲劳，还是愿意把整包软糖塞进你的嘴里，让你无用的下巴来完成这项工作？

首先，您需要使用 pip 安装多处理包

```
pip install multiprocessing
```

**开个玩笑**它附带了自 **python 2.6** 以来的所有 Python 标准版本。

所以只需导入我们将要使用的函数。

```
from multiprocessing import Process, Pool, Manager, Queue
```

接下来，让我们编写一个虚拟函数来演示如何使用多处理中的一些基本功能。

```
def doComplexCalculation (num) :
    otherNum = 9 if num == 10 :
      return 21

    return num + otherNum
```

如果我们需要多次执行 doComplexCalculation，我们总是可以创建一个工人池来执行任务。

```
pool = Pool(5)result = pool.map(doComplexCalculation, [25, 42, 69])print (result)// [34, 51, 78]
```

够简单吧？如果你的函数需要特殊的关键字参数呢？

```
def doEvenMoreComplexStuff (num, isMutable=False, isDirty=True) : otherNum = 9 if isMutable == True : otherNum += 6 if isDirty == False : num += 6 return num + otherNum
```

为此，更直接的方法是生成一个流程对象，并伪同时运行这些对象。

```
processes = []nums = [25, 42, 69] for i in range(3): p = Process( target=doEvenMoreComplexStuff, kwargs={"isMutable":True, "isDirty":False, "num":nums[i]}, ) processes.append(p)for p in processes : p.start()for p in processes : p.join()
```

我现在要分解代码，让你可爱的小脑袋明白。第一段代码生成一个流程对象，并将一个函数与其所需参数进行映射。请注意，在声明目标时，避免在函数末尾添加“()”，因为这将告诉 Python 在该行执行所述函数。

```
p = Process( target=doEvenMoreComplexStuff, kwargs={"isMutable":True, "isDirty":False, "num":nums[i]},)
```

的方法。start()基本上是“启动”进程。鉴于。join()确保您已经启动的这些进程在其余主代码执行之前完成。我们把 start 和 join 方法分开，就好像它们在同一个 for 循环中执行一样，你的解释器只是在开始下一个进程之前等待前一个进程的加入，这实质上违背了多重处理的目的。

```
for p in processes : p.start()for p in processes : p.join()
```

最后，精明的读者(或者那些真正阅读整篇文章的怪人)可能会意识到，您无法从这些函数中检索返回值。你完全正确！

您可能会想，您可以将字典或列表传递给变量来存储值。但是，这些函数是在与主函数分开的内存块中执行的，因此您无法使用普通的字典和列表来检索返回值。相反，您需要使用 Manager 和 Queue 类。

当您对结果的返回顺序不太感兴趣时，可以使用队列。队列也比管理器类需要更少的计算能力。

当您对对象的返回顺序感兴趣时，可以使用管理器类。例如，如果您需要将文件上传到 S3 存储桶中，这将非常有用。我个人使用 dict 函数，但也可以在 Manager 类中随意尝试其他函数。

请注意，管理器类本身需要一个处理核心，所以要考虑到这一点，而队列不需要。

要实现 Manager.dict()或 Queue，我们需要在函数中添加一个关键字参数。

```
## Implementation of a Manager.dict()
def doComplexCalculation (num, queue=None) :
    otherNum = 9 if num == 10 :
        return 21 if queue != None :
        queue['complexCalculation'] = num + otherNum

    return num + otherNum## Implementation of a Queue
def doComplexCalculation (num, queue=None) :
    otherNum = 9 if num == 10 :
        return 21 if queue != None :
        queue.put( num + otherNum )

    return num + otherNum
```

现在把所有这些放在一起。

```
from multiprocessing import Process, Manager## Implementation of a Manager.dict()def doComplexCalculation (num, queue=None) :
    otherNum = 9 if num == 10 :
        result = 21 else :
        result = num + otherNum if queue != None :
        queue[f'complexCalculation for {num}'] = result return resultq = Manager().dict()processes = []process1 = Process(target=doComplexCalculation, kwargs={'num': 42, 'queue': q})
process2 = Process(target=doComplexCalculation, kwargs={'num': 69, 'queue': q})
process3 = Process(target=doComplexCalculation, kwargs={'num': 10, 'queue': q})processes.append(process1)
processes.append(process2)
processes.append(process3)for p in processes :
    p.start()for p in processes :
    p.join()print (q)"""
{'complexCalculation for 42': 51, 'complexCalculation for 69': 78, 'complexCalculation for 10': 21}
"""
```

她就写了这么多！