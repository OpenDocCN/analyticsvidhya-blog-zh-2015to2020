# 如何用 Python 创建线程安全的单例类

> 原文：<https://medium.com/analytics-vidhya/how-to-create-a-thread-safe-singleton-class-in-python-822e1170a7f6?source=collection_archive---------0----------------------->

![](img/951488aef26623e232455f53c7a74bea.png)

***2022 年 12 月 22 日更新*** *:我最初是用 Python 3.9 写的这篇文章。我刚刚检查了 3.11 版本，一切正常。我还添加了一个示例单元测试。*

在本文中，我将向您展示如何用 Python 创建线程安全的单例类。我写这篇文章是因为网上大多数关于 Python 中单例的例子要么[糟糕透顶](https://www.geeksforgeeks.org/singleton-method-python-design-patterns/)要么[不是线程安全的](https://www.tutorialspoint.com/python_design_patterns/python_design_patterns_singleton.htm)。这是最后的代码，因为您可能是为了快速回答问题，以便解决当前的任务:

```
class Singleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None: 
            with cls._lock:
                # Another thread could have created the instance
                # before we acquired the lock. So check that the
                # instance is still nonexistent.
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

# 说明

***什么是对付巨蟒的*** `***__new__***` ***盾德法？***

每当 Python 实例化一个类的新对象时，就会调用`__new__`。通常，`__new__`会转到该类的超类，即`Object`，并实例化一个新对象，然后将该对象传递给`__init__`，并带有传递给`__new__`的任何参数。我们截取这个方法，并告诉它创建一个且只有一个类实例(即 Singleton)。然后这个类对象像往常一样被传递给`__init__`方法。

*`***Lock***`***怎么回事？****

*这是一个好问题。`threading.Lock`是一个实现原始锁对象的类。它允许运行我们代码的线程成为访问锁的[上下文管理器](https://book.pythontips.com/en/latest/context_managers.html)中代码的唯一线程，只要它持有锁。这意味着没有其他线程可以与拥有锁的线程同时运行`with cls._lock`块中的代码。*

****这两个*** `***cls._instance***` ***检查有什么关系？****

*在获取锁之前，我们检查`cls._instance`是否为`None`。有一种边缘情况，在这个线程中`cls._instance`是`None`，而另一个线程将要调用`cls._instance = super(Singleton, cls).__new__(cls)`。在这个例子中，创建了两个类对象，从而破坏了我们类的 Singleton 属性。*

****为什么不把整个*** `***__new__***` ***方法放在*** `***Lock***` ***上下文管理器中并避免第二个*** `***if not cls._instance***` ***检查，这样呢？****

```
*class Singleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
        return cls._instance*
```

*这是可行的，乍看之下，*似乎比*更好，因为代码行减少了。然而，问题是获取锁是一个昂贵的操作。拥有一个在不需要的时候获取锁的类/方法会导致代码运行缓慢，很难确定。仅在必要时获取锁。*

*我怎么知道这真的有效？*

*很棒的问题！我们可以通过单元测试断言正确的单体行为。下面的例子是用 [pytest](https://docs.pytest.org/) 编写的通过单元测试:*

```
*def test_singleton_is_always_same_object():
    assert Singleton() is Singleton()

    # Sanity check - a non-singleton class should create two separate
    #  instances
    class NonSingleton:
        pass
    assert NonSingleton() is not NonSingleton()*
```

***更新**:如果用`if not cls._instance`代替`if cls._instance is None`时，该代码以前使用过。像这样:*

```
*class Singleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:  # This is the only difference
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance*
```

*正如 pawewiszniewski 在评论中指出的，如果你的单例重载了 T10，你可能会遇到奇怪的行为。所以显式检查`if _instance is None`更安全，而不是依赖`_instance`是[假的](https://www.pythonmorsels.com/truthiness/)。希望您的测试套件能够在第一时间捕捉到任何此类潜入软件的问题。*