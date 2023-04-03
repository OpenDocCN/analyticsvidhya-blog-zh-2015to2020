# 在 Python 中保留和传递变量的信息状态

> 原文：<https://medium.com/analytics-vidhya/retaining-and-passing-information-states-of-variables-in-python-7b39802d1362?source=collection_archive---------9----------------------->

![](img/fb589ec4bd57b600f1f23bfebccefc91.png)

照片由[Rene b hmer 拍摄](https://unsplash.com/photos/YeUVDKZWSZ4)

状态信息在许多现实世界的程序中起着至关重要的作用。状态决定了实体(函数、类、对象..等等)。实体-属性关系是其使用的关键因素。

在尝试解决一个真实世界的程序时，我遇到了这样一个问题，在这个程序中，我希望实体保存系统的状态，以便以后使用。在这个过程中，我发现有多种方法可以做到这一点，我试图简化这些方法，以便对其他人有所帮助。

让我们来看一个简单的问题，通过这个问题我们可以理解程序中状态的保持。问题陈述是实现一个程序，根据函数的初始状态来计算调用函数的次数。

….**第一次尝试**🤔

```
count = 0def stateful(init_state):
    global count
    count = init_state      # intialize the state to global variable
    def inner():
        global count        # Acess global variable and increment
        count += 1
        return "This is a sample program", count, init_state
    return innerret = stateful(0)
ret()
ret()
ret()
print(ret())               # firstret2 = stateful(10)
print(ret2())              # Second
print(ret())               # Third
```

上面的代码打印出以下输出

```
('This is a sample program', 4, 0)
('This is a sample program', 11, 10)
('This is a sample program', 12, 0)
```

在上面的代码中，我们使用了一个全局变量`count`来跟踪对`stateful()`的调用次数的状态。由于对`stateful()`的调用没有保存`count`变量的状态，我们正在访问全局`count`变量，并将函数的初始状态`init_state`赋给`count`，最后为每个调用递增变量。虽然这确实部分解决了我们的问题，但是我们仍然需要保留发送到`stateful()`的初始状态。

对函数的`first`调用将`count`记为`4`。

对函数的`second`调用将全局变量 count 重新初始化为`10`，这`count`被视为`11`。

对函数的`third`调用仍然使用先前的初始化状态，并将先前的`count`值增加到`12`

现在，我们看到上面的代码有两个问题。太多重复的代码行(我讨厌这一点)其次，我想隔离新的对象初始化和独立的行为。所以，我一直在寻找更多的答案。

….**第二次尝试**🤔

```
def stateful(init_state):
    count = init_state
    def inner():
        nonlocal count
        count += 1
        return "This is a sample program", count, init_state return inner ret = stateful(0)             # init state 0
ret()
ret()
ret()
print(ret())                  # first ret2 = stateful(10)            # init state 10
print(ret2())                  # second
print(ret())                  # third
```

上面的代码输出如下

```
('This is a sample program', 4, 0)
('This is a sample program', 11, 10)
('This is a sample program', 5, 0)
```

在上面的代码中，我们使用了语句`nonlocal`，这样我们不仅可以在后面的阶段使用变量`count`作为参考，还可以通过调用访问和修改变量`count`的状态。

对函数`ret()`的`first`调用将`count`作为`4`。

对函数`ret2()`的`second`调用将`count`作为`11`。

对函数`ret()`的`third`调用给出的计数为`5`。

***注意*** :注意`ret`和`ret2`的对象是如何被`init_state`保留的。对函数`ret`的注释行`third`调用是对该函数的第五次调用，因此计数增加到`5`。当对象`ret2`用不同的`init state`即`10`初始化时，它使用它的一个状态来跟踪变量`count`，该变量增加到`11`。

现在这个解决方案让我很高兴，因为它给出了问题陈述的答案，但我想知道是否有更多的解决方案给我更多的灵活性来改变状态变量。

… **第三次尝试**🤔

```
def stateful(init_state):
    def inner():
        inner.count += 1
        return "This is a sample program", inner.count, init_state
    inner.count = init_state
    return inner ret = stateful(0)              # init state 0
ret()
ret()
ret()
print(ret())                   # first ret2 = stateful(10)            # init state 10
print(ret2())                  # second
print(ret())                   # thirdprint(ret.count, ret2.count)
ret2.count = 20                # change in entity's init state 
print(ret2())                  # fourth
```

上面的代码输出如下

```
('This is a sample program', 4, 0)
('This is a sample program', 11, 10)
('This is a sample program', 5, 0)
5 11
('This is a sample program', 21, 10)
```

在上面的代码中，我们使用了函数属性来跟踪`count`调用的状态。由于函数`inner`已经定义，我们可以在返回函数对象之前创建一个使用函数名创建的函数属性`inner.count`。对象`ret`和`ret2`具有它们相应的功能属性，这些属性保留了状态信息。

对函数`ret()`的`first`调用给出了`count` 4

对函数`ret2()`的`second`调用将`count`作为`11`给出。

对函数`ret()`的`third`调用给出的计数为`5`。

对函数`ret2()`的`fourth`调用给出的计数为`21`。这是因为我们已经使用功能属性`inner.count`将`init_state`修改为新状态。

***注:*** 注意我们如何能够直接访问变量并打印它们相应的值。我们也使用函数属性`inner.count`改变了状态，但是，我们不能修改嵌入到对象中的值`init_state`，因此我们在输出行`(‘This is a sample program’, 21, 10)`中看到值`10`。

虽然这个解决方案给了我更好的改变状态的灵活性，但是它也打开了状态被另一个模块改变的可能性，这可能导致意想不到的结果。

… **第四次尝试**

```
class stateful:
    def __init__(self,init_state):
        self.count = init_state
        self.init_state = init_statedef inner(self):
        self.count +=1
        return "This is a sample program", self.count, self.init_state ret = stateful(0)                    # init state 0
ret.inner()
ret.inner()
ret.inner()
print(ret.inner())                   # first ret2 = stateful(10)                  # init state 10
print(ret2.inner())                  # second
print(ret.inner())                   # thirdprint(ret.count, ret2.count)
ret2.count = 20                      # change in entity's init state
print(ret2.inner()).                 # fourth
```

上面的代码输出如下

```
('This is a sample program', 4, 0)
('This is a sample program', 11, 10)
('This is a sample program', 5, 0)
5 11
('This is a sample program', 21, 10)
```

在上面的代码中，我们使用了 class 属性来保留状态信息。我们已经用类对象`ret`和`ret2`来调用类方法。

对函数`ret()`的`first`调用给出了`count` 4

对函数`ret2()`的`second`调用将`count`作为`11`给出。

对函数`ret()`的`third`调用给出的计数为`5`。

对函数`ret2()`的`fourth`调用给出的计数为`21`。这是因为我们已经使用类别属性`count`将状态修改为新的状态。

***注意:*** 我们已经对类方法进行了显式调用。

我们可以修改这个解决方案，使它成为一个可调用的，这是我最后的选择。

… **第五次尝试🤔**

```
class stateful:
    def __init__(self,init_state):
        self.count = init_state
        self.init_state = init_state
    def __call__(self, *args, **kwargs):
        self.count +=1
        return "This is a sample program", self.count, self.init_state ret = stateful(0)              # init state 0
ret()
ret()
ret()
print(ret())                   # first ret2 = stateful(10)            # init state 10
print(ret2())                  # second
print(ret())                   # thirdprint(ret.count, ret2.count)
ret2.count = 20                # change in entity's init state
print(ret2())                  # fourth
```

上面的代码输出如下

```
('This is a sample program', 4, 0)
('This is a sample program', 11, 10)
('This is a sample program', 5, 0)
5 11
('This is a sample program', 21, 10)
```

在上面的代码中，我们使用了 class 属性来保留`count`的状态。

对函数`ret()`的`first`调用给出了`count` 4

对函数`ret2()`的`second`调用将`count`作为`11`。

对函数`ret()`的`third`调用给出的计数为`5`。

对函数`ret2()`的`fourth`调用给出的计数为`21`。这是因为我们已经使用类属性`count`将状态修改为一个新的状态。

***注意:*** 注意我们是如何使用 Python 的 dunder 方法使对象可调用的。

# 结论

我试图解释解决问题的不同方法。虽然这些解决方案使我们能够跨程序传递状态信息，但我们没有考虑每种方案的利弊。处理状态信息时最重要的因素，跨程序传递状态信息(性能)的成本甚至没有被考虑。

也许我会再写一篇文章来讨论每种方法的优缺点。