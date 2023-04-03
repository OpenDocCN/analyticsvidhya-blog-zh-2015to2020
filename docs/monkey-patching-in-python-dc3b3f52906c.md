# Python 中的猴子补丁

> 原文：<https://medium.com/analytics-vidhya/monkey-patching-in-python-dc3b3f52906c?source=collection_archive---------2----------------------->

![](img/764afaac7cc1a040edaa3908ed9e023b.png)

杰米·霍顿在 [Unsplash](/s/photos/monkey?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

# 什么是猴子补丁

在 Python 中，术语 monkey patch 仅指运行时对类或模块的动态修改，这意味着 monkey patch 是一段 Python 代码，它在运行时扩展或修改其他代码。

猴子补丁只能用动态语言来完成，python 就是一个很好的例子。在 Monkey patching 中，我们在运行时重新打开现有的类或类中的方法，并改变行为，这应该谨慎使用，或者只在真正需要时使用。因为 Python 是一种动态编程语言，所以类是可变的，所以你可以重新打开它们，修改甚至替换它们。

它通常用于用自定义实现替换或扩展模块或类级别的方法。

让我们实现猴子打补丁——

```
class MonkeyPatch:
    def __init__(self, num):
        self.num = num def addition(self, other):
        return (self.num + other)obj = MonkeyPatch(10)
obj.addition(20)
```

输出-

```
30
```

正如你在下面看到的，现在只有两个方法`__init__`和`addition`可用于上面的类对象。(你也可以使用`dir(obj)`来获取对象的成员)

```
import inspect inspect.getmembers(obj, predicate=inspect.ismethod)
```

输出-

```
[('__init__',
  <bound method MonkeyPatch.__init__ of <__main__.MonkeyPatch object at 0x7f32495d7c50>>), ('addition',
  <bound method MonkeyPatch.addition of <__main__.MonkeyPatch object at 0x7f32495d7c50>>)]
```

在上面的代码中，我们已经用一个`addition`方法定义了一个`MonkeyPatch`类，稍后，我们将向`MonkeyPatch`类添加一个新方法。假设新方法如下。

```
def subtraction(self, num2):
    return self.num - num2
```

现在我们如何将上述方法添加到`MonkeyPatch`类中，简单地用赋值语句将`subtraction`函数放入`MonkeyPatch`类中，如下所示

```
MonkeyPatch.subtraction = subtraction
```

新创建的`subtraction`函数将可用于所有现有实例以及一个新实例，现在让我们来做一些实际操作

```
import inspectinspect.getmembers(obj, predicate=inspect.ismethod)
```

输出-

```
[('__init__',
  <bound method MonkeyPatch.__init__ of <__main__.MonkeyPatch object at 0x7f28186b8c88>>), ('addition',
  <bound method MonkeyPatch.addition of <__main__.MonkeyPatch object at 0x7f28186b8c88>>), ('subtraction',
  <bound method subtraction of <__main__.MonkeyPatch object at 0x7f28186b8c88>>)]
```

您是否注意到我们之前定义的现有对象`obj`包含新创建的函数，让我们检查一下新创建的函数是否适用于现有实例-

```
>>> obj.subtraction(1)               # Working as expected
9
>>> obj_1 = MonkeyPatch(10)          # create some new object
>>> obj_1.subtraction(2)
8
```

`obj.subtraction`按预期工作，但是请记住，如果同名的方法已经存在，那么这将改变现有方法的行为。

# 要记住的事情

最好的事情是不要猴子补丁。您可以为想要改变的类定义子类。然而，如果需要猴子补丁，那么遵循这些规则-

1.  如果你有一个很好的理由，就使用它(比如临时的关键修补程序)
2.  编写适当的文档，描述 monkey 补丁的原因
3.  文档应包含有关删除猴子补丁的信息以及需要注意的事项。很多猴子补丁都是临时的，所以应该很容易移除。
4.  试着让 monkey 补丁尽可能透明，并把 monkey 补丁代码放在单独的文件中

# 结论

现在我们已经学会了如何用 Python 来做一个猴子补丁。但是，它有自己的缺点，应该小心使用。嗯，这通常意味着您的应用程序的糟糕的体系结构，并且不是一个好的设计决策，因为它在磁盘上的原始源代码和观察到的行为之间产生了差异，并且在故障排除时会非常混乱。

# 参考

[https://en.wikipedia.org/wiki/Monkey_patch](https://en.wikipedia.org/wiki/Monkey_patch)
[https://stack overflow . com/questions/5626193/什么是猴子打补丁](https://stackoverflow.com/questions/5626193/what-is-monkey-patching)