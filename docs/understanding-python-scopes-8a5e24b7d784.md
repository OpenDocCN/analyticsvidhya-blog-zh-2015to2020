# 了解 Python 范围

> 原文：<https://medium.com/analytics-vidhya/understanding-python-scopes-8a5e24b7d784?source=collection_archive---------18----------------------->

![](img/ff0b029405131ce24a9a526695226f35.png)

照片由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [Jp Valery](https://unsplash.com/@jpvalery?utm_source=medium&utm_medium=referral)

作为一名深度学习工程师，你发现自己需要为不同的问题编写非常不同的代码。理解 Python 开发的本质细节会让你成为一个更有创造力和更强大的问题解决者。

我们在编写代码时经常做的事情之一就是使用赋值(例如 a = 5)。无论你是在调用一个函数，一个类还是进行一个简单的计算，你都可能在使用作用域。范围定义了在代码的不同部分对不同标签的访问权限。也就是说，如果从函数 1 或函数 2 中执行，计算+5 可能有不同的意义。为了准确理解您正在处理的“a”是什么，并使它按预期运行，您需要理解范围。

# 内置和全局范围

我们将讨论的第一个范围是内置范围。当你在模块本身中，而不是在任何函数或类中时，你可以这样写:

```
print('this is just a print')---> this is just a print
```

Python 完全知道您的意思，它只是打印您请求的字符串。Python 怎么知道什么是“打印”？假设这是我们的整个脚本，Python 在模块中寻找打印的定义(也称为全局范围)，当它找不到任何定义时，它就转到**内置的**范围，在那里找到它需要的内容。Python 开始查看它被执行的范围，并且每当它找不到它需要的东西时就沿着链向上。这意味着我可以很容易地覆盖“print ”,并使其行为不同。

```
def print(*args):
    return 3 + 4print('this is just a print')
---> 7
```

现在当我执行 print 函数时，我实际上只得到整数 7。由于现在在**全局**作用域中有打印的定义，这就是正在使用的。这个**全局**打印范围定义，隐藏了**内置**范围定义，从而改变了它的行为。

# 全局和本地范围

上面提到的全局作用域是主模块作用域，这里没有定义任何特别的东西。然而，如果我们定义一个函数，函数内部的任何东西都是局部范围。这个**局部**范围是函数的上下文。让我们看看这是如何表现的:

```
# define x = 40 in the global scope.
x = 'python'
print(x)
---> 'python'# get access to global scope from a function
def print_global_x():
    print(x)
---> 'python'# define a function to change x 
def change_x():
    x = 5
    print('the local scope x =', x)change_x()
print('the global scope x =', x)---> the local scope x = 5
---> the global scope x = 'python'
```

正如我们所看到的，我们有两个不同的 x。一个是 5，第二个是“python”。你可以想象，当我们想把 x 作为一个整数使用时，这会变得很混乱，但我们实际得到的是字符串‘python’。在这种情况下，x 在 change_x 的上下文中是 5，但在主模块的上下文中是“python”。

在函数 print_global_x 中，我们可以再次看到寻找作用域函数的过程。它首先尝试执行的作用域，在这个例子中是函数本身，它没有找到 x 的任何赋值，所以它继续到下一个全局作用域，在那里找到并打印它。如果 x 没有在全局作用域中定义，Python 将在内置作用域中寻找它，在那里它找不到它，因此返回一个“ **NameError** 异常，说明 x 没有定义。

# 控制范围

这是否意味着我们只能在模块本身中定义全局范围？其实不是。我们甚至可以在函数内部改变全局范围，使用**全局** **关键字**

```
a = 20def change_a():
    global a 
    a = 4print('a before change in global scope =', a)
change_a()
print('a after change in global scope =', a)---> a before change in global scope = 20
---> a after change in global scope = 4
```

我们在函数内部的全局作用域中改变了 a 引用的值(从而改变了我们调用 a 时得到的值)。当我们试图在全局上下文中访问一个，无论是从函数还是从模块，我们都会得到 4。

现在，假设我有几个嵌套的函数，我需要改变主函数的范围，所以它会影响到所有其他的函数。我们可以通过使用**非本地** **关键字**来实现这一点。这里我们定义了一个函数 define_b，它定义了 b，并将其打印为 4。然后我们定义一个名为 change_b 的新函数，它将 b 的**非局部**上下文更改为 10。当我们稍后调用新函数 print_b 时，我们看到 b 被修改了。

如果我们试图在全局范围内打印 b，我们仍然会得到一个未定义的 **NameError** ，因为 b 只存在于 define_b 函数的范围内。

```
def define_b():
    b = 4
    print('b before change =', b) def change_b():
        nonlocal b
        b = 10
    change_b()

    def print_b():
        print('new b is ', b)
    print('b after change =', b)
    print_b()
```

**外地**改什么范围？如果我们有更多的嵌套函数，我们会看到非局部函数在层次结构中上升到可以找到 b 的下一次赋值的位置，在我们的例子中，b 的下一次赋值是在 define_b 函数中，因此它改变了这个范围。注意，非局部不会改变全局范围，为此我们仍然必须使用**“全局 b”。**

我希望这能让使用作用域变得更容易理解，因为这是一个非常重要的话题。

如有任何其他问题，请随时通过 Bakugan@gmail.com 或 [Linkedin](https://www.linkedin.com/in/chen-margalit-4b1a67117/) 联系我