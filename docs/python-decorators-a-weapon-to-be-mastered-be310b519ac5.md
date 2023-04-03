# Python 装饰者——需要掌握的武器

> 原文：<https://medium.com/analytics-vidhya/python-decorators-a-weapon-to-be-mastered-be310b519ac5?source=collection_archive---------0----------------------->

![](img/d905fb4a63946a0ffcdd1b9b966a6cb8.png)

你好 Pythonist！。在这篇文章中，我们将学习 Python Decorators。我想让你知道并感受一下简单 python 函数的威力，以及它们是如何让 Decorator 成为 Python 如此美丽的特性的，而不是直接跳到它的工作上。

Python 函数拥有三种能力，这就是为什么它们被称为 ***【第一类对象】*** 。

那些能力是什么？他们能做什么？让我看看-

1.  **你可以把它们分配给变量-**

```
def parent():
    print("Inside parent function")fun=parentfun()>> 
Inside parent function
```

*这里“fun”变量保存了父函数的引用，而赋值父函数没有被调用。现在可以调用“fun”变量作为执行父函数*的函数

**2。它们可以作为值传递给其他函数**

```
def neighbor():
   print("Hey, I am neighbor")def parent(func):
   print("hi there!")
   func()fun=parent
fun(neighbor)>>
hi there!
Hey, I am neighbor
```

*这里“neighbor”作为值传递给“parent”函数，然后在其中调用。*

**3。它们可以作为其他函数的值返回**

```
def neighbor():
   print("Hey, I am neighbor, where is your son?")
   return 1def parent(func):
   print("hi there!")
   call = func() # nested function
   def son():
      print("Hi neighbor, I am his son") # nested function
   def daughter():
      print("Hi neighbor, I am his daughter")

   if call == 1:
      return son
   else:
      return daughterfun=parentchild = fun(neighbor) # returns reference of nested function>>
hi there!
Hey, I am neighbor, where is your son? child() # nested function "son" gets called
>>
Hi neighbor, I am his son
```

*这里我们在父函数中定义了两个嵌套函数，即“儿子”和“女儿”。这种嵌套函数既不能从外部直接调用，也不能在外部函数被调用时自动执行。但是我们可以在外部函数之外返回它们的引用，这样它们就可以从外部被调用。*

> python 函数的这三种能力是最有用和最强大的武器(即装饰者)的成分。

> 没错，是 Python Devs 击落很多问题的利器！！！

记住这三个要素，它会帮助你在我们前进的时候理解装饰者。

# W 什么是装饰工？

> 教科书定义-装饰器是接受另一个可调用函数(函数、方法和类)并扩展其行为而不显式修改它的函数。
> 
> 通俗地说——装饰者包装一个可调用对象，修改它的行为。

听起来有趣吗？

这是一个简单的装饰者的例子

```
def decorator(func):
    def wrapper():
      print("Before the function gets called.")
      func()
      print("After the function is executed.")
    return wrapperdef wrap_me():
    print("Hello Decorators!")wrap_me = decorator(wrap_me)
wrap_me()>>
Before the function gets called
Hello Decorators!
After the function is executed
```

# 它是如何工作的？

在上面的例子中，我们看到了 python 函数的所有三种功能。

我们通过传递函数 *wrap_me* 来调用*装饰器*。当*装饰器*被调用时；*包装器*函数被定义为保存 *wrap_me* (一个闭包)的引用，并使用它来包装输入函数(即 *wrap_me* )，以便在调用时修改其行为。

> 什么是终结？-
> 
> 将一些数据(在我们的例子中是“wrap_me”函数引用)附加到代码上的技术在 Python 中被称为 [**闭包。**](/@amitrandive442/python-closures-b71e8847286f)
> 
> 即使变量超出范围，封闭范围内的值也会被记住。

包装闭包可以访问未修饰的输入函数，并且可以在调用输入函数之前和之后自由地执行额外的代码。

*装饰器*返回由 *wrap_me* 变量捕获的*包装器*函数的引用，当我们调用它时，我们将实际调用*包装器*函数。

此时，嵌套函数被执行，并在原始 *wrap_me* 函数调用前后执行指令。

> 这可能感觉有点沉重，但一旦你完全理解这是如何工作的，你一定会爱上 it❤

# @ (Pi)语法

您可以使用 Python 的@ (pi)语法，而不是在 wrap_me 上调用 decorator 并重新分配 *wrap_me* 变量。

```
def decorator(func):
    def wrapper():
      print("Before the function gets called.")
      func()
      print("After the function is executed.")
    return wrapper@decorator
def wrap_me():
    print("Hello Decorators!")wrap_me()>>
Before the function gets called
Hello Decorators!
After the function is executed
```

使用@ syntax 只是添加了语法糖来实现这种常用的模式或简写来调用输入函数上的装饰器。

# 接受论点的装饰者-

现在，任何人都可能会问的一个明显的问题是——“我如何将装饰器应用于接受参数的函数？”

Python 允许这样做，而且非常简单。你还记得我们有类似于 **args* 和 ***kwargs* 的东西来处理可变数量的参数吗？

就是这样！这两个人随时准备帮忙。

让我们写一个装饰器来记录函数参数信息-

```
def dump_args(func):
  def wrapper(*args,**kwargs):
      print(f'{args}, {kwargs}')
      func(*args, **kwargs)
  return wrapper@dump_args
def wrap_me(arg1,arg2):
    print(f"Arguments dumped")wrap_me("arg_dump1","arg_dump2")>>
('arg_dump1', 'arg_dump2'), {}
Arguments dumped
```

它使用*包装器*定义中的*和**操作符来收集所有位置和关键字参数，并将它们分别存储在变量 args 和 kwargs 中，然后转发给输入函数(即 wrap_me)。

装饰者是可重用的。这意味着你可以对多个函数使用同一个装饰器。你也可以从其他模块导入它。

# 堆叠装饰者

多个装饰器可以用于一个功能。

例如，假设我们需要记录函数的执行时间和参数-

```
import datetimedef dump_args(func):
   def wrapper(*args,**kwargs):
       print(f'{func.__name__} has arguments - {args}, {kwargs}')
       func(*args, **kwargs)
   return wrapperdef cal_time(func):
   def wrapper(*args,**kwargs):
       now = datetime.datetime.now()
       print("start of execution : ",  now.strftime("%d/%m/%Y %H:%M:%S"))
       func(*args,**kwargs)
       now = datetime.datetime.now()
       print("end of execution : ",  now.strftime("%d/%m/%Y %H:%M:%S"))
  return wrapper@cal_time
@dump_args
def wrap_me(arg1,arg2):
   print("Arguments dumped")wrap_me("arg_dump1","arg_dump2")
```

这里我们在 *wrap_me* 函数上堆叠了两个装饰器。想想现在会是什么行为？

事情是这样的-

```
>>start of execution : 08/05/2020 21:13:11
wrap_me has arguments - ('arg_dump1', 'arg_dump2'), {}
Arguments dumped
end of execution : 08/05/2020 21:13:11
```

装饰者是自下而上的。首先，输入函数被 *@dump_args* 装饰器包装，然后产生的(装饰的)函数被 *@cal_time* 装饰器再次包装。

在实际的可调用执行之前，你需要执行的任何通用功能，只要简单地把它写成 decorator，并把它放在特定的可调用上。

例如

*测井*

*强制认证*

*计算功能性能*

*   *缓存等。*

正如我前面说过的，decorators 是可重用的构建模块，使这个特性更加强大，它经常被用于标准库和第三方库中。

在编写面向对象的 Python 代码时，你经常会遇到 *@staticmethod、@abstractmethod、@classmethod* 对吗？现在你知道它们的真正含义了。当我们将它应用于任何函数时，试着探究一下这个装饰器到底做了什么。

请记住，装饰不应该被过度使用，因为它不能解决所有问题，但如果你知道何时何地使用它，它们是非常有用的。

**如果你到达这里并且理解了这篇文章中的一切，那么*祝贺你！！*你刚刚学到了 Python 中最难的一个概念。**

**干杯！！**