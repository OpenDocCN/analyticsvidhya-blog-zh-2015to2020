# __init__()是 python 中的构造函数吗？

> 原文：<https://medium.com/analytics-vidhya/is-init-the-constructor-in-python-43bb744e73b?source=collection_archive---------2----------------------->

TLDR；__init()不是构造函数，__new__()是。

![](img/59542e22a571bd6a2dd439474be90c82.png)

由[塞巴斯蒂安·赫尔曼](https://unsplash.com/@officestock?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

**一般来说，**刚接触 python 的开发者被告知 __init__()方法被称为构造函数。

在深入潜水之前，让我们先了解一下**自我**。

我们已经看到 self 被用作该方法的第一个参数。我们举个例子来理解这一点。

```
class Power:
     def __init__(self,n):
         self.n=n

     def cal(self):
         return self.n**self.n
```

在这里，Power 是一个有两种方法的类。cal 用于计算数的幂。实例化后，我们得到以下输出

```
p=Power(5)
p.cal()
3125
```

现在，添加另一个方法 show()并且不要传递 self。

```
def show():
    print('method without self')
```

调用 we 后出现以下错误

```
--------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-84-02f33f5f7a8f> in <module>
----> 1 p.show()

TypeError: show() takes 0 positional arguments but 1 was given
```

上述问题可以通过添加 self 作为参数或使其成为静态方法来解决。

**到目前为止的结论:**

*   __init__()定义了两个参数
*   我们只传递了一个，它仍然没有给出错误。
*   为什么把**自我**作为第一个论证，而不是别的什么。

现在让我们回到 Power 类，检查一下内部发生了什么。

```
type(p.cal)
<class 'method'>
type(Power.cal)
<class 'funcion'>
```

迷茫**！！**

在 Python 方法中，对象本身作为第一个参数传递给相应的函数。简言之，方法调用`p.cal()`等于`Power.cal(p)`。

通常，当我们用一些参数调用一个方法时，相应的类函数是通过将方法的对象放在第一个参数之前来调用的。所以，像`**obj.method(args)**`这样的东西就变成了`**Class.method(obj)**`。

这就是类中函数的第一个参数必须是对象本身的原因。写这个参数本身仅仅是一个惯例。它不是一个关键字，在 Python 中没有特殊含义。我们可以使用其他名字(比如`this`)。

**构造函数是创建对象的方法。**

**至此结论:**

*   `__init__()`难道**不是建造师**。我们将自我视为第一个参数，它只不过是对象本身，即对象已经存在。
*   `__init__()`在对象创建后立即被**调用，用于初始化对象。**

**如果不是 __init__()那么？**

答案是 `**__new__()**`

```
__new__(cls, *args, **kwargs) # signature
```

__new__()以 cls 作为第一个参数，表示需要实例化的类，编译器在实例化时自动提供这个参数。args 和**kwargs 是要传递的参数。

__new__()总是在 __init__()之前调用

示例:

```
class Power:

    def __new__(cls,*args):
        print("new method")
        print(cls)
        print(args)# create our object and return it
        obj = super().__new__(cls)
        return obj

    def __init__(self,n):
        print("init method")
        self.n=n

    def cal(self):
        return self.n**self.np=Power(5)
p.cal()
```

输出:

```
new method
<class '__main__.Power'>
(5,)
init method
3125
```

希望这对你有用。如果你在理解这个问题上有任何困难，或者你需要任何帮助，请联系我。

*电子邮件:sambhavchoradia@gmail.com*

*社交媒体:* [*LinkedIn*](https://www.linkedin.com/in/sambhav-choradia/)

快乐编码。