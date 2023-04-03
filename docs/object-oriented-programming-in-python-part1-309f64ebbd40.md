# Python 中的面向对象编程第 1 部分

> 原文：<https://medium.com/analytics-vidhya/object-oriented-programming-in-python-part1-309f64ebbd40?source=collection_archive---------28----------------------->

2020 年 7 月 3 日

# **Blog16**

![](img/e5a420464f2940aa6953b44c4a701ab2.png)

我们都有意或无意地使用过面向对象编程。如果你正在收集成为数据科学家的技能，那么 OOP 也是一个需要学习的重要课题。

你想知道我们使用的著名软件包 scikit-learn 是如何工作的吗(它是如何构建的)？？？

当我们使用

```
import pandas as pd 
pd.read_csv()
```

什么是 pandas，这个 read_csv()是从哪里来的，不需要我们自己创建这样的函数就可以使用吗？？？

那么这个博客是给你的。

让我们开始吧。欢迎:)

我们实际上使用哎呀的概念

那么 OOP 是什么呢？？

# 面向对象编程

面向对象编程是一种使用类和对象编写程序的风格。

面向对象编程允许您创建大型的模块化程序，这些程序可以很容易地随着时间的推移而扩展。

面向对象的程序对最终用户隐藏了实现。当你用 Scikit-learn 训练一个机器学习算法时，你不需要知道任何关于算法如何工作或者它们是如何编码的。您可以直接关注建模。如果实现发生了变化，作为软件包的用户，您可能永远也不会发现。

Python 包不需要使用面向对象的编程。您可以简单地拥有一个包含一组函数的 Python 模块。然而，大多数(如果不是全部)流行的 Python 包都利用了面向对象编程，因为:

1.  面向对象的程序相对容易扩展，尤其是因为继承
2.  面向对象的程序对用户隐藏了功能。

对象是由属性和方法定义的。

把物体想象成现实世界中存在的东西。\例如，如果我们以餐馆为例

*   餐馆本身就是客体
*   食物菜肴是客体
*   服务员也是对象

那么，如果服务员是一个对象，它的属性(特征)和方法(动作)是什么？？？

服务员的属性是姓名、地址、电话号码、薪水

服务员的工作方式有点菜、加薪和上菜。

现在我们知道了对象，但是两个对象可能具有不同的值，这意味着有两个对象，但是属性和方法是相同的，这意味着两个对象具有相同的属性类型，但是具有不同的值，这意味着两个对象具有相同的蓝图。

所以一个对象的蓝图叫做类。使用这个蓝图，我们可以创建许多对象。

所以我们来看看类的代码。

```
class Waiter:
    """The waiter class represents an type of person who takes order and serve dishes in a restaurant
    """

    def __init__(self, name, address, height, salary):
        """Method for initializing a Pants object

        Args: 
            name (str)
            address (str)
            height (int)
            salary (float)

        Attributes:
            name (str): name of waiter object
            address (str): address of a waiter object
            height (int): height of a waiter object
            salary (float): salary of a waiter object
        """

        self.name = name
        self.address = address
        self.height = height
        self.salary = salary

    def hike_salary(self, hike_percent):
        """The hike_salary method changes the salary attribute of a waiter object

        Args: 
            hike_percent (float): the new salary of the waiter object

        Returns: None

        """
        sal = self.salary + (hike_percent*self_salary)
        self.salary = sal
```

什么是自我？？我在代码中多次使用 self。

它用于将值传递给属性，并区分这两个对象。

Self 告诉 Python 在计算机内存的什么地方寻找 a 对象。然后 Python 改变了那个对象的值。当您调用特定的方法时，self 被隐式传入。

你注意到看起来和函数相似的方法了吗？？函数和方法看起来非常相似。它们都使用 def 关键字。它们也有输入和返回输出。区别在于方法在类的内部，而函数在类的外部。

这段代码讲述了 OOPs 以及文档字符串的使用。

# 邓德方法

Python 中的 Dunder 或 magic 方法是在方法名中有两个前缀和后缀下划线的方法。邓德在这里的意思是“双下(下划线)”。这些通常用于运算符重载。等等！！你看到上面代码中的那些了吗？？没错，__init__ 是 dunder 方法，它覆盖了默认行为。

现在让我们看看高斯分布类，并了解如何使用邓德。

```
class Gaussian():
    """ Gaussian distribution class for calculating and 
    visualizing a Gaussian distribution.
    """
    def __init__(self, mu = 0, sigma = 1):

        self.mean = mu
        self.stdev = sigma
        self.data = []

    def calculate_mean(self):
        self.mean = 1.0 * sum(self.data) /len(self.data)
        return self.mean

    def calculate_stdev(self, sample=True):
        if sample:
            n = len(self.data) - 1
        else:
            n = len(self.data)
        mean = self.mean
        sigma = 0
        for d in self.data:
            sigma += (d - mean) ** 2
        sigma = math.sqrt(sigma / n)
        self.stdev = sigma        
        return self.stdev
```

因此，如何添加两个高斯分布，如果你看到数学解释，这似乎很容易，但如何在代码中做到这一点，如果你尝试以下你会得到错误。

```
gaus_a + gaus_b = Yes we get error!!!
```

现在是邓德方法。

Python 类中有一个名为 __add__ method 的 dunder，它将帮助添加自定义对象的两个实例。这意味着我们可以通过修改或拒绝 __add__ 方法来控制两个对象相加的结果。

如果您将这段代码添加到上面的 Gaussian 类中，就会出现一些奇迹。

```
def __add__(self, other): 
    result = Gaussian() 
    result.mean = self.mean + other.mean 
    result.stdev = math.sqrt(self.stdev**2 + other.stdev**2) 
    return result
```

现在之前给你错误的代码可以正常工作了:)

这样我们就可以重写代码并改变所有的默认行为，这不是很有用吗？？？

# 遗产

在餐馆的例子中，我们看到 food dish 是一个对象，这意味着所有其他的食物都有单独的类，但是为什么要麻烦地为所有的东西建立单独的类呢，有没有更好的方法呢？？？

所以继承的概念在这里很有帮助。

我们可以有一个名为 food dish 的通用类，拥有所有食物项目共有的所有属性，并为不同的食物继承该类。现在，如果你想把这个属性添加到所有叫做季节性的食物中，而不是添加到所有的食物中，我们可以把它添加到主根类中，所有其他的类都会继承它。这样省时省力很多。

要继承一个类，我们需要编写一个通用的主类，并在子类的括号中使用这个类名。

```
class Gaussian(Distribution):
```

分布是主类，高斯(子)使用分布类。

在这篇博客中，我们已经看到了 OOP 的所有基础知识。记得在博客的开始我说过著名的包将使用 OOP 概念来构建包，所以除非你应用你的知识去使用 OOP，否则对 OOP 的学习是不完整的。

因此在下一篇博客中，我们将看到如何使用 OOP 概念用 python 创建一个包，并将其上传到 PyPI，之后您就可以使用使用 pip install 创建的包了。

**学分:** Udacity 课程。

本博客第二部分将于 2020 年 7 月 11 日在[这里](https://kirankamath.netlify.app/blog/oop-in-python-part2-make-a-python-package)发表。

感谢您阅读本博客:)

*最初发布于*[*https://kirankamath . netlify . app*](https://kirankamath.netlify.app/blog/object-oriented-programming-in-python-part1/)*。*