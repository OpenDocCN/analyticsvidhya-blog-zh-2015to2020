# 坚实的原则

> 原文：<https://medium.com/analytics-vidhya/s-o-l-i-d-principles-7caf040fab96?source=collection_archive---------13----------------------->

![](img/d5c87ac328f9b54b6ece0729997b7e22.png)

在这篇文章中，我将尝试解释复习所谓的**扎实** **原理**，以及如何以一种 Pythonic 式的方式实现它们。我们将学习如何开发可维护的。将设计模式应用于**坚实** **原则**的软件系统。

SOLID 是 OOP(面向对象编程)的基本原则，在本世纪初由 **Michael Feathers** 提出定义，并以前五项原则 **Robert C. Martin** 命名。当谈到坚实的原理时，参考**类**和**模块**。这并不意味着它们只适用于 OOP，尽管它们确实能很好地与 OOP 一起工作。

OOP 是关于通过反转关键依赖来管理依赖，以防止**刚性代码**、**脆弱代码**和**不可重用代码**。描述不可重用代码有三个术语:

*   **刚性**:当一个程序的一部分的改变会破坏另一部分时
*   脆弱:当东西在不相关的地方破碎
*   **不可移动性**:当代码不能在其原始上下文之外重用时

我们可以用**坚实的原理**解决这些问题。它被认为是面向对象编程(OOP)的基本原则，定义了**五个**设计原则。

这些原则使得软件设计更容易理解，更灵活，更易维护。

*   单一责任原则
*   **O** 笔闭合原理( **OCP** )
*   **L** 伊斯科夫替代原理( **LSP**
*   **I** 界面偏析原理( **ISP**
*   **D** 依赖反转原理(DIP)

让我们一个一个来看看这些原理。

# 1.单一责任原则

**这个原理经常被表述为:**

```
A software component(Module or Class) must have only one responsibility.
```

一个项目中的每个班级都应该专注于一项任务。因此，不要将因不同原因而改变的方法放在同一个类中。我们可以得出结论，它必须只有一个改变的理由。

在这个例子中，有一个**圆**类。它提供了一个叫做 **draw()** 的方法。

```
class Circle():
    def draw(self):
        """Draw a circle"""
        pass
```

# 2.开闭原则(OCP)

**这个原理经常被表述为:**

```
A software entities (source file, module, class, 
or function) should be open for extension but closed for modification.
```

这意味着我们应该能够添加功能，而不必从头重写一切。

在这个例子中，我们得到了一个绘制**圆**和**正方形**的应用程序。画图时，如果类型是**圆**或**方**就这样画。但是我们如果以后要按照三角形来画就违背了这个原则。

```
class Shape:
    def draw_circle(self):
        """Draw a circle"""
        pass
    def draw_square(self):
        """ Draw a square"""
        pass
    def draw_triangle(self):
        """ Draw a triangle"""
        pass
```

考虑到这种情况，下面的代码片段就是按照这个原则编写的。

```
class Shape
    def draw(self):
        """Draw a shape"""
        passclass Circle(Shape):
    def draw(self):
        """Draw a circle"""
        passclass Square(Shape):
    def draw(self):
        """Draw a square"""
        pass
```

# 3.利斯科夫替代原理

**这个原理经常被表述为:**

```
If for each object o1 of type S there is an object o2 of type T 
such that for all programs P defined in terms of T, the 
behavior of P is unchanged when o1 is substituted for o2 
then S is a subtype of T.Barbara Liskov, “Data Abstractions and Hierarchy”, 1988.
```

Liskov 替换原则指出，设计必须提供用一个子类的实例替换父类的任何实例的能力。如果一个父类能做一些事情，那么一个子类也必须能做。

在这个例子中，Vehicle 类有一个 engine 方法。但是自行车没有发动机。

```
class Vehicle:
    def __init__(self, name):
        self.name = name def engine(self):
        """A Vehicle engine"""
        pass def get_name(self):
        """Get vehicle name"""
        return f'The vehicle name {self.name}'class Bicycle(Vehicle):
    def engine(self):
        passclass Car(Vehicle):
    def engine(self):
        pass
```

所以我们根据这个规则修改代码。

```
class Vehicle:
    def __init__(self, name):
        self.name = name def get_name(self):
        """Get vehicle name"""
        return f'The vehicle name {self.name}'class VehicleWithoutEngine(Vehicle):
    passclass VehicleWithEngine(Vehicle):
    def engine(self):
        """A Vehicle engine"""
        passclass Bicycle(VehicleWithoutEngine):
    passclass Car(VehicleWithEngine):
    def engine(self):
        """A vehicle engine"""
        pass
```

# 4.界面分离原理(IPS)

**这个原理经常被表述为:**

```
A client should not be forced to implement an interface that it does not use.
```

接口隔离原则(ISP)指出，拥有许多小接口比拥有几个大接口更好，并且提供了一些我们已经反复讨论过的观点的指导原则:接口应该是小的。

在面向对象的术语中，接口由对象公开的一组方法来表示。也就是说，一个对象能够接收或解释的所有消息构成了它的接口，这也是其他客户端可以请求的。接口将类的公开行为的定义与其实现分开。

在这个例子中，我们从 Circle 和 Square 类中调用一些方法。但是它们是不必要的方法。

```
class Shape:
    def draw_circle(self):
        """Draw a circle"""
        pass def draw_square(self):
        """Draw a square"""
        pass

class Circle(Shape):
    def draw_circle(self):
        """Draw a circle"""
        pass def draw_square(self):
        """Draw a square"""
        passclass Square(Shape):
    def draw_circle(self):
        """Draw a circle"""
        pass def draw_square(self):
        """Draw a square"""
        pass
```

考虑到这种情况，下面的代码片段就是按照这个原则编写的。

```
class Shape:
    def draw(self):
        """Draw a shape"""
        passclass Circle(Shape):
    def draw(self):
        """Draw a circle"""
        passclass Square(Shape):
    def draw(self):
        """Draw a square"""
        pass
```

# 5.从属倒置原则

**这一原则通常被表述为:**

```
High-level modules should not depend on low-level 
modules. Both should depend on abstractions. Abstractions 
should not depend on details. Details should depend on 
abstractions.Or, source code dependencies should refer only to 
abstractions, not to concretions.
```

抽象必须以不依赖于细节的方式来组织，反之亦然——细节(具体实现)应该依赖于抽象。

假设我们设计中的两个对象需要协作，A 和 B。A 使用 B 的一个实例，但事实证明，我们的模块并不直接控制 B(它可能是一个外部库，或者由另一个团队维护的模块，等等)。如果我们的代码严重依赖于 B，当这种变化发生时，代码将会崩溃。为了防止这种情况，我们必须反转依赖关系:使 B 必须适应 a。这是通过提供一个接口并强制我们的代码不依赖于 B 的具体实现，而是依赖于我们定义的接口来实现的。然后，B 有责任遵守该接口。

在这个例子中， **Sports** 类是一个高级模块。**足球**和**篮球**为低级模块。

```
class Football:
    """Low-level module"""
    @staticmethod
    def play_football():
        print('play football')class Basketball:
    """Low-level module"""
    @staticmethod
    def play_basketball():
        print('play basketball')class Sports:
    """High-level module"""
    def __init__(self):
        self.football = Football()
        self.basketball = Basketball() def playing(self):
        self.football.play_football()
        self.basketball.play_basketball()
        return 'Playing...'
```

**结果**

```
In [1]: s = Sports()
In [2]: s.playing()
play football
play basketball
Out[2]: 'Playing...'
```

让我们根据**依存倒置原则**来解决问题。

```
class Football:
    """Low-level module"""
    def playing(self):
        self.__play_football() @staticmethod
    def __play_football():
        print('play football')class Basketball:
    """Low-level module"""
    def playing(self):
        self.__play_basketball() @staticmethod
    def __play_basketball():
        print('play basketball')class Playing:
    """Abstract module"""
    def __init__(self):
        self.football = Football()
        self.basketball = Basketball() def playing(self):
        self.football.playing()
        self.basketball.playing()class Sports:
    """High-level module"""
    def __init__(self):
        self.__playing = Playing()
    def plays(self):
        return self.__playing.playing()
```

**结果**

```
In [1]: sport = Sports()
In [2]: sport.plays()
play football
play basketball
```

# 摘要

在本文中，我们探索了**坚实的原则**，目的是理解干净的设计。这些原则不是一条神奇的规则，但是它们提供了很好的指导方针，在过去的项目中已经被证明是有效的，并且将使我们的软件更有可能成功，并且是当今使用的最有影响力的面向对象指导方针之一。

# 参考

*   [单一责任原则](https://blog.cleancoder.com/uncle-bob/2014/05/08/SingleReponsibilityPrinciple.html)
*   [pep-3119](https://www.python.org/dev/peps/pep-3119/)