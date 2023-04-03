# 共变，逆变，不变:为什么和如何？

> 原文：<https://medium.com/analytics-vidhya/covariant-contravariant-invariant-what-for-and-how-3df4175c4d91?source=collection_archive---------16----------------------->

![](img/966051fdcdd87f73b061e1c518788f71.png)

来自 unsplash.com

今天有许多编程语言，它们有静态类型检查、强类型系统和类型推理。这种工具允许开发人员添加额外的验证，避免编译或代码分析阶段的错误。在编程语言中，支持这些的有 [Haskell](https://www.haskell.org/) 、 [F#](https://fsharp.org/) 、 [Python](https://www.python.org/) (可选)等。类型检查的基础是类型的**协方差**、**逆变**和**不变性**的概念。在某些情况下，如下所示，这不是微不足道的，除了违反直觉。让我们来看一下这些案例，并理解为什么它们是这样设计的。

# 这是什么？

首先，我们来定义一个类型之间的关系。当你说“ *type1* 是 *type2* 的*子类型*时，这意味着 *type1* 包含了 *type2* 所有可能值的子集。简而言之，我们来介绍一下运算符 **<** 。如果*类型 1* 是*类型 2* 的子类型，那么表达式**类型 1 <类型 2** 为**真**。

```
/*pseudo code*/Type int
Type float// true, integer is a subtype of float
assert int < float// false
assert float < int
```

大多数更好的类型表达式的编程语言都有**泛型类型**(你好， [Go](https://golang.org/) )。**通用类型**是类型，你可以在其中指定内部类型。使用**类属**的常见情况是容器，如*列表*、*集合*等。

```
/*pseudo code*/Type List[T]
Type Set[T]
Type int
Type str// all elements in List should have type int
Type IntList = List[int]// all elements in Set should have type str
Type StrSet = Set[str]
```

函数是函数式语言中的第一类公民，你也可以定义一种类型的函数作为参数传递给另一个函数或从该函数返回。此函数类型被描述为**通用类型，**因为它可以有许多类型作为输入，也可以有许多类型作为输出。让我们定义函数类型**可调用**如下

```
/*pseudo code*/Callable[[<argument-types>,...], <return-type>]// function that has one int parameter and return float Callable[[int], float]// function that has two int parameters and return float Callable[[int, int], float]
```

当你面对**泛型**时，*子类*会发生什么？如果*List【int】*还是*List【float】*的子类型？还是 *List[float]* 成为 *List[int]* 的子类型？这就是**协方差**、**逆变**、**不变性**的概念。我们来声明两种类型: *type1* 和 *type2* ，而 *type2* 是 *type1* 的一个子类型

```
/*pseudo code*/Type type1
Type type2assert type2 < type1 // true
```

然后泛型类型 *GenericType* 被调用:

*   **协变**，如果 *GenericType[type1]* 是 *GenericType[type2]* 的子类型

```
assert GenericType[type1] < GenericType[type2]
```

*   **逆变**，如果 *GenericType[type2]* 是 *GenericType[type1]* 的子类型

```
assert GenericType[type1] > GenericType[type2] // or
assert GenericType[type2] < GenericType[type1]
```

*   **不变量**，如果以上两者都不成立

我在[人教版 483 中发现了一个很好的数学例子——理论类型提示](https://www.python.org/dev/peps/pep-0483/#covariance-and-contravariance)对于**协方差**、**逆变**和**不变性**的定义:

```
def cov(x: float) -> float:
    return 2*xdef contra(x: float) -> float:
    return -xdef inv(x: float) -> float:
    return x*x
```

如果 *x1 < x2* ，那么总是 *cov(x1) < cov(x2)* ，和*contra(x2)<contra(x1)*，但是关于 *inv 什么也说不出来。*

**协方差**、**逆变**、**不变性**类型的例子:

*   *ImmutableList【T】*是**协变**。如果 *int* 是 *float* 的子类型，那么*immutable list【int】*是*immutable list【float】*(**协方差**)的子类型

```
assert ImmutableList[int] < ImmutableList[float] // true
```

*   *可变列表【T】*是**不变量**。如果*苹果*是*水果*的子类型，那么你不能说*可变列表【苹果】*是*可变列表【水果】*的子类型，反之亦然。不变性是可变类型的常见行为

```
Type Fruit
Type Appleassert Apple < Fruit // trueassert MutableList[Fruit] < MutableList[Apple] // false
assert MutableList[Apple] < MutableList[Fruit] // false
```

# 为什么可变类型是不变的？

事实上，为了检测可变类型的类型检查错误，它们应该是**不变的**。让我们用上一段中一个非常简单的例子来看看这个。你有一个基础类型*水果*和两个子类型:*苹果*和*香蕉*

```
Type Fruit
Type Apple
Type Bananaassert Apple < Fruit // true
assert Banana < Fruit // true
```

对于我们的例子，你需要一个可变的**泛型类型***【T】*。为基本类型*水果*和*苹果*定义一个可变容器类型

```
Type MutableList[T]Type MutableList[Fruit]
Type MutableList[Apple]// define a varibles for list of Apples and list of Fruits
apples :: MutableList[Apple]
fruits :: MutableList[Fruit]
```

让我们想象一下，那个**泛型** *可变列表【T】*是**协变**。这意味着，那

```
// if Apple is a subtype of Fruit
assert Apple < Fruit // true// then
assert MutableList[Apple] < MutableList[Fruit] // true, covariant// and we assign apples to fruits
fruits = apples
```

太好了！从类型的角度来看，一切都是正确的。水果是水果列表和*的列表，香蕉*是*水果*的子类型，所以我们可以在水果列表中添加一个香蕉

```
// fruits is MutableList[Fruit]
// and Banana is a subtype of Fruit
assert Banana < Fruit// we can add banana to a list
banana::Banana
fruits.append(banana)
```

看起来还是正确的。水果列表包含*香蕉*和*苹果*种类。但是我们有变量*苹果*，也就是指向同一个列表，它的类型是*可变列表【苹果】*而*香蕉*不是*苹果*的子类型。但是经过我们的操作后*易变列表【苹果】*包含*香蕉*类型。**协变**可变类型破坏了类型检查，这就是为什么我们不能让可变类型**协变**！

# 可调用类型协方差/逆变

**可调用的**类型是**共变的**返回类型。也就是说，对于两个**可调用的**类型:*可调用[[type]，rtype1]* ，*可调用[[type]，rtype2]* 。如果 *rtype1* 是 *rtype2* 的子类型，那么 *Callable[[type]，rtype1]* 是 *Callable[[type]，rtype2]* 的子类型

```
assert int < float // true
assert Callable[[str], int] < Callable[[str], float] //true
```

为了实现这一点，让我们假设您有一些 *map* 函数，它期望 *Callable[[str]，float]* 作为参数。在*映射内*函数开发者可能会调用你的可调用函数，并期望它返回一个 *float* 类型。既然 *int* 是 *float* 的一个子类型，那么把 *Callable[[str]，int]* 传递给你的 *map* 函数就可以了。

但是**可调用的**类型是**自变量类型中的逆变**

```
assert Callable[[float], str] < Callable[[int], str] // true
// if
assert int < float // true
```

比如这里你需要三种类型: *int* ， *float* 和 *complex* 。并且让我们声明 *int* 是 *float* 的子类型， *float* 是 *complex* 的子类型

```
Type int
Type float
Type complexassert int < float < complex
```

你还需要一个函数 *map* ，它有一个参数——类型为 *Callable[[float]，str]* 的函数。让我们想象一下，如果你将一个 *Callable[[int]，str]* 传递给函数 *map* ，会发生什么。作为一个*映射*作者，我期望**可调用**的第一个参数是*浮动*。所以我试着把我的 *float* 传递给 **Callable** ，那是什么？我不能，因为我将 *int* 作为第一个参数类型，并且 *float* 不是 *int* 的子类型

```
assert float < int // false
```

这就是为什么**可调用的**类型在参数中不能是**协变的**！另一方面，将 *Callable[[complex]，str]* 作为参数传递给函数 *map* 是安全的。 *float* 和 *int* 类型都可以传递给 *complex* 类型。它是安全的。

我希望在理解了这些概念之后，作为开发者的你将开始设计更少错误的系统，并使用你最喜欢的程序语言类型系统的所有能力。感谢您的宝贵时间！