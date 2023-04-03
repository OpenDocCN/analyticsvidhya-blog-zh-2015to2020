# Python 中的 Duck 类型— 3

> 原文：<https://medium.com/analytics-vidhya/duck-typing-in-python-3-b52e3fbe340?source=collection_archive---------11----------------------->

# 让我们来学习使用 Python 中的鸭子类型

![](img/20ecfb69ca3ca16ee6b185d3aa7c8939.png)

照片由[佐里夫](https://unsplash.com/@zoeeee_?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# 鸭子打字

对于经验丰富的程序员来说，Duck typing 应该不是一个陌生的概念。对于新手来说，这可能听起来是一个有趣的短语。鸭子和编程有什么关系？

从回顾的角度来看，这个概念改编自以下表达式，作为[溯因推理](https://en.wikipedia.org/wiki/Abductive_reasoning)的一种形式，你可以在[维基百科](https://en.wikipedia.org/wiki/Duck_test)上找到。

> 如果它长得像鸭子，游起来像鸭子，叫起来像鸭子，那么它很可能就是一只鸭子。

我们现在不需要把这个表达式和编程联系起来，但是我们应该已经注意到这个表达式和我们如何识别一只鸭子有关。本质上，我们不需要对感兴趣的动物进行基因组测序就能知道它的身份。我们不是接近内在因素，而是根据它的外在表现和行为得出结论。

# 动态与静态类型

duck typing 的概念主要被支持动态类型的编程语言所采用，比如 Python 和 JavaScript。在这些语言中，一个共同的特征是我们声明变量而不需要指定它们的类型。

```
>>> a = 2000>>> type(a)<class ‘int’>>>> a = ‘Dynamic Typing’>>> type(a)<class ‘str’>>>> a = [1, 2, 3]>>> type(a)<class ‘list’>
```

从上面的代码片段中可以看出，我们最初给变量`a`赋了一个整数，使其成为`int`类型。后来我们给同一个变量赋了一个字符串和一个数字列表，类型分别变成了`str`和`list`。解释器没有抱怨同一个变量的数据类型的变化。

相比之下，许多其他类型的编程语言都以静态类型为特色，比如 Java 和 Swift。当我们声明变量时，我们需要明确这些变量的数据类型。后来，如果我们希望改变数据类型，编译器不允许我们这样做，因为它与初始声明不一致。

# 概念示例

在上一节中，我们已经提到 Python 是一种动态类型语言，正如涉及内置数据类型的最基本的例子所示。但是，我们可以进一步将动态类型应用于自定义数据类型。下面我们来看一个概念性的例子。

>>>类鸭:……def swim _ 嘎嘎(自):

… print(“我是一只鸭子，我会游泳，还会嘎嘎叫。”)…

> > > class robotic bird:……def swim _ quak(self):

…

print("我是一只机器鸟，我会游泳，还会嘎嘎叫。")

…

>>>类鱼:

…

定义游泳(自己):

…

print("我是一条鱼，我会游泳，但不会呱呱。")

…

> > > def 鸭 _ 测试(动物):

…

animal.swim _ 嘎嘎()

…

> > > duck_testing(Duck())我是一只鸭子，我会游泳，会嘎嘎叫。

> > > duck_testing(RoboticBird())我是一只机器鸟，我会游泳，会嘎嘎叫。

>>>鸭子 _ 测试(Fish())

回溯(最近一次呼叫):

文件<stdin>，第 1 行，在<module>中</module></stdin>

duck_testing 中文件“<stdin>”的第 2 行</stdin>

attribute error:“Fish”对象没有属性“swim _ quack”

在上面的代码片段中，我们可以看到`Duck`类的一个实例当然可以游泳和嘎嘎叫，正如成功调用`duck_testing`函数所反映的那样。对于`RoboticBird`类也是如此，它也实现了所需的`swim_quack`函数。然而，`Fish`类没有实现`swim_quack`函数，导致它的实例没有通过 duck 测试评估。

通过这些观察，我们应该理解这里 duck 类型的一个基本符号。当我们为特定目的使用自定义类型时，相关功能的实现比数据的确切类型更重要。在我们的例子中，尽管一只机器鸟不是一只真正的鸭子，但是它对`swim_quack`函数的实现“使”它成为一只鸭子——一种会游泳和嘎嘎叫的动物。