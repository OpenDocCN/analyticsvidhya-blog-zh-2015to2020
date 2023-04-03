# python 中的下划线现象

> 原文：<https://medium.com/analytics-vidhya/underscore-phenomena-in-python-fdebcfdd45f0?source=collection_archive---------10----------------------->

![](img/60837add69647cc959e399ea9ec37931.png)

我第一次接触下划线是在我高中开始学习 c++(T1)的时候。我的第一堂课是变量的命名惯例，有趣的是下划线是唯一可以在变量名中使用的非字母数字字符。后来在大学里，当我转向 Python 时，我发现下划线是如何以如此多的不同方式实现不同目标的，这非常有趣。
这篇文章将有以下主题:

*   单前导下划线
*   单个尾随下划线
*   单个独立下划线
*   双前导下划线
*   双前导和尾随下划线

**_ single _ leading _ 下划线:**

许多在线资源将前导下划线定义为 python 中受保护的访问修饰符。但事实并非如此。Python 没有类似于 Java 或 C++等其他面向对象语言的访问修饰符。
让我们看一个例子:

```
#file1.py
class Example:def __init__(self):
  self.foo = 1
  self._bar = 2>>> from file1 import Example
>>> example = Example()
>>> print(example.foo, example._bar)
1 2
```

有趣的是，在方法名中有一个前导下划线确实会以某种方式改变它的行为。

```
# file1.py
def first_method():
    print("output --- First Method")def _second_method():
    print("output --- second method")
```

让我们将这些函数导入到另一个文件中。

```
from file1 import *first_method()
_second_method()
```

输出:

```
output --- First Method
Traceback (most recent call last):
  File "file2.py", line 4, in <module>
    _second_method()
NameError: name '_second_method' is not defined
```

产生上述输出的原因是，如果我们使用通配符导入从模块中导入所有名称，Python 将不会导入带有前导下划线的名称。PS:使用通配符导入从来都不是一个好主意。

**单 _ 尾 _ 下划线 _:**

尾随下划线的任何特殊用法都完全是约定俗成的，而不是规则。有时最合适的变量名已经被 Python 语言中的一个
关键字所采用。在这种情况下，您可以附加一个下划线来消除命名冲突:

```
 >>> def make_object(name, class):
SyntaxError: “invalid syntax”
>>> def make_object(name, class_):
        pass
```

**_:**

单个下划线通常用作占位符变量。但这完全是惯例。例如，在下面的循环中，我们不需要访问运行索引，我们可以使用“_”来表示它只是一个临时值:

```
for _ in range(32):
    print('Hello, World.')
```

或者在下面这段代码中，我们只需要打印列表中每个元组的第一个和第三个属性，这样我们就可以 ***忽略*** 第二个。

```
list_ = [('red', 'suzuki', '2016'), ('Blue', 'suzuki', '2017'), ('Green', 'Sedan', '2017') ]for color, _, make in list_:
    print(color, make)
```

输出:

```
red 2016
Blue 2017
Green 2017
```

**_ _ 双前导下划线:**

在上述三种命名模式中，我们到目前为止所涉及的一切都是约定俗成的。但是现在事情要改变了。

“双下划线前缀导致 Python 解释器重写属性名
，以避免子类中的命名冲突。”

让我们做一些实验:

```
class Test:def __init__(self):
  self.foo = 10
  self._bar = 20
  self.__baz = 30test = Test()
print(test.foo, test._bar, test.__baz)
```

输出:

```
Traceback (most recent call last):
  File "file1.py", line 10, in <module>
    print(test.foo, test._bar, test.__baz) 
AttributeError: 'Test' object has no attribute '__baz'
```

Python 解释器无法识别测试类中的任何属性 __baz。让我们试着深入了解一下。

```
>>> t = Test()
>>> dir(t)
['_Test__baz', '__class__', '__delattr__', '__dict__',
'__dir__', '__doc__', '__eq__', '__format__', '__ge__',
'__getattribute__', '__gt__', '__hash__', '__init__',
'__le__', '__lt__', '__module__', '__ne__', '__new__',
'__reduce__', '__reduce_ex__', '__repr__',
'__setattr__', '__sizeof__', '__str__',
'__subclasshook__', '__weakref__', '_bar', 'foo']
```

dir()给出了一个对象的属性列表。让我们看一下名单。自我。foo 和自我。_bar 变量在属性列表中显示为未修改。然而，与自我。_ _ 巴兹的东西看起来有点奇怪。没有属性 __baz。但是到底发生了什么呢？
如果你仔细观察，你会发现这个对象上有一个名为 _Test__baz
的属性。这是 Python 解释器
应用的*名 mangling* 。这样做是为了防止变量在
子类中被覆盖。

**_ _ double _ leading _ trailing _ 下划线 __:**

令人惊讶的是，如果一个名字以
开头并以双下划线结尾，那么就不会出现名称混淆。

```
class PrefixPostfixTest:
def __init__(self):
    self.__bam__ = 42
>>> PrefixPostfixTest().__bam__
42
```

然而，具有前导和尾随双 under-
分数的名称在语言中被保留用于特殊用途。这条规则涵盖了像对象构造函数的 __init__ 或 __call__ 这样的事情，以使 o B-
对象可调用。因此，为了避免任何麻烦，最好不要使用这样的名称，以避免与任何未来的 python 版本发生命名冲突。