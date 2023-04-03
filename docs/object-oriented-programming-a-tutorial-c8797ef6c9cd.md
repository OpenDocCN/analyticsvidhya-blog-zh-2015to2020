# 面向对象编程—教程

> 原文：<https://medium.com/analytics-vidhya/object-oriented-programming-a-tutorial-c8797ef6c9cd?source=collection_archive---------25----------------------->

在编程时，人们可能会发现自己一遍又一遍地为相似的对象使用相同的函数，这些对象都具有相同的蓝图、相似的特征和共同的函数。面向对象编程旨在通过将适用于一组相似“对象”的信息和功能聚集在一个地方来解决这种效率杀手问题。为了让这一点更清楚，我们将通过一个简单的例子来看这个想法；狗。

OOP 的核心是类；这基本上是指定给该类的所有对象的框架。所以，狗将成为我们的阶级:

```
class Dogs:
    pass#first letter of the class capitalized due to naming convention
#insert “pass” as the placeholder for future code
```

现在，为了给类添加内容，我们将添加一个实例方法。实例方法与函数相同，但它是在类中定义的。所以在这个例子中，我们可以创建两个实例方法，它们适用于 Dogs 类中的任何对象；叫着吃。

```
class Dogs:
    def bark(self):
        return "Wooof"
    def eat(self):
        return "Yum!"
```

注意实例方法中的参数“self”。这意味着以后当我们创建该类的唯一实例时，作为实例的对象将作为参数传递。“自我”一词的使用是约定俗成的。

现在，为了将这个类及其关联的实例方法用于一个单独的对象，我们需要创建这个类的一个实例。让我们为我的狗 Ringo 实例化这个类。

```
ringo = Dogs()
ringo.bark()
'Wooof'
```

请注意，当我们为 Ringo 创建实例时，我们将他的变量设置为“Dogs()”，这是该类的一个唯一实例，而不仅仅是“Dogs”，即该类本身。当我们接着调用实例方法时，bark()，Ringo 将作为参数传入，我们得到方法的输出；“呜呜呜”。

现在让我们说，我们想要添加对所有与类 Dogs 一致的对象都成立的一般特征。为此，我们在类中定义变量，如下所示:

```
class Dogs:
    legs = 4
    tail = 1
    ears = 2
    def bark(self):
        return "Wooof"
    def eat(self):
        return "Yum!"
```

现在要获取这些属性，我们可以用我们的实例 Ringo 来引用这个变量。

```
print(ringo.legs)
4
```

现在，假设我们开始一项遛狗业务，从三只狗开始，我们希望将每只狗的品种和体重添加到它们的实例中。我们首先为每只狗创建一个实例，然后为它们各自的品种和体重添加两个实例变量:

```
riley = Dogs()
riley.breed = 'Golden Retriever'
riley.weight = '60lbs'tony = Dogs()
tony.breed = 'German Shepherd'
tony.weight = '70lbs'coconut = Dogs()
coconut.breed = 'Maltese'
coconut.weight = '10lbs'
```

然后，我们可以将所有这些狗添加到一个列表中，然后在一个 for 循环中打印所有的品种名称，如下所示:

```
all_dogs = [riley, tony, coconut]
for dog in all_dogs:
    print(dog.breed)Golden Retriever
German Shepherd
Malteseriley.bark()
'Wooof'
```

正如您所看到的，实例现在有了自己独特的手动分配的特征，同时保持了它们被分配到的类的实例方法。

因此，您可以看到 OOP 在为利用相似功能的对象建立蓝图方面是如何有用的，以及您可以如何基于对象的独特特征定制它们的独特实例。我希望这有助于为任何试图掌握 OOP 或者需要复习 OOP 的人阐明这个基础话题。

干杯