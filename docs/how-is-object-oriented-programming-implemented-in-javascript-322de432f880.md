# JavaScript 中面向对象编程是如何实现的？

> 原文：<https://medium.com/analytics-vidhya/how-is-object-oriented-programming-implemented-in-javascript-322de432f880?source=collection_archive---------17----------------------->

![](img/04c45ae1906f4dc7313038fbe991dd16.png)

**OOP 概述** 面向对象编程(OOP)是一种设计和构造代码的方式。对象本身包含数据或信息，编程的美妙之处在于对数据进行操作，以便我们可以读取数据、创建额外的数据并添加到集合中、更改和修改现有数据，甚至删除数据。这在最简单的层面上描述了 OOP，你可以想象我们可以做得更多。让我们通过一些例子更深入地探讨一下。

```
let name = “Senna”
let birthday = “May 6 2015”

function sayName(name) {
 return `Hi! My name is ${name}
}function sayBirthday(birthday) {
 return `My birthday is ${birthday}!`
}sayName(name) *# => Hello, my name is Senna.*
sayBirthday(birthday) *# =>* My birthday is May 6 2015!
```

按照上面的代码，狗的名字存在，狗的生日存在，并且可以作用于狗的不同函数存在，但是它们没有被捆绑在一起。它们都是独立的部分，基本上彼此相邻，没有*将*彼此关联，也没有关联到我们的代码中。现在想象一下，我们想添加更多有自己名字和生日的狗，然后我们想添加猫和它们的名字以及最喜欢的猫玩具。然后，假设我们想要添加所有者，并将他们与每种动物相关联。你可以想象，如果有这样的系统，或者缺少这样的系统，事情会变得非常混乱，很难跟踪。

幸运的是，有一个设计我们代码的系统来帮助管理这样复杂的系统！回车，**O**O**O**O**P**编程。通过遵循 OOP 原则，你可以构建你的代码，使它以你想要的方式关联起来，并且*知道*它的关联！

让我们在例子中实现 OOP 原则。首先让我们做一个类:

```
class Dog {}
```

一个类本质上是一个对象的模板或蓝图。我倾向于认为，如果我要做超过 1 英镑的东西，我应该考虑建立一个类。如果我们想要制造一只以上的狗，这是我们的计划，那么这个类将派上用场。

接下来，让我们在这个类中添加相同的方法。

```
class Dog { sayName(name) {
   return `Hi! My name is ${name}`
  } sayBirthday(birthday) {
   return `My birthday is ${birthday}!`
  }}
```

请注意，您现在不需要在 sayName 和 sayBirthday 前面写“function”——通过将它放入一个类中，就不再需要标识符了。引擎盖下都处理好了。

现在，让我们创建 Dog 类的一个实例，将其设置为一个变量:

```
let bella = new Dog 
bella *# => Dog {}*
```

太好了！我们刚刚创建了 dog 类的一个实例，bella。但它是一个空的对象。里面什么都没有。如果我们想创造贝拉，同时给她一个名字和生日，会怎么样？我们可以用构造函数方法来实现:

```
class Dog {constructor(name, birthday) {
    this.name = name;
    this.birthday = birthday;
  }sayName(name) {
   return `Hi! My name is ${this.name}`
  }sayBirthday(birthday) {
   return `My birthday is ${this.birthday}!`
  }
}
```

你不需要担心过于熟悉构造函数方法和“this”来开始理解这里发生的 OOP 原理。简单地说，构造函数方法处理对象实例的初始化(下面的例子)。你可以认为“这个”指的是“我正在调用它的对象”。构造函数方法允许我们这样做:

```
**let** bella **=** **new** Dog("Bella", "April 16 2007")bella *# => Dog {name: "Bella", birthday: "April 16 2007"}*
```

现在，当我们返回 bella 时，我们得到的不是一个空对象— *Dog {} —* 而是一个里面有名字和生日的对象。我们找到了一种方法来创建一只狗的特定实例，并添加与其相关联的键和值。感谢构造者！

![](img/cf07801adee53a872720cce32d522591.png)

现在我们用这种方式创建了一个类，实际上我们可以用存储在对象中的数据做更多的事情。

我们可以调用一个方法，而不用在括号中传递任何参数:

```
bella.sayName()
*# => "Hi! My name is Bella"*
```

OOP 允许我们以一种更平滑、更有效的方式与数据交互。现在，如果我们想添加 3 个新的方法来允许所有的狗叫，挖和吃，我们只需要添加一次，到狗类。然后，狗的所有实例——塞纳、贝拉和我们想要创建的任何人——将自动拥有这 3 个新方法。

虽然 OOP 不仅仅是在 JavaScript 中实现类，它还举例说明了实现 OOP 原则的一种方式，以帮助我们阅读代码、共享代码、更新和添加更改，使我们的代码在整体上更有组织性。