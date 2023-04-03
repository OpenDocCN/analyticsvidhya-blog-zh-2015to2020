# 使用 JavaScript 类语法的示例

> 原文：<https://medium.com/analytics-vidhya/an-example-of-using-javascript-class-syntax-ebadd1cc4d5c?source=collection_archive---------9----------------------->

我希望这篇文章是关于采用伪经典继承风格编写的 JavaScript 代码，并将其转换为使用 ES6 类。

伪经典继承我就不太深入了。相反，我将从一些使用伪经典继承的示例代码开始。下面是我们的结构，下面是代码:

![](img/b0e4f72963eb7adeb8a8ba86ba4ca181.png)

```
function Animal() {}
  Animal.prototype.eat = function() {
  console.log("I'm eating");
}Animal.prototype.sleep = function() {
  console.log("I'm sleeping");
}function Mammal(legs) {
  this.legs = legs;
  this.warmBlooded = true;
}Mammal.prototype = Object.create(Animal.prototype);
Mammal.prototype.constructor = Mammal;
Mammal.prototype.giveLiveBirth = function() {
  //the gift of life
}function Dog(name, breed) {
  Mammal.call(this, 4);
  this.name = name;
  this.breed = breed;
}Dog.prototype = Object.create(Mammal.prototype);
Dog.prototype.constructor = Dog;function Cat(name, breed) {
  Mammal.call(this, 4);
  this.name = name;
  this.breed = breed;
}Cat.prototype = Object.create(Mammal.prototype);
Cat.prototype.constructor = Cat;const nocturnalMixin = {
  isNocturnal: true
}Object.assign(Cat.prototype, nocturnalMixin);let tito = new Cat("Tito", "American shorthair")
console.log(tito.isNocturnal) function Reptile() {
  Animal.call(this);
}Reptile.prototype = Object.create(Animal.prototype);
Reptile.prototype.constructor = Reptile;
Reptile.prototype.shed = function() {
  console.log("I'm molting");
}function LeopardGecko() {}
LeopardGecko.prototype = Object.create(Reptile.prototype);
LeopardGecko.prototype.constructor = LeopardGecko;
Object.assign(LeopardGecko.prototype, nocturnalMixin);let eco = new LeopardGecko()
console.log(eco.isNocturnal)
```

我们的 **Animal** 构造函数没有任何属性，但是在它的原型上有几个方法，因为这些行为对所有动物都是通用的。

**哺乳动物**是**动物**的一种。哺乳动物有腿，是温血动物。它们还可以产生 live birth，我已经排除了这个实现，但是任何使用**哺乳动物**构造函数创建的对象都可以访问这个方法。

**狗**是**哺乳动物**的一种。**狗**有名字，有品种。它们通常有 4 条腿，所以默认情况下我已经包括在内了。

以上所有代码都演示了伪经典继承。用子类型创建的对象从称为超类型的其他构造函数继承方法和属性。这种继承方式类似于典型的 OOP 语言使用类来实现继承。

介绍**爬行动物**和**豹壁虎**构造器给了一个展示**混合**的机会。

我们的 **nocturnalMixin** 将允许我们将 **isNocturnal** 属性添加到由我们选择的指定构造函数创建的对象中。

在我们的具体例子中，**豹皮壁虎**是夜行动物，而**猫**是夜行动物，但**狗**不是。但它们都继承的唯一构造函数是动物。

并非所有的动物都是夜间活动的。我们可以给所有这些不同种类的哺乳动物加上夜间活动的特性，但是这是重复的。而且我们不能在**哺乳动物**构造函数本身上定义它。更不用说不是所有的爬行动物都是夜间活动的。那么我们如何将它包含在我们的代码中呢？

```
Object.assign(Cat.prototype, nocturnalMixin);
Object.assign(LeopardGecko.prototype, nocturnalMixin);
```

**Object.assign()** 将属性和方法从源对象复制到目标对象。在这种情况下，我们将混合中的属性添加到 **Cat.prototype** 对象和 **LeopardGecko.prototype** 对象。

现在任何用**猫**或**豹壁虎**构造器创建的对象都将是夜间活动的，而我们的**狗**不受影响。我们可以放置任何其他原型对象来代替 **Cat.prototype** 或 **LeopardGecko.prototype** 。 **Object.assign()** 的功能不仅限于 mixins。

解释这一点很重要，因为我们也在 JavaScript ES6 类中使用 mix-in。

现在我们准备将构造函数转换成类。

```
class Animal {
  eat() {
    console.log("I'm eating");
  }
  sleep() {
    console.log("I'm sleeping")
  }
}class Mammal extends Animal {
  constructor(legs) {
    super();
    this.legs = legs;
    this.warmBlooded = true;
   }

  giveLiveBirth() {
    //ah yes what a miracle
  }
}class Dog extends Mammal {
  constructor(name, breed) {
    super(4);
    this.name = name;
    this.breed = breed;
  }
}const nocturnalMixin = {
  isNocturnal: true
}class Cat extends Mammal {
  constructor(name, breed) {
    super(4);
    this.name = name;
    this.breed = breed;
  }
}Object.assign(Cat.prototype, nocturnalMixin);let tito = new Cat("Tito", "American Shorthair");
console.log(tito.isNocturnal);class Reptile extends Animal {
  shed() {
    console.log("I'm molting");
  }
}class LeopardGecko extends Reptile {}Object.assign(LeopardGecko.prototype, nocturnalMixin);let eco = new LeopardGecko()
console.log(eco.isNocturnal);
```

不是用 **Object.create()** 创建新的原型对象，而是用 **extends** 关键字为我们处理这些。当一个类扩展另一个类时，它继承了在所需类的原型上定义的所有属性和方法。

我们也不必在它们的原型上定义我们的类方法，我们可以在类中包含我们的定义。重要的是要注意，要定义一个类的方法，我们必须使用关键字 static，就像这样:

```
class Computer {
  static powerOn() {
    console.log("beep boop");
  }
}
```

类的最大区别之一是构造函数方法的使用。当我们使用类实例化一个新对象时，构造函数方法执行。JavaScript 会自动完成这项工作。

我们并不总是需要定义一个**构造函数()**方法。当我们希望用该类创建的新对象具有某些属性时，或者当我们希望它在它所继承的构造函数中定义属性时，我们确实需要定义它。如果这两个条件都不成立，我们可以完全省略构造函数方法，就像我们对动物类所做的那样。

类的另一个大区别是在构造函数方法中使用了 **super()** 函数。就像我说的，如果你试图继承的类没有构造函数，就没有必要包含构造函数。

然而，如果父类确实有一个接受参数的构造函数，并且您希望用继承类创建的任何对象都在父类的构造函数方法上定义了属性(并分配给您提供的参数)，那么您将需要 **super()** 函数。

**super()** ，在构造函数方法内调用时，执行父类的构造函数。例如，如果我想让我的狗有 4 条腿并且是温血动物，我需要在我的 **Dog** 类构造方法中调用 **super()** 。或者，如果我想添加额外的属性，如名称和品种，我将需要调用 **super()** ，并且我需要在定义名称和品种属性之前调用 **super()** ，否则我将得到一个错误消息。

对于伪经典继承，当我们想要重用一个构造函数时，我们必须包含这个。例如在我们的**狗**构造函数中，我们调用了**哺乳动物**构造函数，并在**中传递了这个**和 4。我们必须包含**这个**，因为我们想要指定**哺乳动物**构造函数是用新的**狗**对象作为其执行上下文来执行的。换句话说，我们想要一个新的 **Dog** 对象，我们想要它有 4 条腿，所以我们指定 provide **this** 作为**哺乳动物. call()** 的执行上下文，然后我们提供**哺乳动物**构造函数所期望的参数，也就是腿。

**super()** 类似，但不需要我们输入**这个**作为自变量。执行上下文已经设置为正在创建的新对象。如果你试图添加这个作为参数，你会得到一个错误。

我们的混合工作方式与第一个代码示例非常相似。我们使用 **Object.assign()** ，将目标原型对象作为第一个参数传递，将源对象作为第二个参数传递。现在，从 **LeopardGecko.prototype** 和 **Cat.prototype** 继承的所有对象都将有一个**is non turnal**属性设置为 true。