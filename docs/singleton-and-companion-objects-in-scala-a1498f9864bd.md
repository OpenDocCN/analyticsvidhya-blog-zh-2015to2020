# Scala 中的单例对象和伴随对象。

> 原文：<https://medium.com/analytics-vidhya/singleton-and-companion-objects-in-scala-a1498f9864bd?source=collection_archive---------9----------------------->

Scala 是一种面向对象的编程语言，它定义了单例对象和伴随对象，使它的运行令人愉快..

Scala 比 Java 更面向对象，所以 Scala 不包含任何静态关键字的概念。Scala 拥有 singleton 对象，而不是静态关键字。单例对象是定义一个没有类的对象。单例对象提供了程序执行的入口点。如果您没有在程序中创建一个 singleton 对象，那么您代码会成功编译，但不会给出输出。因此，您需要一个 singleton 对象来获取程序的输出。使用 object 关键字创建单例对象。

![](img/27852b054b8b75a2c2404742a4e44c52.png)

由 [C D-X](https://unsplash.com/@cdx2?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

**Scala 中的单例对象:-**

Scala 作为一种面向对象的编程语言，能够创建一个无需类就能定义其成员的对象。

> 一个 **单例对象**没有与之相关联的类，即它没有任何类。它是使用 object 关键字而不是 class 关键字定义的，并且是一个实例，而不是一个蓝图，因此它不需要外部调用来执行其方法。

**重点:-**

> Scala 支持静态成员，但与 Java 的方式不同。Scala 提供了一种替代方法，叫做单例对象。单例对象类似于普通类，除了它们不能使用 **new 关键字**实例化。

**关于单例对象的要点:-**

1-使用 object 关键字创建。
2-在单一对象的情况下，无法创建实例。
3-允许继承，即可以扩展类和性状。
4-对于 singleton 对象的多余成员，我们将用 Singleton 对象的名字点成员名。单一对象中的方法是全局可访问的。
6-不允许在 singleton 对象的主构造函数中传递参数。7-在 Scala 中，一个 main 方法总是出现在 singleton 对象中。

**为什么 Scala 中的 Singleton 对象:-**

每个程序都需要一个执行起点。在 OOPS 中，类需要对象来执行。但是需要首先执行 main()方法来调用该类的其他成员。

对于在 scala 中执行 main()方法，许多面向对象的编程语言使用 static 关键字，但是 scala 编程语言没有 static 关键字。这就是为什么在 scala 中我们使用单例对象来定义 main 方法。

**可以使用的情况:-**

> 假设您有一个方法可以识别输入的密码是弱密码还是强密码，那么您可以在一个 Singleton 对象中创建这个方法，并与您的团队成员共享它，以便在需要时使用它。

**语法:-**

```
object singleton_objname {
	    // object code , member functions and fields. 
    }
```

示例:-

```
**object** **summing**{
    **var** a **=** **56**
    **var** b **=** **21**
    **def** sum()**:** **Int** ={
        **return** a+b;
    }
}
**object** **Main** 
{ 
    **def** print(){
        printf("The sum is : "+ **summing**.sum());
    }
	**def** main(args**:** **Array**[**String**]) 
	{ 
        print();
	} 
}
```

**输出**

```
The sum is : 77
```

# Scala 中的伴随对象:-

如果一个类和一个单例对象同名。然后，这个类被称为伴随类，单例对象被称为单例对象。

类和对象都在同一个程序文件中定义。

```
**class** **companion**{  
    **var** a **=** **23**;
    **var** b **=** **78**;
    **def** sum(){
        println("The sum is: "+ (a+b));
    }
}  

**object** **companion**{  
    **def** main(args**:Array**[**String**]){  
        **new** companion().sum(); 
    }  
}
```

这个程序是为了说明一个**伴星**的概念。这里，程序是用来求给定整数的和的。为了计算总和，我们在同伴类中有一个方法 sum。我们将使用点操作符从伴随对象中调用这个方法。

谢谢！推荐我在 Scala 上的帖子的读者，要了解更多有趣的话题，你可以查看我以前的帖子..

[](https://duke-lavlesh.medium.com/basics-of-functions-and-methods-in-scala-534a1692b8d6) [## Scala 中函数和方法的基础。

### 当程序变大时，你需要某种方法把它们分成更小的、更易管理的部分。为了分开…

duke-lavlesh.medium.com](https://duke-lavlesh.medium.com/basics-of-functions-and-methods-in-scala-534a1692b8d6)