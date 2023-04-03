# Scala 中函数和方法的基础。

> 原文：<https://medium.com/analytics-vidhya/basics-of-functions-and-methods-in-scala-534a1692b8d6?source=collection_archive---------7----------------------->

当程序变大时，你需要某种方法把它们分成更小的、更易管理的部分。对于划分控制流，Scala 提供了一种所有有经验的程序员都熟悉的方法:将代码划分成函数。事实上，Scala 提供了几种方法来定义 Java 中没有的函数。除了作为某个对象的成员
的函数的方法之外，还有嵌套在函数、函数文字和函数值中的函数。

Scala 是一种函数式编程语言，它包含了作为一级值和方法的两种函数，既有相似之处，也有不同之处。函数和方法都是可重用代码块，也用于将重复的代码存储在一个位置，这使得函数调用执行特定的特定任务。它们还使代码更容易调试和修改。

![](img/cf4761333a87e562f92e183ac1d3bee7.png)

照片由[Cookie Pom](https://unsplash.com/@cookiethepom?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

> 函数是一组组合在一起执行特定任务的语句。代码可以在逻辑上划分为独立的功能，其中每个功能都分配有特定的任务。
> 
> Scala 中的函数是一个完整的对象，可以赋给变量，而 Scala 中的方法是类的一部分，有名字、签名、字节码和注释。函数名可以包含++、+、-、–等字符。

然而，函数是一个在变量中初始化的对象，但是方法以“def”关键字开始，后面是方法名、参数列表、带有返回值的方法体。

**你将学习以下主题:**

1-方法声明和定义
2-方法调用
3-带命名参数的方法
4-默认参数值
5-变长参数
6-递归函数
7-匿名函数

**方法声明和定义:-**Scala 中的方法由以下部分开始:

```
def 'method_name' ('parameters':'return_type_parameters') : ('return_type_of_method') = {
    'method_body'
     return 'value'
}
```

> Scala 中的方法从以下部分开始:

1- **'def':** 用于声明方法的关键字。
2-**‘方法名’:**是你的方法名，小写。
3- **'parameters':** 是方法参数，可以没有参数，也可以只有一个参数，当有多个参数时，用逗号分隔。
**4-' return _ type _ of _ parameters ':**需要根据' parameters_list '的数据类型进行匹配，是必选的
**5-' return _ type _ of _ method ':**是可选的，但默认返回' Unit '，但值可以用' return '关键字返回。
**6-赋值符号(' ='):** 是可选的，如果使用将赋值返回值，不使用将使方法不返回任何东西。
**7- 'method_body':** 是用花括号' {} '括起来的代码块，由所需的逻辑或某些任务或操作组成。
**8- return:** 是用于返回所需值的关键字，也是终止程序的关键字，但在 Scala 中很少使用。

**方法调用:-**Scala 中方法调用的语法如下:

```
method_name(arguments)
```

方法调用可以通过**‘Method _ name’**快速完成，这是您想要调用的对应方法的名称，并传递参数。

**带有命名参数的方法:-**

具有命名参数的方法将允许在方法调用期间将参数传递给方法的参数，其中每个参数都与方法参数一一匹配。您将在下面的示例中看到传递带有函数声明和定义的参数以及正在运行的方法调用。

> 对象计算结果{
> def funSub(x:Int，y:Int) : Int =
> {
> 
> var diff:Int = 0
> diff = x — y
> 
> //返回值
> return diff
> }
> def main(args:Array[String]){
> 
> //函数调用
> println("值的差为:"+ funSub(8，6))；
> println("值的差为"+ funSub(y=6，x = 8))；
> }
> }

上面的程序给出的输出为:
**的差值为:2**
**的差值为:2**

上面的程序包含一个对象 **'calculateResult'** ，里面包含一个名为 **'funSub** '的方法，参数 x 和 y 的返回类型都是' Int '，该方法的总返回类型是' Int '。后面跟着一个赋值语句来赋值返回值。花括号表示方法体的开始，其中变量' diff '用初始值 0 初始化。main 方法中完成的方法调用 **'funSub(8，6)** '将 8 匹配到' x '，6 匹配到' y '，并执行减法运算，返回' diff '的值并最终打印出来。类似地，' **funSub(x=8，y=6)'** 在方法调用期间将 6 匹配到' y '并将 8 匹配到' x '到方法中的参数，其中顺序与执行类似操作的位置无关，返回并打印出结果。

**默认参数值:-**

您可以通过初始化相应的值来指定方法参数的默认值，并且可以通过不传递参数来将方法调用保留为空

> 对象计算结果{
> def funSub(x:Int=9，y:Int=6) : Int =
> {
> 
> var diff:Int = 0
> diff = x — y
> 
> //返回值
> return diff
> }
> def main(args:Array[String]){
> 
> //函数调用
> print("最终值为:"+funSub())；
> 
> }
> }

以上程序给出的输出为:
**最终值为:3**

你可以看到上面的程序包含定义为 **'calculateResult'** 的对象，里面有一个名为 **'funSub'** 的方法，带有参数 x 和 y，两者的返回类型都是' Int '，方法的总返回类型是' Int '，后面是赋值语句，它将对返回值进行赋值。花括号表示方法体的开始，其中变量' diff '用初始值 0 初始化。方法调用是从 main 方法内部完成的，在 main 方法中 **'funSub()'** 调用并初始化 x 为 9，y 为 6，并执行操作，使得值' diff '被返回并打印出来。

**可变长度参数:-**

可变长度参数是接受任意可变数量的参数的参数，可以由用户或客户端传递。方法中的最后一个参数是使用需要重复的“*”声明的。

> 对象变量参数{
> def main(args:Array[String]){
> printAll(" Scala "，" is "，" great")
> }
> 
> def printAll(strings:String *){
> var I:Int = 0；
> 
> for(valueprintln(value)；
> I = I+1；
> }
> }

上面的程序给出的输出为:
**Scala**
**is**
**great**

您可以看到上面的程序包含一个带有“printAll”方法的对象“variableArgument ”,其中可变长度参数“String*”在末尾定义，在方法调用期间可以传递一个字符串列表。传递的字符串列表在主函数中循环并显示为输出。

递归函数:-
递归在纯函数式编程中起着很大的作用，Scala 非常好地支持递归函数。递归意味着函数可以重复调用自己。试试下面的程序，这是一个很好的递归例子，其中计算传递数的阶乘。

> object new obj {
> def main(args:Array[String]){
> for(I<-1 到 5)
> println(" Factorial of "+I+":= "+Factorial(I))
> }
> 
> def Factorial(n:BigInt):BigInt = {
> if(n<= 1)
> 1
> else
> n * Factorial(n-1)
> }【T14

程序的输出

```
Factorial of 1: = 1
Factorial of 2: = 2
Factorial of 3: = 6
Factorial of 4: = 24
Factorial of 5: = 120
```

**匿名功能:-**

匿名函数是那些没有名字的轻量级函数定义，在 Scala 中被认为是函数文字。

**匿名函数的语法示例如下:**

*   匿名函数的第一个语法和示例是:

**语法:**

```
('first_variable':'data_type', 'second_variable':'data_type') => "certain_expression"
```

**举例:**

```
(var1:String:var2:String) => var1 + var2
```

匿名函数的第二个语法和例子是:

```
(_:'data_type')operator(_'data_type')
```

**例如:**

```
(_:String)+(_:String)
```

第一个语法显示“= >”之后的表达式计算为特定值，而“= >”之前的变量列表用于计算表达式。上面的第二个语法就像一个占位符，它只接受一次作为“通配符”的值，然后在它们之间进行操作。

您将在下面看到一个匿名函数的示例:

> object anonymous demo
> {
> def main(args:Array[String])
> {
> 
> var function1 = (var1:Int，var 2:Int)= > var 1+var 2
> var function 2 =(_:Int)+(_:Int)
> 
> //函数调用
> println(function1(5，5))
> println(function2(7，3))
> }
> }

上面的程序给出的输出为:
**10**
**10**

你可以在上面看到名为**‘anonymous demo’**的对象，它的‘main’函数包含两个匿名函数。它们在语法上有所不同，但可以产生相同的结果，其中 **'function'1** 通过在被调用时从函数调用中获取参数来进行计算，即传递值 5 和 5，这导致输出被打印为 10，而 **'function2'** 也称为传递 7 和 3，其中它接收一次值，它接受任何有效的“整数”。在您的情况下，进行加法运算并输出结果。

> 恭喜你，你已经看完了这篇教程。

如果你喜欢我的这篇文章，请点击拍手按钮后欣赏！！

要阅读更多关于 **Scala、**的内容，你也可以参考下面的链接。

[](/@duke.lavlesh/java-vs-scala-7ff9eb50141) [## Java vs Scala！！

### 在开始阅读这篇文章之前，如果你真的想从头开始学习 Scala，你可以参考我以前的…

medium.com](/@duke.lavlesh/java-vs-scala-7ff9eb50141) [](/@duke.lavlesh/functional-programming-aspects-in-scala-3de975f9e3f2) [## Scala 中的函数式编程

### 函数式编程(FP)是一种编写计算机程序的方法，作为数学函数的评估，它…

medium.com](/@duke.lavlesh/functional-programming-aspects-in-scala-3de975f9e3f2)