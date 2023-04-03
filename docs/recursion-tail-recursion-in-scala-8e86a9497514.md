# Scala 中的递归/尾递归

> 原文：<https://medium.com/analytics-vidhya/recursion-tail-recursion-in-scala-8e86a9497514?source=collection_archive---------12----------------------->

与其他语言相比，Scala 有一些高级特性，包括对尾部递归的支持。但是让我们先看看它的意思，以及为什么它有助于构建递归算法。

在编程世界里，ecursion 相当普遍。你可能知道，这是通过将问题分解成更小的子问题来解决问题的过程。如果您看到一个方法用较小的输入子集调用自己，您可以很容易地发现递归。

试试下面的程序，这是一个很好的递归例子，其中计算传递数的阶乘。

```
object Demo {
   def main(args: Array[String]) {
      for (i <- 1 to 10)
         println( "Factorial of " + i + ": = " + factorial(i) )
   }

   def factorial(n: BigInt): BigInt = {  
      if (n <= 1)
         1  
      else    
      n * factorial(n - 1)
   }
}
```

# 尾部递归概述:-

尾部递归是一个子程序(函数，方法)，其中最后一条语句被递归执行。简单来说，函数中的最后一条语句多次调用自己。

这些函数更加高效，因为它们利用了尾部调用优化。它们只是继续调用一个函数(函数本身)，而不是在内存中添加一个新的堆栈帧。

尾部递归函数应该通过它们的参数传递当前迭代的状态。它们是不可变的(它们将值作为参数传递，而不是将值重新赋给相同的变量)。

**Scala 中的尾部调用优化:-**

Scala 通过执行尾部调用优化来支持尾部递归函数。它还有一个特殊的注释， [**@tailrec**](http://twitter.com/tailrec) **，**来保证方法可以以尾部递归的方式执行，否则编译器产生错误。

**写尾递归函数有什么要求？**

1-必须有一个退出条件:如果没有这个条件，函数将在一个永无止境的循环中结束，这肯定不是任何迭代计算的目标。退出条件可以通过使用条件来实现，当条件满足时，函数返回值，否则它继续用不同的属性值集合调用自己。

2 -尾部递归函数必须能够调用自身:如果在第一次迭代中满足退出条件，这可能不会发生，但是对于任何后续调用，所有参数都必须传递给函数本身。

3-在编写 tail 递归程序之前，需要导入 tailrec 注释。

## 下面是 Scala 中尾部递归函数的一个例子:

> def factorial(n:BigInt):BigInt =
> {
> [@ tailrec](http://twitter.com/tailrec)def factorial ACC(ACC:BigInt，n:BigInt):BigInt =
> {
> if(n<= 1)
> ACC
> else
> factorial ACC(n * ACC，n — 1)
> }
> factorialAcc(1，n)
> }
> // Main 方法
> def Main(t

感谢读者推荐我关于这个话题的帖子！

我将在以后的博客文章中更详细地讨论其他 Scala 主题！