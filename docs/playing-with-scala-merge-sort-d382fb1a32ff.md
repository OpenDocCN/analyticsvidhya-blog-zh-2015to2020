# 使用 Scala——合并排序

> 原文：<https://medium.com/analytics-vidhya/playing-with-scala-merge-sort-d382fb1a32ff?source=collection_archive---------5----------------------->

Scala 的模式匹配真的真的很强大。为了体验它的强大，我尝试使用模式匹配实现一个非常基本但强大的排序算法。

![](img/45e2be52ba7719e441a8b9d096d2c8c6.png)

安德鲁·雷德利在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

基本合并排序算法包括两个函数

```
***def mergeSort(seq: List[Int]): List[Int]***
```

→它接受一个未排序的集合并返回一个排序的集合。

```
***def merge(seq1: List[Int], seq2: List[Int]): List[Int]***
```

→这需要两个排序的集合，并通过合并它们返回一个排序的集合。

这个版本非常简单，使用了模式匹配的能力。

合并排序函数→

```
def mergeSort(seq: List[Int]): List[Int] = seq match {
  case *Nil* => *Nil* case xs::*Nil* => *List*(xs) 
  case _ => 
    val (left, right) = seq splitAt seq.length/2
    *merge*(*mergeSort*(left), *mergeSort*(right))
}
```

合并功能→

```
def merge(seq1: List[Int], seq2: List[Int]): List[Int] = 
        (seq1, seq2) match {
           case (*Nil*, _) => seq2
           case (_, *Nil*) => seq1
           case (x::xs, y::ys) =>
            if(x<y) x::*merge*(xs,seq2)
            else y::*merge*(seq1,ys)
}
```

这对于较小的输入非常有效，但是对于较大的输入，由于显而易见的原因，它会抛出 StackOverflow 异常。我将它的性能与 Scala 中标准的 ***排序*** 函数进行了比较。

100 个数字，标准排序:71 毫秒，合并排序(上图):10 毫秒

1000 个数字，标准排序:76 毫秒，合并排序(上图):15 毫秒

10000 个数字，标准排序:82ms，合并排序(上图): ***堆栈溢出***

***如何修复堆栈溢出错误？***

尾部递归:马丁·奥德斯基解释如下:

> 将自己作为最后动作的函数称为尾递归函数。Scala 编译器会检测到尾部递归，并在用新值更新函数参数后，用跳回到函数开头来代替它…只要你做的最后一件事是调用自己，它就会自动进行尾部递归(即优化)。

现在我们如何优化我们的功能？

```
def merge(seq1: List[Int], seq2: List[Int]): List[Int] = 
        (seq1, seq2) match {
           case (*Nil*, _) => seq2
           case (_, *Nil*) => seq1
           case (x::xs, y::ys) =>
            if(x<y) x::*merge*(xs,seq2)
            else y::*merge*(seq1,ys)
}
```

我们最后做的其实是 ***x::merge(xs，seq2)*** 或者 ***y::merge(seq1，ys)。没有资格打最后一个电话吗？尾音？***

不会。如果你再想想，最后发生的事情实际上是将 x 或 y 的*和 ***merge()*** 函数调用的结果串联起来形成一个链表。*

*我们如何解决它？*

*我们可以在这个函数中传递一个累加器，并在那里附加我们的结果。*

```
****@tailrec***
def merge(seq1: List[Int], seq2: List[Int], ***accumulator: List[Int] = List()***):List[Int] = (seq1, seq2) match {
  case (*Nil*, _) => ***accumulator*** ++ seq2
  case (_, *Nil*) => ***accumulator*** ++ seq1
  case (x::xs, y::ys) =>
    if(x<y) *merge*(xs,seq2, ***accumulator* :+ x**)
    else *merge*(seq1,ys, ***accumulator* :+ y**)
}*
```

****@tailrec*** →这个注释帮助我们验证是否真的是尾调用。如果不是，编译器抛出一个错误。*

***比较性能:***

**100 个数字，标准排序:74 毫秒，合并排序(上图):15 毫秒**

**1000 个数字，标准排序:72 毫秒，合并排序(上图):40 毫秒**

**10000 个数字，标准排序:82 毫秒，合并排序(上图):1236 毫秒**

**10 万个数字，标准排序:171 毫秒，合并排序(上图):90 秒**

*尽管标准算法在较大的输入上表现得更好，但这个版本至少让我们避免了 StackOverflow。多亏了尾部递归。*

*你有其他更好的版本吗？与我分享。我会尝试一下。*

*谢了。*