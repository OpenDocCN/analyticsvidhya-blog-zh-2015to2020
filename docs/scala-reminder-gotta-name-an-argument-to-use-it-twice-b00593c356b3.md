# Scala 提醒:必须命名一个参数才能使用两次

> 原文：<https://medium.com/analytics-vidhya/scala-reminder-gotta-name-an-argument-to-use-it-twice-b00593c356b3?source=collection_archive---------28----------------------->

![](img/43e53139adfe7199091f9cc2f205ae1d.png)

micha Parzuchowski 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

这将是一篇短文。Scala 是一种非常强大的编程语言，语法非常简洁。问题是，我时常会忘记这个简洁语法的某些细节。比如说，你如何使用通配符两次？

我曾经很快学会了如何使用通配符。例如:

```
scala> LazyList.iterate(1: BigInt)(_ * 7)
res0: scala.collection.immutable.LazyList[BigInt] = LazyList(<not computed>)
```

第一对括号将 1 括起来作为一个`BigInt`。在第二对括号中，下划线通配符在第一次迭代中表示初始值(在本例中为 1)，在以后的迭代中表示由前一次迭代计算出的值。

因此，在本例中，通配符代表的值乘以整数 7。由于我通常认为理所当然的隐式转换，`Int`被转换为`BigInt`，并且由于“操作符重载”，我们可以使用乘法操作符来代替`multiply()`或`times()`或其他任何可能的名称。

因此，第一次迭代产生 7，然后第二次迭代产生 49，以此类推。

```
scala> res0.take(20).toList
res1: List[BigInt] = List(1, 7, 49, 343, 2401, 16807, 117649, 823543, 5764801, 40353607, 282475249, 1977326743, 13841287201, 96889010407, 678223072849, 4747561509943, 33232930569601, 232630513987207, 1628413597910449, 11398895185373143)
```

复制并粘贴到[OEIS 搜索框](https://oeis.org/)，第一个结果应该是 A420，7 的幂。事实上这应该是唯一的结果。

但是，有时您需要使用通配符两次。

例如，考虑有时被认为是艾萨克·牛顿爵士提出的近似平方根的方法。通过“牛顿法”，我们从最初的猜测 *g* (0)开始逼近 *x* 的平方根，我们通过公式*g*(*n*)=(*g*(*n*-1)+*x*/*g*(*n*-1))/2 逐步细化。

在这里，*g*(*n*-1)似乎是在迭代函数中两次使用通配符的`LazyList`迭代的完美选择。除了你不能在 Scala 中这样做。有些 Scala 表达式可以使用两个下划线通配符，但这不是其中之一。

```
scala> def newtonSqrt(x: Double, initGuess: Double): LazyList[Double] = LazyList.iterate(initGuess)((_ + x/_)/2)                                                                                             
                                                ^             
       **error:** missing parameter type for expanded function ((<x$1: error>, x$2) => x$1.$plus(x.$div(x$2)))
                                                      ^
       **error:** missing parameter type for expanded function ((<x$1: error>, <x$2: error>) => x$1.$plus(x.$div(x$2)))
```

所以我必须给下划线通配符一个类型？听起来不太对劲。

在谷歌上搜索了一番后，我看到了布兰登·奥康纳的 Scala 备忘单。

> `(1 to 5).map(x => x * x)`
> 
> 匿名函数:要使用一个[参数]两次，[你]必须给它命名。

啊哈！我必须命名“通配符”，但我不必指定它的类型，这是由编译器推断出来的。

```
scala> def newtonSqrt(x: Double, initGuess: Double): LazyList[Double] = LazyList.iterate(initGuess)(g => (g + x/g)/2)
newtonSqrt: (x: Double, initGuess: Double)LazyList[Double]
```

好吧，看起来有用。但是证据在布丁里。

```
scala> newtonSqrt(13.0, 4.0)
res12: LazyList[Double] = LazyList(<not computed>)scala> res12.take(20).toList
res13: List[Double] = List(4.0, 3.625, 3.605603448275862, 3.6055512758414574, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896, 3.6055512754639896)scala> res13(19) * res13(19)
res14: Double = 13.000000000000002
```

对于类型为`Double`(Java 中的`double`)的 JUnit 中的`assertEquals()`，我一般使用 0.00001 的 delta。因此，13.0000000000002 对于该增量来说就足够了。

好了，你知道了。在 Scala 中，你不能总是使用下划线通配符，但有时你可以命名一个通配符并使用两次。