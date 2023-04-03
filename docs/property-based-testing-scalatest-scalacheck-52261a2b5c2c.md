# 基于属性的测试:ScalaTest + ScalaCheck

> 原文：<https://medium.com/analytics-vidhya/property-based-testing-scalatest-scalacheck-52261a2b5c2c?source=collection_archive---------7----------------------->

![](img/d36c84e1afa57f87895ac2f570028401.png)

基于属性的测试是测试软件的另一种策略或哲学

一个*属性*可以是一个方法或对象的任何行为特征，它应该在任何模糊情况下保持*为真*。

*基于属性的测试*源自函数式编程社区:**快速检查**在 **Haskell** 中

**定义:**

> 基于属性的测试是这样构造测试的，当这些测试被模糊化时，测试中的失败揭示了被测系统的问题，这些问题不能通过直接模糊化该系统来揭示。

让我们快速看看 ScalaTest 和 ScalaCheck 是如何工作的

**ScalaTest:**

ScalaTest 是一个简洁、易读的测试框架，开发它是为了快速编写测试用例，并以近乎纯文本的方式表达它们

**ScalaCheck:**

scalaCheck 是基于属性测试的 Scala 框架。它与 ScalaTest 也有很好的集成。

让我们跳到代码，看看它们有多简单，

**美芬依赖:**

```
<dependency>
    <groupId>org.scalatest</groupId>
    <artifactId>scalatest_2.11</artifactId>
    <version>3.0.8</version>
    <scope>test</scope>
</dependency>
<dependency>
    <groupId>org.scalacheck</groupId>
    <artifactId>scalacheck_2.11</artifactId>
    <version>1.14.0</version>
    <scope>test</scope>
</dependency>
```

假设我们有两个方法，

*   *div* —将两个数相除
*   *追加—* 在列表末尾追加一个整数

```
**object** Main {

  **def** div(n: Int, d: Int): Double = {
    *require*(d != 0, "Denominator should not be 0")
    n / d
  }

  **def** append(list: List[Int], postfix: Int): List[Int] = { 
    list :+ postfix 
  }
}
```

在 **ScalaTest** 中，我们有两种写 ScalaCheck 的风格——

1.  斯卡拉切克风格
2.  使用 ScalaTest ***匹配器*** 【我们将使用这个】

我将解释我们如何在使用和不使用 ***ScalaCheck*** 的情况下进行测试，以了解差异，

***不带 ScalaCheck:***

```
"div function" should "return expected values" in {
  *div*(4, 2) shouldBe 2
}"append function" should "return only one element when added in empty list" in {
  **val** emptyList = *List*.*empty*[Int]

  *append*(emptyList, 1) should *have* size 1
}
```

测试使用*匹配器* trait 来使用*should*子句定义断言。ScalaTest 支持许多类似的断言简写，这使得它具有可读性和简洁性。

***同***

```
"div function" should "return expected values" in {
   forAll(*arbitraryInts*, *nonZeroInts*) { (n: Int, d: Int) =>
     *div*(n, d) shouldBe n / d
   }
}"append function" should "return only one element when added in empty list" in {
  **val** emptyList = *List*.*empty*[Int]

  forAll(*arbitraryInts*) { n: Int =>
    *append*(emptyList, n) should *have* size 1
  }
}
```

在 *ScalaCheck* 中定义属性非常简单:在上面的例子中，我们正在测试 *div* 和 *append* 函数的属性，给定一些模糊或随机的输入，它们仍然产生预期的结果。

*forAll* 方法用于提供输入—上面的例子有两个随机 *Gen* 对象，它们产生随机 *Int* 来测试 *div* 功能。

随机输入定义如下，

```
**val** *arbitraryInts*: Gen[Int] = arbitrary[Int]
**val** *nonZeroInts*: Gen[Int]   = arbitrary[Int] suchThat (_ != 0)
```

*arbitrary[Int]* 是产生 *Gen* 对象的方法之一，该对象是用于生成给定类型的任意输入的抽象。

下面给出了其他几个例子，

```
Gen.choose(0,100) // Will give you one Int from the rangeGen.oneOf('A', 'E', 'I', 'O', 'U', 'Y') // One from the listGen.containerOf[List,Int](Gen.oneOf(1, 3, 5))// With some condition : any even number from 0 - 200
Gen.choose(0,200) suchThat (_ % 2 == 0)// Picking any 5 from the range 1 to 6
Gen.pick(5, 1 to 6)// Even with some distribution of each value
Gen.frequency(
  (3, 'A'),
  (4, 'E'),
  (2, 'I'),
  (3, 'O'),
  (1, 'U'),
  (1, 'Y')
)
```

以上所有方法为我们提供了*Gen*object——它将在 *forAll* 方法中使用，为我们的测试用例提供模糊性。

基于属性的测试为我们提供了一种在测试逻辑中包含模糊性的简单方法，这种方法在我们传统的测试方法中是无法评估的。在传统的方法中，我们提供一组期望的输入和期望的输出，通过它们我们不测试方法属性的极端情况——这可能导致生产中的严重错误。

在 *ScalaCheck* 中，我们可以配置模糊度并测试负载以及以下属性:

```
forAll(*nonZeroInts*, minSuccessful(50)) { n: Int =>
  *div*(n, n) shouldBe 1
}
```

*forAll* 方法包含变量 argument 来提供*属性检查配置*来定义测试条件。上面示例定义了要执行的最少 50 次成功执行。

*maxDiscardedFactor，sizeRange，workers* 是其他可以使用的配置。

*ScalaCheck* 还提供了一些复杂的模糊创建逻辑，包括生成整个 case 类对象。有关详细的实现以及与 ScalaTest 的集成，请参考下面给出的链接。

对于 Scala 开发者来说， *ScalaTest* 和 *ScalaCheck* 是非常好的选择，可以快速编写简洁易读的测试用例。

**参考文献:**

*示例代码*

[](https://github.com/subashprabanantham/hakuna-matata/tree/master/scalatest) [## subashprabanantham/hakuna-matata

### 此时您不能执行该操作。您已使用另一个标签页或窗口登录。您已在另一个选项卡中注销，或者…

github.com](https://github.com/subashprabanantham/hakuna-matata/tree/master/scalatest) 

*ScalaTest*

 [## ScalaTest

### ScalaTest Maven 插件允许您通过 Maven 运行 ScalaTest 测试，而不需要…

www.scalatest.org](http://www.scalatest.org/user_guide/using_the_scalatest_maven_plugin) 

*Scala test 中的 Scala check*

 [## ScalaTest

### ScalaTest 支持基于属性的测试，其中属性是行为的高级规范，应该包含…

www.scalatest.org](http://www.scalatest.org/user_guide/property_based_testing) 

*ScalaCheck 用户指南*

 [## 类型级别/标量检查

### ScalaCheck 是一个测试 Scala 和 Java 程序的工具，基于属性规范和自动测试数据…

github.com](https://github.com/typelevel/scalacheck/blob/master/doc/UserGuide.md)