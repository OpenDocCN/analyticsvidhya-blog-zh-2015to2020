# Kotlin 中的数据类

> 原文：<https://medium.com/analytics-vidhya/data-class-in-kotlin-32bf038729a1?source=collection_archive---------8----------------------->

## 不仅仅是平等。

让面试官知道你为什么喜欢科特林的一门数据课，这是面试过程中常见的问题。正确答案似乎是免费平等。尽管这是一个非常棒的特性，但 data class 还有更多这样的特性。

![](img/5f1a27ab49759f2392da2c4ae25f04bb.png)

[https://www . pexels . com/photo/woman-touching-a-mirror-3881965/](https://www.pexels.com/photo/woman-touching-a-mirror-3881965/)

# 数据类

一个简单的关键词变成了编程游戏的改变者——数据。数据类用于描述传输对象。编程就是获取输入数据，处理它们并返回输出数据。下面是数据类的帮助。

```
data class Man(
    val name: String,
    val surname: String,
    val bDate: String? = null
)
```

## 平等

我承认当我跳入 Kotlin 时，我停止了重写*等于*和*哈希码*方法。大多数情况下都有一个内置功能。尽管是两个不同的实例，但当所有构造函数参数都相等时，两个精确的对象是相等的。

```
@Test
fun `equality of two objects`() {
    val man1 = Man("Ken", "Mattel", "01/01/1961")
    val man2 = Man("Ken", "Mattel", "01/01/1961")

    assertTrue(**man1 == man2**)
    assertFalse(**man1 === man2**)
}
```

## 散列码

现在，每当我重写*等于*时，我不需要记得重写 *hashCode* 方法。数据类替我做了。我可以使用数据类对象作为哈希映射中的键！

```
@Test
fun `hashcode of two objects`() {
    val man1 = Man("Ken", "Mattel", "01/01/1961")
    val man2 = Man("Ken", "Mattel", "01/01/1961")
    val message = "I'm here already!"

    val map = *hashMapOf*(**man1 *to* message**)

    assertTrue(man1.hashCode() == man2.hashCode())
    assertEquals(**message, map[man2]**)
}
```

## 可读性

我知道这不是一个数据类特性，但是 Kotlin 给了我们一个极大的工具，极大地改进了代码审查。我说的是命名参数。我不再需要用描述性名称来定义局部值。现在，我知道肯是一个真实的名字还是只是一个品牌。

```
@Test
fun `readability for reviewer`() {
    val name = "Ken"
    val surname = "Mattel"
    val bDate = "01/01/1961"
    val man1 = Man(name, surname, bDate)
    val man2 = Man(
 **name =** "Ken",
 **surname =** "Mattel",
 **bDate =** "01/01/1961"
    )

    assertEquals(man1.name, man2.name)
    assertEquals(man1.surname, man2.surname)
    assertEquals(man1.bDate, man2.bDate)
}
```

## 内置生成器

默认值和命名参数一起给了我们一个生成器设计模式的能力。我可以改变数据类声明中参数的顺序，这不会影响我的代码。

```
@Test
fun `built-in builder design pattern`() {
    val man = Man(
        surname = "Mattel",
        name = "Ken"
    )

    assertNull(man.bDate)
}
```

## 内置原型

命名参数对于*复制*方法非常有用。“复制”使用“原始”的值克隆对象，但有一个选项可以覆盖它们。

```
@Test
fun `built-in prototype pattern`() {
    val bDate = "01/01/1961"
    val man = Man(
        name = "Ken",
        surname = "Mattel"
    )

    val manWithBirthDate = man.copy(bDate = bDate)

    assertEquals(man.name, manWithBirthDate.name)
    assertEquals(man.surname, manWithBirthDate.surname)
    assertEquals(bDate, manWithBirthDate.bDate)
}
```

## 打印信息

不再需要覆盖 *toString* 方法。这是免费赠送的。这对于调试或日志记录非常有用。

```
val man = Man("Ken", "Mattel", "01/01/1961")
print(man) // prints Man(name=Ken, surname=Mattel, bDate=01/01/1961)
```

## 破坏组件

数据类可以被提取成组件。我非常喜欢这个功能，尤其是在地图条目中，我可以像这样打开一个键和值:

```
val (key, value) = mapOf(1 to "one").entries.first()
print(key) // prints 1
print(value) // prints one
```

同样的方式，我可以提取数据类。当然，对于大数据类，它不会像对于小数据束那样有用。

```
@Test
fun `components of data class`() {
    val man = Man("Ken", "Mattel", "01/01/1961")
    val (name, _, bDate) = man

    assertEquals(name, man.name)
    assertEquals(bDate, man.bDate)
}
```

# 需要注意什么

总会有一个平衡。就像阴阳一样，为好而存在的东西也可以被用来做坏事。我这里的例子很简单，因为关于 Ken 的数据是不可变的。男人从不改变自己的名字、姓氏或出生日期。另一方面，女人有。我可以这样定义这种情况:

```
data class Woman(
    val name: String,
    **var surname: String,**
    val bDate: String? = null
)
```

当芭比最终嫁给肯时，她会跟他姓。构造函数中的这个变量保留了数据类的所有特性，但是它也为错误提供了空间。

```
@Test
fun `caveat of using var`() {
    val woman = Woman("Barbie", "Girl", "31/12/1959")
    val map = *hashMapOf*(woman *to* "I'm here!")

    woman.surname = "Mattel"

    assertNull(map[woman])
    *print*(map) // prints {Woman(name=Barbie, surname=Mattel, bDate=31/12/1959)=I'm here!}
}
```

我的密钥不再是唯一的。哈希代码已更改，哈希映射打印精确的对象，我无法检索。那是一个很难发现的漏洞。芭比刚刚被留在真空中，肯将独自行走并试图找到她——永远…

…永远。

## 固定

在数据类中使用 var 有一个更好的方法。如果我只是从构造函数中取出 var，equal 和 hashCode 会忽略这个属性。当然，这意味着现在任何出生于 1959 年 1 月 1 日的芭比都是同一个芭比——这是要付出的代价。

```
data class Woman(
    val name: String,
    val bDate: String? = null
) {
    **var surname: String = ""**
}@Test
fun `fix for using var`() {
    val woman = Woman("Barbie", "31/12/1959")
        .*apply* **{** surname = "Girl" **}** val map = *hashMapOf*(woman *to* "I'm here!")

    woman.surname = "Mattel"

    assertNotNull(map[woman])
}
```

# 摘要

数据类是一个非常强大的工具。了解它的优点和缺点有助于找到特定情况下的最佳解决方案。如果我确定我的数据类不会被用作一个映射的键，或者在一个有序集合中使用，那么我看不出为什么我不应该在数据类构造函数中使用 var。但我能永远确定吗？至少它应该打开我脑袋里的一盏黄灯。