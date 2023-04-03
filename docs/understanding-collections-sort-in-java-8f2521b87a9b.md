# 理解 Java 中的 Collections.sort()

> 原文：<https://medium.com/analytics-vidhya/understanding-collections-sort-in-java-8f2521b87a9b?source=collection_archive---------5----------------------->

顾名思义，java 的 Collections 类提供了一个名为 sort()的静态方法，用于按照特定的顺序对项目集合进行排序。

```
**public static** <T> **void** sort(List<T> list, Comparator<? **super** T> c) {
    list.sort(c);
}
```

我们来看看这个方法的定义。正如我们所见，它有两个参数，一个是 T 类型的对象列表，另一个是比较器，它接受 T 类型的对象并返回列表排序的顺序。在这个函数定义中，我们给出了一个比较器来明确定义排序顺序。

Collections.sort()方法还有一个定义。

```
**public static** <T **extends** Comparable<? **super** T>> **void** sort(List<T> list) {
    list.sort((Comparator)**null**);
}
```

在这个定义中，静态排序方法只接受 T 类型的对象列表，并以默认顺序对它们进行排序。这里我们可以看到，为了以某种默认顺序对 T 进行排序，T 必须实现 Comparable 接口。

重要的是要注意到，

> **所有包装类**和字符串**类**用 Java 实现**可比**接口。**包装类按照它们的值进行比较**，字符串**按照字典顺序进行比较**。默认情况下，它们按升序对元素进行排序。

要使用 Collections.sort()方法对 T 类型的剩余对象进行排序，可以:

1.  该对象应实现类似的接口，或者，
2.  我们应该定义一个自定义的比较器，它可以对 T 类型的对象进行排序，并将其作为第二个参数传递给我们的排序函数。

## 可比接口

以整数包装类为例。整数类实现可比较的接口。Comparable 接口有一个称为 compareTo()的方法。在 Integer 类中，它覆盖 compareTo 方法。它将当前对象与排序函数中传递的对象进行比较，我们称之为 var1，当当前对象小于 var1，当前对象等于 var1，当前对象大于 var1 时，它分别返回-1，0，1。

```
**Comparable Interface:
public interface** Comparable<T> {
    **int** compareTo(T var1);
}
____________________________________________________________________**Integer class:
public int** compareTo(Integer anotherInteger) {
    **return** compare(**this**.value, anotherInteger.value);
}

**public static int** compare(**int** x, **int** y) {
    **return** x < y ? -1 : (x == y ? 0 : 1);
}
```

因此，这意味着每当一个类实现 Comparable 接口时，它可以在 compareTo 函数中添加自己的逻辑，因此可以有自己的逻辑进行排序。

现在，如果我们想以不同于默认顺序的其他顺序对 T 类型的对象进行排序呢？为了解决这个问题，比较器应运而生。

## 比较器接口

当我们想要定义一个排序 T 类型的对象的顺序时，我们可以创建一个自定义的比较器，在排序 List <t>list 时，我们可以将它作为第二个参数传入 Collections.sort()方法。</t>

```
@**FunctionalInterface**
**public interface** Comparator<T> {
    **int** compare(T var1, T var2);
    ...
}
```

比较器是一个功能接口。
*Java 中的一个* ***函数接口*** *是一个* ***接口*** *只包含一个抽象(未实现)方法。一个* ***函数接口*** *除了单个未实现的方法之外，还可以包含有实现的默认和静态方法。*

Comparator 接口有一个名为 compare()的抽象方法。它比较两个对象 T1 和 T2，在大多数实现中，当 T1 <t2 when="" t1="=T2" and="">T2 时，通常分别返回-1，0，1。</t2>

示例说明了在按升序对 T 类型的对象进行排序时使用比较器接口。

假设对象 T 有以下两个属性:dataLimit 和 dataUsed，两者都是整数类型。

## 在函数参数中定义比较器

```
Collections.*sort*(list, **new** Comparator<T>() {
    *//Sort based on ascending order of dataLimit* @Override
    **public int** compare(T obj1, T obj2) {
        **return obj1.getDataLimit()-obj2.getDataLimit();**
    }
});
```

## 在外部定义比较器，并在函数的参数中传递它的对象

```
Comparator<T> CustomComparator= **new** Comparator<T>() {
    @Override
    **public int** compare(T o1, T o2) {
        return **o1.getDataLimit()-o2.getDataLimit();**
    }
};
Collections.*sort*(list, CustomComparator);
```

## 将 lambda 表达式与构造函数一起使用

```
list.sort((T o1, T o2) -> o1.getDataLimit()-o2.getDataLimit());orComparator<T> CustomComparator = (o1, o2)-> o1.getDataLimit()-o2.getDataLimit();
list.sort(CustomComparator);
```

按降序排序:此函数 **reversed()** 可通过函数链接与任何比较器一起使用，以将排序逻辑改为逆序。

```
**default** Comparator<T> reversed() {
    **return** Collections.reverseOrder(**this**);
} Comparator<T> CustomComparator = (o1, o2)-> o1.getDataLimit()-o2.getDataLimit();
list.sort(CustomComparator.reversed());
```

您可以探索 comparable 和 Comparator 接口的更多方法，并通过深入研究包装类来理解它的工作原理。

伙计们，我真的希望你觉得这篇文章有价值！如果你还有任何疑问，你可以在下面的评论区讨论。

非常感谢你花时间写这个博客。

请分享给你的同事，并鼓掌表示赞赏！:)