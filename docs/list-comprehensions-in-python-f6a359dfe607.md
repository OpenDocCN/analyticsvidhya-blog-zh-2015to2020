# Python 中的列表理解

> 原文：<https://medium.com/analytics-vidhya/list-comprehensions-in-python-f6a359dfe607?source=collection_archive---------35----------------------->

![](img/4b4630e57ddaa4aa49ee95b1b8b55d9f.png)

# 介绍

列表理解技术是一种基于特定条件从给定列表中导出新列表的方法。如果你还记得你的高中时代，曾经有一个集合生成器符号的概念。

例如，为了表示所有小于 50 的偶数的数学集合(比如 A ),我们可以写

A ={ *x* | *x* 是一个偶数，*x*51 }，如果我们修改第二种记法，写成***A***= { x | x %2 = = 0&x<51 }，这里 x % 2 表示 x 模 2 即 x 除以 2 后的余数。

这里 x 表示成员自然数

|的意思是

x<51 is a condition or filter . In mathematical terminology it is called predicate that is a Boolean valued function.

x before | denotes output that is all x satisfying the condition .

**python 中的列表理解**

假设我们有一个列表 A = [1，2，3，4，5，6，7，8，9，10]，我们想创建一个包含偶数平方的列表。

1.  `**Output expression**`对成员执行的操作，在本例中为 x**2
2.  `**Member**`是 iterable 中的对象。在上面的例子中，x 作为成员工作。
3.  `**Iterable**`任何能逐一返回其元素的物体。它可以是集合、列表、生成器。在我们的例子中，A 充当 iterable。
4.  `**Condition**` 选择成员所依据的规则，可选。在我们的例子中，x%2==0 是条件。

可以看出，列表理解更具可读性，即声明性。

定义列表理解的一些一般方法

`output_expression for member in iterable`

例如，如果我们想生成一个包含" PYTHON" `[x for x in 'PYTHON']`
字符的列表，输出将是['P '，' Y '，' T '，' H '，' O '，' N']

`output_expression for member in iterable if condition`

如果不是基于特定的条件生成列表，而是想要修改列表的特定元素呢？

`output_expression if condition for member in iterable`

如果我们看到一个具体的例子，就会更清楚

回忆 A = [1，2，3，4，5，6，7，8，9，10]。用“偶数”替换偶数，并保持元素原样。

**嵌套列表理解**

假设我们有一个数组 A= [ [1，21，3]，[7，5，6]，[17，8，9] ]，并希望将此数组展平为类似于[1，21，3，7，5，6，17，8，9]的列表。

```
[y **for** x **in** A **for** y **in** x ]#Output
[ 1,21,3, 7,5,6, 17,8,9 ]
```

创建矩阵怎么样

```
arr = [[m for m in range(3)] for n in range(3)]
print(arr)
```

输出将是

```
[[0, 1, 2], [0, 1, 2], [0, 1, 2]]
```

这样我们就可以生成矩阵。请记住，arr 的类型仍将被列出。

另一个例子可以是得到数组/矩阵的转置。

A = [ [1，21，3]、[7，5，6]、[17，8，9] ]

```
transpose **=** [[x[i] **for** x **in** A] **for** i **in** range(3)]
print(transpose)#Output
[[1, 7, 17], [21, 5, 8], [3, 6, 9]]
```

***集合理解*** 类似于列表理解，只是我们用花括号{}代替了方括号[]。

***字典理解*** 作为集合理解工作，稍加修改，需要定义一个键。

示例`{x: x **3 for x in A}`

输出将是{1: 1，21: 9261，3: 27，7: 343，5: 125，6: 216，17: 4913，8: 512，9: 729}

让我们看另一个例子来更熟悉这个概念

`Team **=**` `['India', 'Australia', 'Pakistan']`

`Player **=**` `['Sachin', 'Warne', 'Wasim']`

创建一个字典，以团队为键，以玩家为值。

希望你明白理解是什么，以及如何实现它。