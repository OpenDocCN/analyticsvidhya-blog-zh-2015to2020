# 比较

> 原文：<https://medium.com/analytics-vidhya/comparison-6d20e377f627?source=collection_archive---------32----------------------->

*   大于/小于:`x > y`，`x < y`
*   大于/小于或等于:`X >= y`，`x <= y`
*   Equals `x == y`请注意，双质量符号`==`表示相等测试，而 a `a = b`，single 表示赋值。
*   不等于`!=`

# 布尔是结果

> *所有比较运算符都返回一个布尔值:*

```
alert ( 5 > 1); // true (correct)
 alert (5 < 1); // false (wrong)
alert (5 != 3); // true (correct)
```

> *比较结果可以用变量赋值，就像任何值一样:*

```
let result= 5 > 1; assign the result comparison
alert (result); // true
```

## 字符串比较

```
alert( 'Z' > 'A' ); // true 
alert( 'Glow' > 'Glee' ); // true
alert( 'Bee' > 'Be' ); // true
```

1.  比较两个字符串的第一个字符。
2.  如果第一个字符串的第一个字符大于(或小于)另一个字符串的第一个字符，则第一个字符串大于(或小于)第二个字符串。我们完了。
3.  否则，如果两个字符串的第一个字符相同，以同样的方式比较第二个字符。
4.  重复直到任一字符串结束。
5.  如果两个字符串以相同的长度结束，那么它们相等。否则，字符串越长越大。

在上面的例子中，当字符串`"Glow"`和`"Glee"`被逐字符比较时，比较`'Z' > 'A'`在第一步得到结果:

1.  `G`与`G`相同。
2.  `l`与`l`相同。
3.  `o`大于`e`。停在这里。第一串更大。

## 不同类型的比较

> 当比较不同类型的值时，JavaScript 将这些值转换为数字。

```
alert ('2' > 1); // true string become number
alert ('01' == 1); // true 
```

对于布尔值`true`变为 1，而`false`变为 0。

```
alert (true == 1); // true
alert (false == 0); // true
```

## 严格平等

常规的等式检查`==`有问题。它无法区分`0`和`false`

```
alert( 0 == false ); // true
```

> 空字符串也会发生同样的情况！

```
alert( '' == false ); // true
```

> 这是因为等式运算符`==`将不同类型的操作数转换为数字。空字符串，就像`false`一样，变成零。

**严格相等运算符** `**===**` **检查相等性而不进行类型转换。**

换句话说，如果`a`和`b`是不同的类型，那么`a === b`会立即返回`false`，而不会试图转换它们。

```
alert( 0 === false ); // false, because the types are different
```

还有一个类似于`!=`的“严格不等”运算符`!==`。

严格的等式操作符写起来有点长，但是它使事情变得很明显，并减少了出错的空间。

## 与 null 和 undefined 的比较

当`null`或`undefined`与其他值比较时，会有一种不直观的行为。

进行严格的平等检查`===`

> 这些值是不同的，因为它们都是不同的类型。

```
alert( null === undefined ); // false
```

对于非严格检查`==`

> 有一个特殊的规则。这两个是“甜蜜的一对”:他们彼此相等(在`==`的意义上)，但没有任何其他值。

```
alert (null == undefined); // true
```

数学和其他比较`< > <= >=`

> `null/undefined`转换成数字:`null`变成`0`，而`undefined`变成`NaN`

回避问题

> 除了严格相等之外，要格外小心地对待与`undefined/null`的任何比较
> 
> 不要将`>= > < <=`和一个可能是`null/undefined`的变量进行比较，除非你真的确定你在做什么。如果一个变量可以有这些值，请分别检查它们