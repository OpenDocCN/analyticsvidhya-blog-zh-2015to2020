# 类型转换、基本运算符和数学

> 原文：<https://medium.com/analytics-vidhya/type-conversion-and-basic-operator-and-maths-7333ae5ce37?source=collection_archive---------30----------------------->

大多数情况下，运算符和函数会自动将赋予它们的值转换为正确的类型。

> 例如，`*alert*`自动将任何值转换成字符串来显示。数学运算将值转换成数字。

![](img/ac73acdf59e102b3968edcc8369e5b0d.png)

照片由[巴拉特·帕蒂尔](https://unsplash.com/@bharat_patil_photography?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

## 字符串转换

当我们需要字符串形式的值时，就会发生字符串转换。

例如，`alert(value)`就是用来显示值的。

我们也可以调用`String(value)`函数将一个值转换成一个字符串:

```
let value = true;
alert(typeof value); // booleanvalue = String(value); // now value is a string "true"
alert(typeof value); // string
```

字符串转换大多显而易见。一个`false`变成`"false"`，`null`变成`"null"`，以此类推。

## 数字转换

*   它发生在数学函数和表达式类型的数值转换中。
*   当除法`/`应用于非数字时:

`alert('6' / '2'); // 3, strings are converted to the number`

我们可以使用`Number(value)`函数显式地将`value`转换为数字:

```
let string = '123';
alert(typeof string); //stringlet num = number (string); // becomes a number 123;
alert (typeof num); // number
```

如果字符串不是有效的数字，这种转换的结果是`NaN`

```
let str = Numb("an arbitrary string instead of a number");alert(str); // NaN, conversion failed
```

> 数字转换规则:

**值→变成**

`undefined` → `Nan`

`null` → `0`

`true of false` → `1 and 0`

`string` →删除开头和结尾的空白。如果剩余的字符串为空，则结果为`0`。否则，从字符串中“读取”数字。一个错误给出`NaN`

```
alert( Number('   555  ') ); // 555
alert( Number('a01') );      // NaN (error reading a number at "a")
alert( Number(true) );        // 1
alert( Number(false) );       // 0
```

## 布尔转换

> 它发生在逻辑运算中(稍后我们会遇到条件测试和其他类似的事情)，但也可以通过调用`Boolean(value)`显式执行。

> 直观上“空”的值，如空字符串`*0*`、`*null*`、`*undefined*`和`*NaN*`，变成了`*false*`
> 
> 其他值变为`*true*`

```
alert( Boolean(1) ); // true
alert( Boolean(0) ); // falsealert( Boolean("hi") ); // true
alert( Boolean("") ); // false
```

# 术语:“一元”、“二元”、“操作数”

*   一个*操作数就是*运算符所适用的对象。例如，在`5 * 2`的乘法中有两个操作数:左操作数是`5`，右操作数是`2`。有时候，人们称这些为“自变量”，而不是“操作数”。
*   如果一个操作符只有一个操作数，那么它就是一元操作符。例如，一元否定`-`反转一个数字的符号:

```
let y = 1;y = -y;
alert (y); // y is -1 unray negation is applied
```

如果一个操作符有两个操作数(参数)，那么它就是二元的。同样的减号也存在于二进制形式中

```
let y = 2;
let x = 3;alert(x - y); // 1, binarny minus substracts values
```

## 数学

*   加法`+`，
*   减法`-`，
*   乘法运算`*`，
*   分部`/`，
*   余数`%`，
*   求幂运算`**`。

## 剩余物

余数运算符`%`尽管看起来像，但与百分比无关。

`a % b`的结果是`a`除以`b`的整数的余数

```
alert( 5 % 2 ); // 1, a remainder of 5 divided by 2
alert( 8 % 3 ); // 2, a remainder of 8 divided by 3
```

## 指数运算

取幂运算符`a ** b`将`a`乘以自身`b`倍

```
alert( 2 ** 3 ); // 8  (2 * 2 * 2, 3 times)
alert( 2 ** 4 ); // 16 (2 * 2 * 2 * 2, 4 times)
```

用 binary +连接字符串

通常，加号运算符`+`对数字求和。

> 但是，如果将二进制`*+*`应用于字符串，它会合并(连接)它们

```
let conc = 'hello' + 'world';
alert(conc); // helloworldalert('1' + 2); // "12"
alert(2 + '1'); //"21"
```

> 请注意，如果任何一个操作数是字符串，那么另一个操作数也会转换为字符串

这里有一个更复杂的例子

```
alert(3 + 3 + '2'); // "62" not "232"
```

第一个`+`将两个数相加，因此它返回`6`，然后下一个`+`将字符串`2.`相加

> 二进制码`*+*`是唯一以这种方式支持字符串的操作符

减法和除法的 Ex

```
alert( 6 - '3' ); // 3, converts '3' to a number
alert( '6' / '2' ); // 3, converts both operands to numbers
```

## 数字转换，一元+

> 加号有两种形式:我们上面使用的二进制形式和一元形式。

一元加号，或者换句话说，应用于单个值的加号运算符`+`，对数字没有任何作用。但是如果操作数不是数字，一元加号就把它转换成数字。

```
// No effect on numbers
let x = 2;
alert( +x ); // 2let y = -6;
alert( +y ); // -6// Converts non-numbers
alert( +true ); // 1
alert( +"" );   // 0
```

它的功能与`Number(...)`相同，但更短。

> 经常需要将字符串转换成数字。例如，如果我们从 HTML 表单字段中获取值，它们通常是字符串

二进制加号会将它们作为字符串相加:

```
let me = '5';
let you = '5';// both values converted to numbers before the binary plusalert(+me + +you); // 10// the longer variant
// alert( Number(apples) + Number(oranges) ); // 5
```

## 分配

让我们注意一个赋值`=`也是一个运算符。它被列在优先级表中，具有非常低的优先级`3`

这就是为什么当我们给一个变量赋值时，比如`x = 2 * 2 + 1`，计算首先完成，然后对`=`求值，将结果存储在`x`中。

```
let x = 2 * 2 + 1;alert( x ); // 5
```

链接赋值

```
let a,b,c;a = b = c = 2+2;alert(a); // 4
alert(b); // 4
alert(c); // 4
```

链式赋值从右向左计算。首先对最右边的表达式`2+2`进行求值，然后赋值给左边的`c`、`b`、`a`。

> 所有变量共享一个值。

```
c = 2 + 2;
b = c;
a = c;
```

> 同样，出于可读性的目的，最好将这样的代码分成几行

## 递增/递减

增加或减少一个数字是最常见的数字运算之一！

## 增量→ ++

```
let guessMyNumber = 5;
let guessMynumber++;alert(guessMyNumber); // 6
```

> **增量** `*++*`将变量增加 1

## 减量→- -

```
let guessMyNumber = 5;
let guessMynumber++;alert(guessMyNumber); // 4
```

> **减量**将一个变量减 1

> 递增/递减只能应用于变量。试图在像`5++`这样的值上使用它将会给出一个错误。

运算符`++`和`--`可以放在变量之前或之后。

当操作符追踪变量时，它是“后缀形式”:`guessMyNumber++`

“前缀形式”是当操作符在变量之前:`++guessMyNumber`

这两条语句做同样的事情:将`guessMyNumber`增加`1`

```
let guessMyNumber = 2;
let a = ++guessMyNumber; // (this)alert (a); // 3
```

在`(this)`行中，*前缀*形式`++guessMyNumber`增加`guessMyNumber`并返回新值`3`。所以，`alert`显示的是`3`。

```
let guessMyNumber = 12;
let a = guessMyNumber++; // (this) alert (a); // 12
```

在`(this)`行中，*后缀*形式`guessMyNumber++`也增加`guessMyNumber`但返回*旧的*值(在增加之前)。所以，`alert`显示的是`12.`

> 如果不使用递增/递减的结果，则使用哪种形式没有区别

```
let guessMyNumber = 0;
guessMyNumber++;
++guessMyNumber;alert (guessMyNumber); // 2
```

如果我们想增加值*并且*立即使用运算符的结果，我们需要前缀形式

```
let guessMyNumber = 0;
alert( ++guessMyNumber); // 1
```

如果我们想增加一个值，但使用它以前的值，我们需要后缀形式

```
let guessMynumber = 0;
alert (guessMyNumber++); // 0
```

## **其他操作员之间的增减**

运算符`++/--`也可以用在表达式中。它们的优先级高于大多数其他算术运算。

```
let guessMyNumber = 1;
alert( 2 * ++guessMyNumber ); // 4let guessMyNumber = 1;
alert( 2 * guessMyNumber++ ); // 2, because guessMYNumber++ returns the "old" value.
```