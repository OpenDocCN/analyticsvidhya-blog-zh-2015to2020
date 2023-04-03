# 逻辑运算符

> 原文：<https://medium.com/analytics-vidhya/logical-operators-db00b8e9c66f?source=collection_archive---------24----------------------->

> 尽管它们被称为“逻辑”值，但它们可以应用于任何类型的值，而不仅仅是布尔值。他们的结果也可以是任何类型的。

![](img/dca653cdb43a77f7237a69eeadb0d21e.png)

照片由[费伦茨·阿尔马西](https://unsplash.com/@flowforfrank?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

JS 中有三个逻辑运算符。

*   `||`(或)
*   `&&`(和)
*   `!`(非)

在经典编程中，逻辑“或”只用于处理布尔值。如果它的任何参数是`true`，它返回`true`，否则，它返回`false`。

有四种可能的逻辑组合:

```
alert( true || true );   // true 
alert( false || true );  // true 
alert( true || false );  // true 
alert( false || false ); // false
```

大多数情况下，OR `||`用于一个`if`语句中来测试*给定条件中的任何一个*是否为`true`。

```
let workHours = 10;if (workHours < 10 || > 20){
alert ('office is close');
} 
```

我们可以通过更多的条件:

```
let workHours = 10;
let weekedn = true;if (workHours < 10 || > 20 || weekend){
alert ('We re closed'); // weekend}
```

> 或“||”找到第一个真值

OR `||`操作符执行以下操作:

*   从左到右计算操作数。
*   对于每个操作数，将其转换为布尔值。如果结果是`true`，停止并返回该操作数的原始值。
*   如果所有操作数都已求值(即所有都是`false`)，则返回最后一个操作数

例如:

```
alert( 1 || 0 ); // 1 (1 is true)
alert( null || 1 ); // 1 (1 is the first truthy value)
alert( null || 0 || 1 ); // 1 (the first truthy value)
alert( undefined || null || 0 ); // 0 (all falsy, returns the last value)
```

从变量或表达式列表中获取第一个`true`。

```
let myName = '';
let yourName = '';
let something = 'freshh';alert (myName || yourName || something || damn); // something
```

如果所有的变量都是假的，该死的就会出现！

# &&和

AND 运算符由两个&符号表示`&&`

`fullName = monk && honk;`

在经典编程中，如果两个操作数都为真，则返回`true`，否则返回`false`。

```
alert( true && true );   // true
alert( false && true );  // false
alert( true && false );  // false
alert( false && false ); // falselet hours = 12;
let minutes = 45;if (hours == 12 && minutes 45){
alert ('The time is 12:45');
}
```

与 OR 一样，任何值都可以作为 and 的操作数

```
if (1 && 0) { // evaluated as true && false   alert( "won't work, because the result is false" );
}
```

AND `&&`操作符执行以下操作:

*   从左到右计算操作数。
*   对于每个操作数，将其转换为布尔值。如果结果为，则停止并返回该操作数的原始值。
*   如果已经计算了所有操作数(即所有操作数都为真)，则返回最后一个操作数。

换句话说，如果没有找到，则返回第一个假值或最后一个值。

上面的规则类似于 OR。不同的是，AND 返回第一个 *falsy* 值，而 OR 返回第一个 *truthy* 值。

```
// if the first operand is true,
// AND returns the second operand: 
alert( 1 && 0 ); // 0 
alert( 1 && 5 ); // 5  
// if the first operand is false, 
// AND returns it. The second operand is ignored 
alert( null && 5 ); // null 
alert( 0 && "no matter what" ); // 0
```

我们也可以连续传递几个值。看看第一个错误是如何返回的:

```
alert( 1 && 2 && null && 3 ); // null
```

当所有值都为真时，返回最后一个值:

```
alert( 1 && 2 && 3 ); // 3, the last one
```

# ！(不是)

布尔 NOT 运算符用感叹号`!`表示。

语法非常简单:

```
result = !value;
```

运算符接受单个参数，并执行以下操作:

1.  将操作数转换为布尔类型:`true/false`。
2.  返回反数值。

```
alert( !true ); // false
alert( !0 ); // true
```

double NOT `!!`有时用于将值转换为布尔类型。

```
alert( !!"non-empty string" ); // true 
alert( !!null ); // false
```

> 也就是说，第一个 NOT 将值转换为布尔值并返回倒数，第二个 NOT 再次对其求逆。最后，我们有一个普通的值到布尔的转换。

> NOT `!`的优先级是所有逻辑运算符中最高的，所以它总是先执行，在`&&`或`||`之前。