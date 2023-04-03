# JavaScript 中的真值和假值以及逻辑运算符

> 原文：<https://medium.com/analytics-vidhya/truthy-and-falsy-values-and-logical-operators-in-javascript-2b575ac6e3ab?source=collection_archive---------15----------------------->

![](img/1df9961a22eb8323ce082e30e84d16b5.png)

照片由[Sulhadin ney](https://unsplash.com/@sulhadin?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com/?utm_source=medium&utm_medium=referral) 上拍摄

本质上，值可以是真的，也可以是假的，这取决于它们在布尔环境中的求值方式。一种代数符号系统，用于通过二进制数 0(假)和 1(真)来表示逻辑命题。

![](img/7b7513d84fc911fbf7057db642edffaf.png)

# 虚伪的价值观

JavaScript 中有 6 个 **falsy** 值，这意味着当 JavaScript 中需要一个 boolean 值，并且给定了下面将要陈述的任何一个值时，它将始终计算为 **falsy。**

更明确地说，那些被强制为假的 6 个**假值**用于`if`块；

```
if (false)
if (null)
if (undefined)
if (0)
if (0n)
if (NaN)
if ('')
if ("")
if (``)
```

*   `0`、`0n`算一个，`'', "", ```算一个。

# 逻辑 AND (&&)运算符

在一个逻辑语句&& means 中，如果**对象是 falsy，该语句将返回 ***那个对象*** 。**

```
**let sulhadin = false && "human";// ↪ false**
```

**其他值可以被认为是真实的。**

# **真实价值**

**另一方面，可以指定几个**真值**作为 JavaScript 中**真值**的示例。**

**更明确地说，那些被强制为真的**真值**在`if`块中执行；**

```
**if (true)
if ({})
if ([])
if (42)
if ("0")
if ("false")
if (new Date())
if (-42)
if (12n)
if (3.14)
if (-3.14)
if (Infinity)
if (-Infinity)**
```

# **逻辑 OR (||)运算符**

**在一个逻辑语句中||的意思是，如果**对象为真，该语句将返回****对象，*** 否则该语句将返回 ***第二个对象*** 。*****

```
**let sulhadin = false && "human";// ↪ "human"let oney = true && "human";// ↪ true**
```

**一个值的真或假可以通过将它传递给布尔函数来查看，如下所示。**

```
**Boolean("") // false
Boolean([]) // true**
```

# **(不是不会)！！操作员**

**利用！操作符一次将反转布尔值(即 true 到 false，falsa 到 true)。如果使用！运算符再次将它转换回布尔值。**

```
**Boolean(!"") // true
Boolean(!!"") // falseBoolean(![]) // false
Boolean(!![]) // true**
```

**例子；**

**检查 if 条件中的值的快捷方式，只需要记录 true 或 false。**

```
**const sulhadin = "human";console.log(sulhadin); // ↪ "human"console.log(!sulhadin); // ↪ falseconsole.log(!!sulhadin); // ↪ true**
```

# **资源**

**[](https://developer.mozilla.org/en-US/docs/Glossary/Falsy) [## 福尔西

### 假值是在布尔上下文中遇到时被视为假的值。JavaScript 使用类型…

developer.mozilla.org](https://developer.mozilla.org/en-US/docs/Glossary/Falsy) [](https://developer.mozilla.org/en-US/docs/Glossary/Truthy) [## 真理

### 在 JavaScript 中，真值是在布尔上下文中遇到时被认为是真的值。所有值都是…

developer.mozilla.org](https://developer.mozilla.org/en-US/docs/Glossary/Truthy)**