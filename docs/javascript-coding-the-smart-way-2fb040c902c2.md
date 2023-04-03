# JavaScript——聪明地编码！

> 原文：<https://medium.com/analytics-vidhya/javascript-coding-the-smart-way-2fb040c902c2?source=collection_archive---------24----------------------->

## 让您的 JavaScript 更加智能和健壮的一些技巧。

![](img/719ff15dac1adef13f3c5230b43a722d.png)

[安德鲁·尼尔](https://unsplash.com/@andrewtneel?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍照

随着时间的推移，JavaScript 越来越受欢迎。甚至连布伦丹·艾希自己都没有想到这种愚蠢可爱的语言会如此受欢迎。因此，作为 JavaScript 开发人员，我们有义务将对智能 JavaScript 技术的学习传承下去。

> 正如道格拉斯·克洛克福特所说—
> “JavaScript 是我所知道的唯一一种人们觉得在开始使用之前不需要学习的语言。”

强调这句话——“人们觉得他们不需要学习”
拥有 4 年多的 JavaScript 编码经验，我仍然觉得有很多东西需要学习和适应。JavaScript 永远不会停止给你带来惊喜。

说够了！让我们现在开始行动吧。因此，我将在这里分享一些在日常编码中有用的经验。使用这些技巧和示例，您将能够编写更加智能和健壮的代码。

# 先决条件

本文面向已经熟悉任何格式的 JavaScript 代码(如 TypeScript、JSX 等)或者有这方面经验的开发人员。我假设您已经熟悉现代 JavaScript(后 ES6)语法、概念和其他基本的 JavaScript 要素。

如果你是一个有抱负的 Web 开发人员，并且不太适应最新的 JavaScript 生活标准。我建议你浏览下面的链接，尽快修改这些内容。

[](https://www.freecodecamp.org/news/write-less-do-more-with-javascript-es6-5fd4a8e50ee2/) [## JavaScript ES6 -写得更少，做得更多

### JavaScript ES6 带来了新的语法和令人惊叹的新特性，使您的代码更加现代，可读性更好。它允许…

www.freecodecamp.org](https://www.freecodecamp.org/news/write-less-do-more-with-javascript-es6-5fd4a8e50ee2/) 

ES6 和 ES7 语法是编写智能代码的基础。它给了你少写多做的力量。

# 1.类型检查

JavaScript 中的类型是最容易被误解的概念之一，理解它们是如何工作的很重要。

## a.**类型的**操作符

`typeof`操作符返回一个字符串，表示给定值的 JavaScript 类型。

JavaScript 中可用的不同数据类型

## b.相等运算符

等号运算符由两个等号组成`==`
JavaScript 具有弱类型特性。这意味着相等运算符转换`types`是为了比较它们。此外，当类型被转换时，性能也会受到影响。
例如，在与另一个数字进行比较之前，`string`必须转换为`number`。

所以，永远不要使用等式操作符，因为它可能会导致难以跟踪的错误。相反，使用严格的等式运算符`===`

现在，让我们在我们的智能代码中使用这些类型和等式检查。

## c.函数中的类型检查

如果参数不是数字，你抛出一个错误。

## d.类型转换

*   **铸造要数。如果您曾经使用过原生 html 表单，那么您可能知道从表单输入标签中读取值的痛苦。**

说明为什么需要类型转换的例子。

在这里，使用加号运算符会在将值赋给变量之前将其转换为数字。

*   **铸造成串。** 无需使用`toString()`方法。只需在前面加上一个空字符串，任何数字都可以转换成字符串。

只需将您的号码添加到空字符串中。

# 2.无效支票

根据我的经验，我可以说，如果我们在适当的位置添加适当的空值检查，许多损坏的脚本和错误是可以避免的。人们不应该总是盲目地相信那些吹牛大王。有错误，还有那些生产缺陷！空检查在 UI 中也很重要。

## a.原始值

以下是所有的`falsy`值-

*   `null`
*   `undefined`
*   `'’`空弦
*   `0`数量

但是这些是`truthy`—

*   `'0'`绳子
*   `[]`空阵
*   `{}`空的物体

好吧，我已经知道了！告诉我一些新的东西。告诉我如何使用这些？
好！让我们看看如何使用这个技巧—

不需要比较变量的错误值。除非您希望 null 或 undefined 有一些不同的行为。

## b.非原始值

我们知道空集合是真实的，但是如何快速检查它们呢？我们将借助这两个本地方法。

*   对于对象— `Object.keys()`
*   对于数组— `array.length`

不需要检查数组。长度> 0，因为长度不能是负数。所以它要么是 0(假)，要么是任何小于 2 的正整数(真)

如何检查空对象？提示—我们将再次使用相同的概念！

Object.keys 将在一个数组中返回给定对象自己的键。所以如果它是一个空数组，你知道对象是空的。

## c.使用逻辑运算符&&和||

*   **使用** `**&&**`
    如何安全地玩自己不太了解的物体。
    假设有一个嵌套的英雄对象。类似于-

定义英雄对象

但是如果你不确定你正在处理的数据格式，可能会有 API 的回应。让我们说，你的一些英雄喜欢保持神秘的气氛。所以您可能得不到它们的`email`或任何其他属性。

赋值前进行检查，如果没有找到，则赋一个默认值。以使您的脚本更健壮，更不容易出错。

相信我，这可能看起来很乏味，但它会把你从那些会破坏你该死的页面的生产和最后一分钟的错误中拯救出来。

**可选链接(ES2020)🚀**
在最近的 ECMA2020 草案中，已经引入了可选链接。ES2020 从版本 80 开始在谷歌 Chrome 中可用。
所以，上面的例子可以改成——

使用可选的链接会使代码更加简洁。

我们使用 JavaScript 的可选链接的方式有`objects`、`arrays`、`functions` 作为
`obj?.prop`
`obj?.[expr]`
`arr?.[index]`
`func?.(args)`

*   **使用或** `**||**` ***又名*短路操作符**

当用户未定义或为假值时，or 运算符将使用第二个字面值，直到找到非假值。

**零合并(ES2020)🚀**
ES2020 还引入了一种更好的方式来检查`null`和`undefined`。

空合并操作符将遍历列表并返回第一个不是`null`或`undefined`的项目。
需要注意的是，零合并运算符只寻找`null`或`undefined`值。`falsy`值被`null`合并运算符接受。
无效合并的示例-

零合并与 OR 运算符略有不同

## d.使用三元运算符(条件)

三元运算符的简单用例

三元条件很简洁，但是要避免编写嵌套的三元条件，因为这会使你的代码在认知上变得复杂和难以阅读。导致代码更容易出现难以跟踪的逻辑错误。

# 3.更好的迭代

当有人说迭代一个`array`或一个`object`时，你会想到什么？根据您想要执行的场景和任务，有许多数组函数和循环。但是如何决定什么时候使用数组方法或者简单的`for`、`for-of`循环呢？让我试着帮你

## a.大型阵列的性能迭代

但是当迭代大型数组时，最好使用经典的`for`循环来获得最佳性能。

将数组长度存储在局部变量中以提高性能。

将`array` `length`或`numbers.length`存储在变量中甚至会为 JavaScript 引擎节省一些开销。它确实可以将迭代的性能提高 50%(也取决于集合的长度)。

## b.为什么不使用 for-in 循环

`for-in`循环经常被错误地用于循环`array`中的项目。这总是很容易出错，因为它不是从`0`循环到`length-1`，而是在对象及其`prototype`链中的所有当前`keys`上循环。是的，它也穿过了`prototype chain`！

人们总是建议避免使用 for-in 循环来迭代数组

# 4.公用事业

一些代码片段，你可以在你的代码库中正确地使用它们来智能地完成家务。

## a.深度克隆阵列

避免使用旧的`JSON.parse(JSON.stringify(arr))`方式克隆阵列。解析一个`array`或`object`总是会涉及很多风险和资源。它只适用于没有`function`或`Symbol`属性的`Number`、`String`和`Object`文字。

使用递归克隆数组

## b.展平数组

有时我们需要展平一个数组，下面的例子会很有用

使用递归展平数组

## c.深度克隆一个对象

可悲的是，很难克隆一个嵌套很深的对象。所以我建议你尽可能使用外部库，比如——lodash 有一个 [cloneDeep](https://lodash.com/docs#cloneDeep) 方法。
另一种方法是实现自己的克隆功能。大约

## d.弄平一个物体

复杂嵌套的对象可能很难迭代，或者如果你直接将数据绑定到 UI，也可能导致`[object Object]`

## e.深度冷冻一个物体

JavaScript 中的对象是可变的，不管您是否将它们定义为常量变量。

## f.记忆函数以加速昂贵的计算

记忆化是一种优化技术，在这种技术中，代价高昂的函数调用被缓存起来，以便下次使用相同的参数调用函数时，可以立即返回结果。

计算阶乘的记忆示例

为了记忆一个函数，它应该是纯的，这样每次对于相同的输入返回值都是相同的。此外，记忆化是在增加的空间和增加的速度之间的折衷，因此仅对具有有限输入范围的功能有意义。

# 离别赠言

在这篇文章中，我谈到了一些我经常使用的技巧和窍门。这些是我辛苦学到的一些知识。但是如果你在日常工作中使用这些，它肯定会使你的代码更加智能、易读和健壮。
感谢阅读！我希望这篇文章对你有用。如果你注意到一些不正确的地方，或者可以解释得更好的地方，请随意发表评论。

# 资源

*   [https://www.30secondsofcode.org/](https://www.30secondsofcode.org/)
*   [http://bonsaiden.github.io/JavaScript-Garden](http://bonsaiden.github.io/JavaScript-Garden)
*   [https://www . freecodecamp . org/news/JavaScript-new-features-es 2020/](https://www.freecodecamp.org/news/javascript-new-features-es2020/)
*   [http://inlehmansterms . net/2015/03/01/JavaScript-memo ization/](http://inlehmansterms.net/2015/03/01/javascript-memoization/)
*   [https://www . site point . com/implementing-memo ization-in-JavaScript/](https://www.sitepoint.com/implementing-memoization-in-javascript/)