# JavaScript ES6 面试备忘单

> 原文：<https://medium.com/analytics-vidhya/javascript-es6-interview-cheat-sheet-e52ddc331cb2?source=collection_archive---------12----------------------->

![](img/556e02ced0a77eed4cb829ec6f4e8f35.png)

由 [Kelly Sikkema](https://unsplash.com/@kellysikkema?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在 [Unsplash](https://unsplash.com/s/photos/idea?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

在最近的一次采访中，有人问我什么是 JavaScript…我愣住了。在过去的 6 个月里，我一直使用 JavaScript 编写代码，但是我从来没有想过要定义它。所以在我开始解释 ES6 (ECMA 脚本 6)的一些关键特性之前，我想我应该给你一个 JavaScript 的定义，这样你就不会像我一样呆住了！

> J avaScript 是一种动态、高级、轻量级的编程语言，最常用于自动化和制作网页动画。它是一种松散类型的同步语言，具有面向对象的能力。

Javascript 符合旨在创建通用脚本语言的 ECMAScript 规范。2015 年 6 月发布的 ES6 带来了新功能，使您的代码更具可读性，让您可以编写更少的代码，做更多的事情。让我们来看看一些新功能…

# **Let 和 Const:**

*   `const`是一个**不可变变量**——它不能被重新赋值。
*   `let`是一个可变的**变量—** 可以被重新赋值并取一个新值。
*   `let`和`const`都是封锁范围(仅在其范围内可用)。
*   最好使用`const`而不是`var`，因为`var`是“提升”的。提升是 JavaScript 的默认行为，即在代码执行之前将所有声明移动到作用域的顶部，这意味着 JavaScript 可以在声明之前使用组件

# 箭头功能:

*简单地说，它们是速记和更准确的写作方式的功能。*

箭头函数由参数和箭头(`=>`)以及函数体组成。

```
//ES5
var add = function(x,y){
    return x + y;
}//ES6let add = (x,y) => { return x + y };//or even shorterlet add = (x,y)=> x + y;
```

对于上面的例子，它看起来并没有很大的不同，但是当你写更复杂的函数时，它变得更容易阅读。

箭头功能的主要优点是:

*   它们减少了代码的大小，使其更容易读写。
*   对于单行函数，return 语句是可选的。
*   词汇绑定上下文。
*   对于单行语句，函数大括号是可选的。

# Rest 运算符:

*   rest 操作符提高了处理多个参数的能力。
*   它由三个点(…)表示。
*   本质上，它获取数据并将其放入一个数组中，这意味着您可以调用带有任意数量参数的函数。

```
let sum = (...args) => {
    let total = 0;
    for (let i of args) {
       total += i;
    }
    console.log("Sum :" + total)
}total(10, 20, 30) //=> Sum: 60
```

# 扩展运算符:

*   扩展运算符允许数组之间的连接。
*   与 rest 操作符类似，它由三个点(…)表示，但是请注意，它们的功能完全相反。
*   这有助于轻松创建数组或对象的副本。

```
let arr1 = [4, 5, 6]let arr2 = [1, 2, 3, ...num2, 7, 8, 9] console.log(arr2) //=> [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

# 承诺:

*   承诺是使用异步编程最简单的方法。以前你可以使用回调，但你可能会陷入回调地狱，而承诺克服了这个问题。
*   承诺可以是**待定:**这是每个承诺的初始状态，意味着结果还没有计算出来。
*   承诺可以**兑现:**运算已经计算完毕。
*   可以**拒绝承诺:**运算导致计算过程中出现故障。
*   一旦承诺达到**兑现**或**拒绝**的状态，就变得不可更改。
*   **Promise()** 构造函数接受两个参数:一个 **rejected** 函数和一个 **resolve** 函数，基于异步操作，它将返回第一个参数或第二个参数。

# 模板文字:

*   提供创建多行字符串和执行字符串插值的简单方法。您不再需要使用`+`来连接字符串。
*   它们包括使用反斜杠(`` ` ),将表达式放在大括号{}内，前面有一个美元符号。

```
let str1 = "Hello"let str2 = "World"let str = `${str1} ${str2}!`console.log(str) //=> Hello World!
```

这绝不是 ES6 特性的完整列表，但我希望它对面试准备有用！感谢您的阅读:)