# JavaScript 变量、字母和常量

> 原文：<https://medium.com/analytics-vidhya/javascript-var-let-and-const-99babd635b33?source=collection_archive---------33----------------------->

![](img/3ae12822ff8884492004768e4eb487cc.png)

*免责声明*:本文的目标是帮助 javascript 新手理解非常基本的区别，而不需要深入了解

现在我们开始吧💪

**介绍🥱**

2015 年之前，javascript 开发者习惯于使用 **var** 作为关键字来声明变量，生活很容易，但并不平静😅

由于使用 **var** 来声明变量，开发者不得不在许多领域进行斗争…

**重新申报**🤦‍♀️
信不信由你，使用 **var** 你可以多次使用同一个变量名而不会遇到任何错误，但是你必须为意想不到的结果做好准备😬。

想象以下情况:

```
function sayHi() {
 var name = “Our New User”
 var isLoggedIn = true
 if (isLoggedIn) {
    var name = “Sarah”
 }
 console.log(“Hi” + name) // Hi Sarah
}
```

你现在想到的第一件事是，那又怎样？！，我重新定义了变量，这有什么问题吗？！🤨

这根本不是问题，只要你知道那正是你想要做的，不只是你忘记了这个变量已经声明了 before🧐.

重声明的真正问题来自大型应用程序，一旦你忘记你以前使用过相同的变量名。

*免责声明* ✋:如果你有牢固的记忆，可以帮助你记住你在特定范围内声明的所有变量，这不成问题。

**范围**😵
上一行以单词 **scopE** 结束，在深入挖掘之前，让我们先了解什么是 SCOPE，把 SCOP 想象成一个**盒子**，在里面可以访问一些函数和变量。

使用 **var** 关键字声明的变量有可能是无限的，除非它们是在函数中声明的。

这意味着如果一个变量不在一个函数中，它将在整个应用程序中被访问😨。

现在试着将这一点与之前的一点联系起来，*重新声明*，
现在开发人员必须记住他们在全局/函数作用域中声明的所有变量，否则他们会发现自己陷入了意想不到的结果中。

想象以下情况…

```
function sayHi() {
 var name = “Our New User”
 var isLoggedIn = true
 if (isLoggedIn) {
    var name = “Sarah”
    console.log(“Hi” + name) // Sarah
 }
 console.log(“Hi” + name) // Sarah
}
```

if 块内部的日志是有意义的，因为我们正在记录这个块中定义的变量，但是 if 块外部的日志是突出问题的那个，它应该打印"*我们的新用户* ***"*** 在 if 块外部声明的变量的值，但是这里发生的是在 if 块内部声明的变量 **name** 完全替换了在 if 块外部声明的变量，这里我们必须提到提升。

**吊装**😧
提升是将变量和函数声明提升到其作用域顶部的过程。

用关键字 **var** 声明的变量被提升到全局/函数范围的顶部，并用未定义的值初始化。

将这一点与前一点联系起来，

```
function sayHi() {
 var name = “Our New User”
 var isLoggedIn = true
 if (isLoggedIn) {
    var name = “Sarah”
    console.log(“Hi” + name) // Sarah
 }
 console.log(“Hi” + name) // Sarah
}
```

我们现在可以发现这里发生了什么，在 if 块中声明的新变量被提升到函数的顶部，当然取代了原来的变量，这解释了为什么两个控制台日志打印出相同的结果。

现在，我们讨论了 js 开发人员花了很长时间解决的问题，现在让我们继续讨论 ES2015 是如何拯救我们的😉。

关于重新声明，使用 **let** 或 **const** 声明的变量不能在同一范围内重新声明。

提到作用域， **let** 和 **const** 都是**块作用域**，*代码块是{}内的任意一组代码，这意味着如果在{}内使用 **let** 或 **const** 声明了一个变量，那么在这些{}之外就不能访问它，尽管它们被提升到了它们作用域的顶部，即{}。*

现在让我们来看看我们的 sayHi 函数…

```
 function sayHi() {
 let name = “Our New User”
 var isLoggedIn = true // out of our scope
 if (isLoggedIn) {
    let name = “Sarah”
    console.log(“Hi” + name) // Sarah
 }
 console.log(“Hi” + name) // Our New User
} 
```

现在它像预期的那样工作了，在 if 块中声明的新变量**停留在 if 块中，它不影响 if 块外的变量**

但现在的问题是，使用 **let** 还是 **const** 哪一个🤔？！在这个问题的答案中，有一点需要提及，但为了简单起见，答案是，随你便😇，只要记住用 **let** 声明的变量可以更新，而用 **const** 创建的变量不能更新。

感谢阅读，如果你有任何问题或任何话题想让我写，我会很乐意帮助你，你的评论和建设性的笔记是非常欢迎❤️