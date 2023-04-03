# JavaScript 中该关键字的介绍

> 原文：<https://medium.com/analytics-vidhya/introduction-to-this-keyword-in-javascript-78d97617dff0?source=collection_archive---------24----------------------->

![](img/193cf33683e7e77f6bdf98de192e3560.png)

照片由 [sydney Rae](https://unsplash.com/@srz?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

**免责声明**:我不是 JavaScript 专家。我只是分享一下我对`this`关键词用法的学习经验。所以，让我们开始吧。

> 记住这句台词
> 
> **为所有常规函数调用，** `**this**` **指向窗口对象。**

现在，让我们首先了解常规函数调用。

**例 1:**

```
// Let's define our function firstfunction sayHello() {
    console.log("Hello Reader!");
}// Now let's call our functionsayHello(); // This type of function call is called Regular Function call.**Output:
Hello Reader!**
```

上面的例子很容易理解，因为它只包含一个函数定义和一个函数调用。

**例 2:**

```
// Defining our functionfunction funcGenerator() {
    return function() {
      console.log("Hello Reader");
    }
}// Assigning a variable to a returned functionlet generatedFunction = funcGenerator();//Regular Function callgeneratedFunction(); **Output:
Hello Reader!**
```

在这个例子中，我们有一个返回函数的函数。因此，我们可以将一个变量赋给返回的函数，并用该变量调用该函数。

现在，让我们回到`**this**`关键词。

`**this**` keyword 就是 JavaScript 中引用一个对象的关键字。

```
console.log(this);
```

如果你只是试图在浏览器的控制台或节点中运行这一行代码，`**this**`关键字将指向全局对象，即浏览器中的窗口对象和节点中的空对象。

```
var user = {
  firstName: "Arteev",
  lastName: "Raina",
  printName: function() {
       console.log(this);
   }
}user.printName();**Output:
Object { 
  firstName: "Arteev", 
  lastName: "Raina", 
  printName: printName() 
}**
```

记住，`user.printName();`不是常规的函数调用。因此，有一点很清楚`**this**`将不是指全局对象而是现在呢？`**this**`应该指哪里？

在这种情况下，`**this**`将引用方法`printName()`的所有者，即`user`对象。因此，`user`对象登录到控制台。

因为在这种情况下`**this**`关键字指的是用户对象，这意味着我们可以使用`**this**`关键字获得用户对象的属性，就像现在我们可以编写`this.firstName`一样。

现在，让我们打印我们的名字。我打印我的，你试着打印你的。

```
var user = {
  firstName: "Arteev",
  lastName: "Raina",
  printName: function() {
      console.log(this.firstName + " " + this.lastName);
   }
}user.printName();**Output:
Arteev Raina**
```

现在，让我们举另一个例子来澄清我们的理解

```
var testObject = {
  printThis: function() {
       return function() {
           console.log(this);
       }
    }
} var func = testObject.printThis();
func();**Output:
Window Object in Browser
        (or)
Empty Object in Node**
```

> 你一定想知道为什么`this`指的是`window object`。为什么它指的不是`testObject`。但是，我们调用一个函数，它返回另一个函数，我们把这个函数保存在变量`func`中，然后调用`func()`，这也是一个常规函数调用，在常规函数调用中，这个关键字引用窗口对象。**站在圆点前的对象就是*这个*关键字将要绑定的对象。**