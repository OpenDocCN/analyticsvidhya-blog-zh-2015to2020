# 在 JavaScript 中析构数据

> 原文：<https://medium.com/analytics-vidhya/destructure-the-data-in-javascript-867d6121e9b3?source=collection_archive---------29----------------------->

在 javascript 中析构数据:

![](img/8fab6fefc8ef1c6269394ac50b92724b.png)

## 我们要讨论什么话题？

1.  什么在破坏数据？
2.  我们可以在哪里以及如何利用它？

## 1.什么在破坏数据？

**析构**是一种从存储在(可能是嵌套的)对象和数组中的**数据**中提取多个值的便捷方式

简而言之，这是关于如何挑选数据，这整个事情，在一个非常普遍的规模上也被称为数据的**解构**。

析构背后的整个思想是，只要记住这个概念，无论右边的数据类型是什么，都应该和左边的相似，然后你可能可以进行析构，注意这里我说的可能。

## 2.我们可以在哪里以及如何利用它？

很多人可能知道析构只能对对象进行，但这不是真的。析构可以在很多地方发生，这是 JavaScript 使用的一种非常常见的技术。我们只需要记住一个概念，正如我之前告诉你的，两边的数据类型应该相同。

## 用数组进行析构:

让我给你们展示一下，即使第一个例子本身也会给很多人带来一点惊喜。让我们启动代码编辑器，看看:

```
const user = ["abhinandan", 3,"admin"]; //[name, loginCount, role]var name = user[0];
var loginCount = user[1];
var role = user[2];console.log(`userName is: ${name}, has login: ${loginCount} times and his role is: ${role}`);
```

> **提示:**两侧的数据类型应该相同。

看起来很正常，您可能已经猜到了输出，但是您仍然可以在下面检查它:

```
//Output:"userName is: abhinandan, has login: 3 times and his role is: admin"
```

但是如果我说，同样的代码，我们可以用更少的代码行写，可以节省你的时间，我们怎么做呢？让我们一起来看看。

```
const user = ["abhinandan", 3,"admin"]; //[name, loginCount, role]var [name, loginCount, role] = user; //L.H.S(array) = R.H.S(array)console.log(`userName is: ${name}, has login: ${loginCount} times and his role is: ${role}`);
```

这里，我们利用了数据的析构，正如我们可以看到的，我们在两边都有相同类型的数据(数组),这意味着左边的变量将被右边的可用值填充。现在，如果再次运行这段代码，您将看到与之前相同的输出。

```
//Output:"userName is: abhinandan, has login: 3 times and his role is: admin"
```

## 用对象进行析构:

现在，大多数时候你不会看到数组对数据的析构，相反你会看到对象。因此，让我们来看看一些对象的工作示例。

```
const myUser = {
    name: "abhinandan",
    loginCount: 3,
    role: "admin"
}console.log(myUser.loginCount);
```

> **提示:**数据类型和名称应该与对象中的完全相同。

你可能已经看过一千遍了，但这仍然是最好的学习例子。所以如果你运行它，你可以在屏幕上看到打印数量。

```
//Output:
3
```

同样，如果你想节省一些时间，想减少它的行数，也许你可以去析构它。让我们看看它会是什么样子:

```
const myUser = {
    name: "abhinandan",
    loginCount: 3,
    role: "admin"
}
const {name, loginCount, role} = myUser; //object = objectconsole.log(role);
```

如果你运行这个，你可以在屏幕上看到这个角色。

```
//Output:
admin
```

到目前为止，我想你已经可以析构数据了，但是让我们再深入一步，看另一个例子来理解这个行为。

```
const myUser = {
    name: "abhinandan",
    loginCount: 3,
    role: "admin"
}
const {name, myLoginCount, role} = myUser; //object = objectconsole.log(myLoginCount);
```

这里，我们只是将左侧的变量名从`loginCount`替换为`myLoginCount`，让我们运行它并查看输出。

```
//Output:
undefined
```

**上面显示了一个** `**undefined**` **，为什么会这样？**

*为了析构数据，数据类型必须匹配，但同时，左侧提到的名称应该与右侧匹配。*

如果我们通过`console.log(role)`运行同样的代码，那么你肯定可以在你的输出屏幕上看到打印的`admin` ，这是毫无疑问的，因为这个角色是以我们在对象中提到的同样方式被提到的。

## 结论:

1.  析构不仅发生在对象中，也发生在数组和其他一些东西中。
2.  确保在对象的情况下名称是完全相同的，但在数组的情况下不是，它们的工作方式完全不同，所以没有硬性的这样的要求。

我希望，我已经给了你足够的关于破坏数据的信息，并且到现在为止你已经准备好了。所以，如果你喜欢，欢迎反馈，请随意鼓掌。

*谢谢各位，让我们在新的一年里再续前缘。*