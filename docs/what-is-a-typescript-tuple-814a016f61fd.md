# 什么是类型脚本元组？

> 原文：<https://medium.com/analytics-vidhya/what-is-a-typescript-tuple-814a016f61fd?source=collection_archive---------22----------------------->

![](img/36c2bf112c5695532718d44b4092d160.png)

Typescript 改变了开发人员为 web 编写代码的方式。它通过将静态类型引入 JavaScript，以及大量 JavaScript 中没有或尚未包含在未来的 ECMAScript 规范中的特性，使 web 开发人员的生活变得更加轻松。这些很酷的特性之一是 T **uple** 数据类型。

# 什么是元组？

根据 Typescript 文档:“ [**Tuple 类型允许你用固定数量的元素来表示一个数组，这些元素的类型是已知的，但不需要相同。**](https://www.typescriptlang.org/docs/handbook/basic-types.html#tuple) “简单来说，一个元组可以在一个集合中存储多种不同类型的数据。您可以定义每个位置可以存储哪种类型的数据。

在下面的例子中，我声明了一个元组来保存字符串和数字数据类型

```
let employee: [string, number] //Declaring tuple
```

在声明一个元组之后，要给它赋值，必须传递正确的数据类型。

```
employee= ['David', 10] 
```

如果您向 tuple 传递不正确的数据类型，比如下面的例子，我向 string 传递一个布尔值，typescript 编译器将抛出一个错误。

```
employee = [true, 10]; //Error: ‘Type ‘true’ is not assignable to type ‘string’’
```

# 操作元组

如上面在 typescript 定义中所述，tuple 是一个数组，这意味着您可以访问您拥有的数组的所有方法，例如；**推送**、**弹出**、**查找**、**索引**、**过滤**、**拼接**、 **forEach** 等等。

![](img/31d1bce7a8ef929fc6ec0089b9ef4dfd.png)

**注意**:虽然元组可以被看作是成对的，但是值并不是按索引成对存储的，而是每个值都有自己的索引。在下面的代码中，为了访问我之前存储在元组中的值“David”和数字“10 ”,我将访问每个值，就像通过它们的索引访问常规数组一样。

```
console.log(employee[0]); //David
console.log(employee[1]); //10
```

关于元组要记住的另一件独特的事情是，当给元组赋值时，前几个值必须与定义的类型完全匹配。其后的任何值都可以按任意顺序插入。

例如，在声明 employee 的类型为[string，number]之后，我传递的前两个值的顺序必须是 string，然后是 number。**之后传递的任何值都可以是任意顺序的，并且可以是声明的任何预定义元组类型。**

```
let employee: [string, number]; //declaration
employee = ['David', 10]; //Valid
employee.push(77);
employee.push('Eva');The following would return an error
employee.push(true); //Error: Argument of type 'true' is not //assignable to parameter of type 'string | number'.
```

# 元组和数组有什么不同？

现在你一定想知道，如果一个元组像数组一样工作，并且拥有数组的所有方法，为什么我不能只使用' any '类型的数组来完成上面的例子呢？

这个问题的答案是，是的，你可以，但这将允许某些值，你可能不希望插入到你的数组。例如，如果我要用一个“any”类型的数组来做上面的例子，我将能够向该数组传递除字符串和数字之外的任何值。

```
employee: Array<any>;
employee.push('David'); //valid
employee.push(23); //valid
employee.push(true); // also valid 
```

元组的好处是它允许在数组中保存多种数据类型，同时还可以设置约束。

# 用例

我发现元组的使用在以下方面非常有用:

*   字典/记录键值对
*   当您希望一个方法返回多个值，并且不希望使用对象时。

在评论部分添加，你发现使用元组是一个更好的选择。

如果你喜欢这篇文章，一定要点击拍手按钮，看看我下面的其他文章！

*   [**角路由:命令式 vs pop state**](https://medium.com/p/7d254b495c54/edit)
*   [**异步编程(理论概括)**](https://ericsarpong.medium.com/asynchronous-programming-in-a-nutshell-theory-d5fd07cf3b22)
*   [**深度潜入 Javascript 图**](/@ericsarpong/deep-dive-into-javascript-map-object-24c012e0b3fe)