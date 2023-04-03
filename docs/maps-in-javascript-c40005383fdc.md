# JavaScript 中的地图

> 原文：<https://medium.com/analytics-vidhya/maps-in-javascript-c40005383fdc?source=collection_archive---------30----------------------->

## 地图():

![](img/9c107e223410a0356c1f08ef43dff6f1.png)

## 我们要讨论什么话题？

1.  什么是地图？
2.  它使用了什么方法，我们如何向它插入或检索值？
3.  地图是如何工作的，循环的行为以及幕后发生了什么？

## 什么是地图？

这里的 Map 这个词，我指的是目前在现代 JavaScript 中使用的另一种新方法。在这里，我们将看到 Map 的经典数据类型。大多数情况下，Map 被用于循环概念中，这是一件好事，但是如果你直接使用所有这些概念，那么你将无法理解 Map 的经典数据类型。

因此，让我们启动代码编辑器，通过一些惊人的例子来理解 Map 的概念。

```
var myMap = new Map();
```

这几乎是一种构造性的方法，我们从原型和类似的东西中创建一个对象。

## 地图中使用的方法有哪些？

现在地图有很多属性，我想你应该马上看一下。所以无论你什么时候输入`myMap.`，你都会看到一个方法列表出现在你的屏幕上，比如 get，set，forEach，values 等等。其中一个被广泛使用的常用方法是`set` 和`get`方法，我想从它的名字就可以看出，你可能已经猜到了，这是一些默认的 setters 和 getters，意味着它可以向它们添加值，也可以从它们那里获取值。

## 如何在地图中插入值？

让我们继续尝试`set()`方法，一起享受其中的乐趣:

> 注意:它们几乎像物体一样工作，但是记住这一点，地图不像物体，它们是完全不同的。

```
var myMap = new Map();
myMap.set(1, "Ricky"); //myMap.set(any key, any value)
myMap.set(2, "Monty"); //myMap.set(any key, any value)
myMap.set(3, "Henna"); //myMap.set(any key, any value)
myMap.set(4, "Monta"); //myMap.set(any key, any value)console.log(myMap);
```

如果你试着在你的编辑器中运行这段代码，你会在屏幕上看到这样的输出。

```
Map(4) { 1 => 'Ricky', 2 => 'Monty', 3 => 'Henna', 4 => 'Monta' }
```

请注意这里，默认情况下，地图本身会告诉你有多少属性存储在其中，还有一点你可以在这里看到，他们使用了`=>`键，值对之间的这些箭头来区分，并让你明白它们不是你的常规对象。

## 如何从地图中检索值？

所以，接下来的事情就要看如何从这些地图中抓取数值了。是的，并不缺少获取它的方法，但是，首先，我们将尝试从这些方法中获取所有值，然后我们可以进一步挖掘，看看如何检索单个值。所以让我们开始吧。

**For…of 循环:**

经典的方法是在地图中使用`for...of`循环。让我们看看如何利用它们。

**单独获取密钥，使用 for…of 循环:**

```
var myMap = new Map();
myMap.set(1, "Ricky"); //myMap.set(any key, any value)
myMap.set(2, "Monty"); //myMap.set(any key, any value)
myMap.set(3, "Henna"); //myMap.set(any key, any value)
myMap.set(4, "Monta"); //myMap.set(any key, any value)//console.log(myMap);//get all Keys
for(let key of myMap.keys()){console.log(`key is ${key}`)}
```

这里我们声明了一个`let key`，然后我们使用`of`，然后使用`myMap`，为了从我的地图中获取所有的键，我们在上面使用了一个方法`keys(),`，注意这个`keys()`是我们在地图上调用的一个方法，就像`myMap.keys()`。然后在循环中，我们简单地记录变量 key，它实际上自动循环地图中可用的每一个键，然后在我们的屏幕上得到输出。

```
//Output:
key is 1
key is 2
key is 3
key is 4
```

**单独获取值，使用 for…of 循环:**

如果我们只需要地图上的值呢？然后我们只需要在地图上使用内置方法`values()`而不是`keys()`。我们要怎么做？简单地跟着我做下面的代码:

```
var myMap = new Map();
myMap.set(1, "Ricky"); //myMap.set(any key, any value)
myMap.set(2, "Monty"); //myMap.set(any key, any value)
myMap.set(3, "Henna"); //myMap.set(any key, any value)
myMap.set(4, "Monta"); //myMap.set(any key, any value)//console.log(myMap);//get all values
for(let value of myMap.values()){console.log(`value is ${value}`)}
```

现在这里我们只有值，但是现在问题是如果我需要它们两个在一起，我的意思是所有的键和值在一起。现在，一些困惑开始出现了。

**使用 for…of 循环获取键、值两者:**

所以，我们可以用同一套`for..of`循环，只需要在里面做一些调整，让我们一起来看看。

```
var myMap = new Map();
myMap.set(1, "Ricky"); //myMap.set(any key, any value)
myMap.set(2, "Monty"); //myMap.set(any key, any value)
myMap.set(3, "Henna"); //myMap.set(any key, any value)
myMap.set(4, "Monta"); //myMap.set(any key, any value)//get both keys,valuesfor(let [key, value] of myMap){
console.log(`Key is: ${key} and Value is: ${value}`)}
```

注意，这里我们以数组的形式声明了`let`，其中有一个键-值对，比如- `let[key, value]`，因为我们获取了两个键，值，所以我们不需要在`myMap`上有任何方法，它只是`myMap`。现在运行它，您将看到如下输出:

```
//Output:
Key is: 1 and Value is: Ricky
Key is: 2 and Value is: Monty
Key is: 3 and Value is: Henna
Key is: 4 and Value is: Monta
```

现在，你可能会问嘿*这有什么困惑？*真的非常简单容易。

## 使用 forEach()循环了解地图:

所以，让我也给你看看令人困惑的部分。如果`myMap`也可以访问`forEach()`会怎么样:

```
var myMap = new Map();
myMap.set(1, "Ricky"); //myMap.set(any key, any value)
myMap.set(2, "Monty"); //myMap.set(any key, any value)
myMap.set(3, "Henna"); //myMap.set(any key, any value)
myMap.set(4, "Monta"); //myMap.set(any key, any value)//testing with forEach
myMap.forEach( (key) => console.log(`${key}`));
```

当您运行它时，您可以看到如下输出:

```
//Output:
Ricky
Monty
Henna
Monta
```

在这里，你来了，你可以在你的输出窗口看到一些结果，但请注意，我在代码中要求`key`，但我得到的是所有的值，为什么呢？

**通过经典的方法处理地图，我的意思是用** `**for...of**` **循环我们可以很容易地得到键，但是如果我们用现代的方法，我的意思是用** `**forEach()**` **特此默认这个** `**forEach**` **循环是这样设计的，它假设，你将只使用值而不是索引来做一些事情。在这里的地图中，键被认为是索引，这就是为什么它不直接给你的关键。所以不要用** `**forEach()**` **循环的键，而要单独用值。**

```
**NOTE:** **forEach() loop always gives you the value first.**
```

现在，你知道内在工作是如何进行的了。但是，*如果你想要这两个(键，值)，那么也许你可以在* `*forEach()*` *中传递两个参数，比如:*

```
var myMap = new Map();
myMap.set(1, "Ricky"); //myMap.set(any key, any value)
myMap.set(2, "Monty"); //myMap.set(any key, any value)
myMap.set(3, "Henna"); //myMap.set(any key, any value)
myMap.set(4, "Monta"); //myMap.set(any key, any value)//testing forEach loopmyMap.forEach( (value, key) => console.log(`The value is: ${value} and the key is: ${key}`));
```

现在，如果您要运行这个，那么您可以首先看到值，然后看到来自`myMap`的密钥。

```
//OutputThe value is: Ricky and the key is: 1
The value is: Monty and the key is: 2
The value is: Henna and the key is: 3
The value is: Monta and the key is: 4
```

> **提示:**知道哪个循环先给什么东西非常重要，就像这里的`*for...of*`循环先给`*key*`，但是`*forEach()*`总是先给你值。

## 如何在“地图”中删除值？

除此之外，我们可以在地图上得到更多的乐趣，比如我们可以从地图上删除任何一对，只需将它的键传递给`delete()`方法，比如:

```
var myMap = new Map();
myMap.set(1, "Ricky"); //myMap.set(any key, any value)
myMap.set(2, "Monty"); //myMap.set(any key, any value)
myMap.set(3, "Henna"); //myMap.set(any key, any value)
myMap.set(4, "Monta"); //myMap.set(any key, any value)//delete method
myMap.delete(2);console.log(myMap);
```

所以在输出中，你可以看到一些差异。

```
//Output
Map(3) { 1 => 'Ricky', 3 => 'Henna', 4 => 'Monta' }
```

是的，值为 2 的键现在没有了。

## 结论:

最后，我只想说，地图真的非常强大，有趣，我认为你应该对它做一些更多的研究，因为它有很多方法，如删除，大小，条目，值，等等。由于很难在一篇文章中涵盖所有这些内容，因此我强烈建议去查看一下它的[文档](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map)，以便对它有一个更清晰的了解。

但最后，我已经试图涵盖它的几乎所有基础，我们可以把它总结为:

*   地图不是物体。
*   Set()方法基本上是用来插入一个键，值对。
*   `for...of`循环——一种经典的获取密钥、值的方式，它首先给出密钥。
*   `forEach()` loop- Morder 循环方式，总是先给出值。

到现在为止，你已经对地图有了足够的了解，了解了它的背景和基础知识，所以继续学习更多的知识吧，因为从长远来看这是很有用的。

希望你喜欢我的作品，如果是的话，欢迎鼓掌，如果你有任何疑问，请在评论区告诉我。

*谢谢大家，这一个就到此为止，让我们在新的一个中赶上。*