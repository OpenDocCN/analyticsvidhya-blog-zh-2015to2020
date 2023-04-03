# 学习并理解 Javascript 的 map 函数()

> 原文：<https://medium.com/analytics-vidhya/learn-and-understand-javascripts-map-function-agbejule-kehinde-favour-ac91e3f61fff?source=collection_archive---------11----------------------->

![](img/8c7ff32b62b5c0bfd3825caaa678392a.png)

嘶！嘿，你。是的，你——在那边。你想了解地图的功能吗？嗯，是吗？很好。我可以帮忙。

你只需要学习它的基础知识。
在本文中，我们将讨论 map 函数的工作原理、语法和一些例子。
我是 Agbejule Kehinde 恩宠。

说到这里，让我们直接进入主题吧！

![](img/d223bb668a8aa0b556db1803c0f152a7.png)

我们走吧！

# 地图功能有什么作用？

映射函数()是高阶函数。

高阶函数将函数作为参数并执行它。

map 函数遍历数组中的每一项，并执行赋予它的任务。

你可以说它就像一个 for 循环。
当使用 map 函数时，你总是需要创建一个新变量来包含将要形成的新数组。

它的答案作为一个新变量返回，因为你不能把新数组赋给旧数组变量。

# 语法

map 函数主要以一个**回调**(函数)作为实参，其他参数为:
1。电流值
2。指数；数组中正在处理的当前元素的索引和
3。数组。

最后两个参数是可选的，这意味着如果没有必要，可以不输入这些参数的值。

# 让我举几个例子

## 数组中数字的乘法

```
const numbers = [60, 15, -17, 3, 0]const multiplied = numbers.map(function(num)){
                     return num*2
                   }
console. log(multiplied)// expected answer = [120, 30, -34, 6, 0]
```

新的数组变量(乘以常量)采用前一个数组，并通过它进行映射

在映射函数括号中，我们有一个回调函数(function ),它接受当前值(num)。

**‘num’**由于 map 函数遍历数组，所以会随时间变化。

因此，从数组中取出一个项，执行将该项乘以 2 的函数并得到一个结果。

对数组中的剩余项执行此操作

代码似乎太长了，所以我想使用 es6 语法来缩短它。

```
const multiplied = numbers.map(num =>num*2)
```

# 造句

下面是一组名字

```
const Names = [Silvia, Chavez, zee, Alexis, Gerardo]
```

这个代码块是新赋值变量的 es6 格式代码

```
const sentence = names. map(name => name + ' is the best')

console.log(sentence) 
```

这是长码(正常格式)

```
const sentence = names.map(function(name){
                       return name + ' is the best'
                 } )
```

造出来的句子是；

```
// expected answer = [
                          Silvia is the best,
                          Chavez is the best,
                          Zee is the best, 
                          Alexis is the best,
                          Gerado is the best.
                     ]
```

它遍历 names 变量中的每个名字，获取名字并将其放入句子中，创建一个新的句子数组。

查看这篇关于填充函数的文章

[](https://favouragbejule.medium.com/understand-javascripts-fill-function-tutorial-agbejule-kehinde-favour-51fc4fcea0c) [## 了解 Javascript 的 fill 函数()(教程)| Agbejule Kehinde Favour。

### 所以你想学填充函数！

favouragbejule.medium.com](https://favouragbejule.medium.com/understand-javascripts-fill-function-tutorial-agbejule-kehinde-favour-51fc4fcea0c) 

***说了这么多，希望对你有帮助。***

如果你从中获得了价值，击碎那个鼓掌按钮，并帮助这个内容传播给更多需要这个价值的人。

如果你想获得更多的价值，请务必关注我，以获得更多这些价值包装的内容。

***感谢您的配合！***

一如既往，我是 Agbejule kehinde favour，我会在下一篇文章中看到你。

***您可以使用此*** 查看更多内容

[](https://favouragbejule.medium.com/) [## Favouragbejule 培养基

### 嘶！嘿，你。是的，你-在那边。你想了解地图的功能吗？是吗？很好。我可以帮忙。你只是…

favouragbejule.medium.com](https://favouragbejule.medium.com/) 

***祝你们平安！***