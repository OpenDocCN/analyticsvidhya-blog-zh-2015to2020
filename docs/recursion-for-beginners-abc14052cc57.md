# 初学者递归

> 原文：<https://medium.com/analytics-vidhya/recursion-for-beginners-abc14052cc57?source=collection_archive---------27----------------------->

对于初学者来说，递归一开始可能有点难以理解，但它是一个非常重要的编程概念。递归可能不会用很多次，但在某些需要的情况下，它确实很有用。

**什么是递归？** *递归*是函数调用自身的过程。*递归*用于返回一个问题的解，这个问题依赖于同一个问题子集的解。
递归函数有两个主要部分:

a. ***一个终止条件*** ***(基本情况)-*** 这是导致函数停止调用自己直到无限的退出条件。一旦满足基本情况条件，它就跳出循环并停止函数调用迭代。

b. ***递归 case -*** 这是函数调用自身的部分。

让我们用 for 循环找出一个数的阶乘。
记住 5 的阶乘，5！= 5*4*3*2*1

```
function factorial (n) {
  let total = 1;
  for ( let i=n; i>0 ; i--){
     total= total * i
  }
  return total 
};factorial(5);  //this will print 120
```

现在让我们用递归来写同一个函数:

```
function factorialRecursion (n) {
 if (n === 1) return 1;                //this is the base case
 return n * factorialRecursion (n-1)   //this is the recursive case
};factorial(5)  //this will print 120
```

上面代码片段中的“if”条件是停止迭代的条件。函数“factorialRecursion”在内部尽可能多地调用自己，直到满足基本情况下的条件(当 n 等于 1 时，返回 1，而不是再次调用该函数)

使用*递归*意味着更短的代码行，如上例所示，但是它真正的亮点是当你有大量的嵌套循环时。用普通循环结构难以迭代执行的操作可以用递归来解决，尤其是在嵌套循环的深度未知的情况下。让我用另一个例子来说明。在有孩子、孙子和曾孙的家谱中，我们希望打印出每个孩子的名字。用 for 循环来做这件事即使不是不可能，也是非常困难的，但是递归可以解决这个问题。

家谱对象如下图所示；

```
const tree = {
  father: 'David',
  children : [
   {
   name : 'Charles',
   children : [{name: 'John', children :[]}]
   },
   {
   name : 'Esther',
   children : [ {name: 'Joy', children :[]},
                {name: 'Ben', children :[]}
              ]          
   }
  ]
 };
```

在上面的家谱对象中，大卫有两个孩子查尔斯和埃丝特，他们分别有一个和两个孩子。要打印 David 的所有子子孙孙的名字，使用“for loops”几乎是不可能的。
使用递归-

```
function printName (t) { 
 if ( t.children.length === 0 ) {      //the base case
   return 
 } t.children.forEach (child => {
   console.log(child.name)
   printName (child)                   //the recursive case
 })
};printName(tree);//will print out the names - Charles, John, Esther, Joy and Ben.
```

来解释一下上面的代码片段:
***基础案例***——如果子数组的长度为空，返回即不再调用该函数。
else
***递归 case-*** 循环遍历并 console.log 孩子的名字，再次运行该函数，直到所有名字都打印在控制台上。

想象一下，如果 family 对象有更多的子对象和孙对象创建嵌套循环，事情会变得稍微复杂一些，但是递归将有助于解决这个问题。

n:B——在你的控制台上复制代码片段并运行它，看看它是如何实现的。

我希望这篇文章是有帮助的。
编码快乐！