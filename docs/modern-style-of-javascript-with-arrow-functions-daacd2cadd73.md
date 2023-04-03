# 带有箭头功能的现代风格 Javascript

> 原文：<https://medium.com/analytics-vidhya/modern-style-of-javascript-with-arrow-functions-daacd2cadd73?source=collection_archive---------19----------------------->

## Javascript 中箭头函数的完整解释，以及它如何帮助开发人员编写灵活一致的代码。

![](img/088ef3e311757997aed5f12e78aae05a.png)

来源:谷歌

B 在深入了解箭头函数的概念之前。我想讨论一下旧的 Javascript 传统方法的令人兴奋的语法。ES6 中引入了箭头函数，这是一种编写 javascript 函数的现代风格。由于技术日益现代化，javascript 也日益现代化。

在我们传统的 Javascript 编程中，函数非常容易实现和使用。一个基本的 sayHello 函数将会是这样的—

```
function sayHello(name){
      return "Hello " + name; 
}// Function Call
console.log(sayHello("Javascript"));
```

上面的程序会生成`Hello Javascript.`的输出，传统的 javascript 函数很容易理解和实现。但是说到现代 javascript 函数式编程风格。有点诡异，难以理解。初学者在第一次学习时可能会有点困难，因为语法不是那么容易，但当我们用箭头函数弄脏手时，我们肯定会喜欢它。

## 箭头函数的特征

因为箭头函数对 javascript 社区来说是新的。毫无疑问，它应该比现有的方法具有更好的品质，这样才能对开发者有所帮助。箭头功能的最重要特征是—

1.  带有主体的箭头函数没有隐式 return 语句。
2.  它自动将`this`绑定到周围的代码上下文。
3.  它没有传统的 javascript 代码冗长。
4.  它没有原型属性。
5.  与传统的 Javascript 函数不同，箭头函数没有内置的参数属性。但是我们可以借助 rest 参数来实现。
6.  箭头函数对于回调、承诺或 map、reduce 和 forEach 等方法更方便。

现在，我们将尝试在箭头函数的帮助下重新实现旧的 javascript 函数。语法如下所示—

```
sayHello = name => "Hello " + name;// calling the function
console.log(sayHello("Javascript"));
```

上述程序的输出将与之前编写的代码相同。

**带零参数的箭头函数**

没有参数的箭头函数的语法—

```
helloWorld = () => "Hello World.!";console.log(helloWorld()); 
//OUTPUT:  Hello World.!
```

**带一个参数的箭头函数**

具有单个参数的箭头函数可以通过两种方式实现—

```
squareValue = n => n * n;console.log(squareValue(3));
// OUTPUT: 9
```

或者我们也可以用不同的方式创建上面的函数—

```
squareValue = (n) => n * n;console.log(squareValue(3));
// OUTPUT: 9
```

输出不会改变。我们可以声明带括号或不带括号的参数。

**带多个参数的箭头函数**

多参数的声明将非常简单——

```
add = (number1, number2)=>{ return number1 + number2 };console.log(add(3,2));
//OUTPUT: 5
```

**内置参数属性的问题**

在传统的 javascript 函数中，我们有一个名为 arguments 的内置属性，它帮助我们在默认情况下获取可变数量的参数，这是一个有趣且有用的属性。但是 Arrow functions 不支持该属性。

```
**// Traditional Javascript Approach
function addValues(){** **sumValue = 0;** for(let i=0;i<arguments.length;i++){ **sumValue += arguments[i];** **}** return sumValue;**}**console.log(addValues(1,2,3,4,5));console.log(addValues(10,20,30));
```

在上面的代码中，有一个 argument 属性帮助我们处理`n`多个函数参数。这有助于我们在某些情况下产生动态结果。上述代码的结果将是—

```
console.log(addValues(1,2,3,4,5));  // 15console.log(addValues(10,20,30));   // 60
```

但是箭头函数用下面的语法方法解决了这个问题—

```
**calc = (...args) => {** **sumValue = 0;
         for(let i=0;i<args.length;i++){
             sumValue += args[i];
         }** return sumValue;
**};**console.log(calc(5,3,4,5));
// OUTPUT 17 
```

我们还可以借助 reduce 方法减少行数，如下所示—

**用箭头功能减少方法**

Arrow 函数与 reduce 函数一起使用会更有效，这使得开发人员的工作变得更加容易。

```
function sum(...args) {
  return theArgs.reduce((prev, curr) => {
    return prev + curr;
  });
}console.log(sum(1, 2, 3));
//OUTPUT: 6console.log(sum(1, 2, 3, 4));
// OUTPUT: 10
```

# 结论

我希望你对箭头功能和它为什么如此受欢迎有一个清晰的认识。我非常详细地讨论了关于箭头函数的所有内容。

**忙碌的人们，你们好，我希望你们在阅读这篇文章时感到愉快，也希望你们在这里学到了很多！我希望你在这里看到了对你有用的东西。下次再见！😉🤓**

**玩得开心！不断学习新的东西，不断编码解决现实世界的问题。😇**