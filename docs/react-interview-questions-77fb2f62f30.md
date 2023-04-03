# 回应面试问题

> 原文：<https://medium.com/analytics-vidhya/react-interview-questions-77fb2f62f30?source=collection_archive---------28----------------------->

![](img/0fcd32738c3c1b6dab8271abf138ffb7.png)

图片来自谷歌

今天的话题

```
1\. Closures2\. How do React hooks work?3\. useMemo Hook4\. Dependency on useEffect hook5\. useCallback6\. state7\. Custom Hooks8\. Regular functions vs arrow functions9\. useState10\. useEffect
```

> **主题 1:** 闭包

当一个函数能够记住并访问它的词法范围时，即使这个函数是在它的词法范围之外执行的，这就叫做闭包。

> **话题 2:**React 挂钩是如何工作的？

钩子是在用户界面中封装有状态行为和副作用的一种非常简单的方法。它们最初被引入 React，并被 Vue、Svelte 等其他框架广泛采用，甚至被用于一般的函数式 JS。然而，它们的功能设计需要对 JavaScript 中的闭包有很好的理解。

> **话题三:**使用备忘录挂钩

useMemo 返回一个记忆值。
传递一个“创建”函数和一个依赖数组。useMemo 只会在其中一个依赖关系发生变化时重新计算 memoized 值。这种优化有助于避免每次渲染时进行昂贵的计算。
如果没有提供数组，将在每次渲染时计算一个新值。

> **主题 4:** 对 useEffect 挂钩的依赖

我说的是 useEffect hook 的第二个参数。如果我们改变第二个参数的值，useEffect 钩子将刷新。

示例:

```
useEffect(() => {
    console.log(`You clicked ${count} times`)
}, count)
```

这里，如果我们改变 count 的值，useEffect 钩子将刷新。

> **主题 5:** 使用回调

这些钩子防止了不必要的重新渲染，使得我们的代码更加高效。它返回一个记忆的回调

示例:

```
const memoizedCallback = useCallback(() => {
    doSomething(a, b);
}, [a, b],)
```

> **话题 6:** 状态

我们在类组件中使用**‘state’**。但是在功能组件中，我们使用了 useState 钩子而不是 State，并且**‘useState’**的工作与**‘state’**相同。

示例:

```
const [value, setValue] = useState(initialValue);
```

这里的值是类组件中的状态。要改变状态的值，我们可以使用 setValue 函数

> **话题七:**定制挂钩

定制钩子允许我们创建可以跨不同组件重用的功能。当然，你可以只使用函数来重用功能，但是钩子带来的好处是能够“挂钩”到组件生命周期和状态之类的东西。这使得它们在 React 世界中比常规函数更有价值。要创建自定义挂钩，挂钩名称将以“use”开头。

```
import React, { useState } from "react";const useInputValue = initialValue => {
    const [value, setValue] = useState(initialValue);

    return {
        value,
        onChange: event => {
            setValue(event.target.value || event.target.innerText);
        };
    };
};export default useInputValue;
```

> **话题 8:** 常规函数 vs 箭头函数

箭头功能是 ES6 中引入新功能。尽管常规函数和箭头函数的工作方式相似，但它们之间也存在一些差异。看这个例子:

```
const person = {
      name: "Khan",
      func1:() => {
          console.log("From arrow function: " + this.name); 
          // no 'this' binding here
      },
      func2(){
          console.log("From regular function: " + this.name); 
          // 'this' binding works here
      }
}person.func1()  // From arrow function: undefined
person.func2()  // From regular function: Khan
```

让我们看另一个例子:

***常规功能:***

```
const person = {
    show(){
        console.log(arguments);
        // { '0': 1, '1': 2, '2': 3 }
    }
};person.show(1, 2, 3);
```

***箭头功能:***

```
const person = {
    show : () => {
        console.log(...arguments);
        // arguments is not defined
    }
};person.show(1, 2, 3);
```

Arguments 对象在箭头函数中不可用，但在常规函数中可用。

> **主题 9:** 使用状态

useState 是一个内置的钩子，我们需要从 react 包中导入它。我们可以在功能组件中使用这个钩子。有一个变量和一个函数可以改变 useState 钩子中变量的值。让我们看一个例子:

```
import React, { useState } from 'react';function App() {
    const [count, setCount] = useState(0); return (
        <div>
            <p>You clicked {count} times</p>
            <button onClick={() => setCount(count + 1)}>
                Click here to increase count
            </button>
        </div>
    );
}
```

这里我们声明了一个新的状态变量，叫做“count”。为了改变变量，我们声明了“setCount”函数。

> **话题 10:** 使用效果

useEffect 是一个内置的钩子，我们需要像 useState 一样从 react 包中导入它。useEffect 是一个钩子，用于封装有“副作用”的代码。它接受一个函数作为参数。该函数在组件首次渲染时运行，并在随后的每次重新渲染/更新时运行。React 首先更新 DOM，然后调用传递给 useEffect()的任何函数。让我们看一个例子:

```
import React, { useState, useEffect } from 'react';function App() {
    const [count, setCount] = useState(0); useEffect(() => {
        console.log(`You clicked ${count} times`)
    }, count) return (
        <div>
            <p>You clicked {count} times</p>
            <button onClick={() => setCount(count + 1)}>
                Click here to increase count
            </button>
        </div>
    );
}
```

每当 count 的值改变时，useEffect 就会刷新。

*更多信息，请访问:*

[](https://reactjs.org) [## react——用于构建用户界面的 JavaScript 库

### React 使得创建交互式 ui 变得不那么痛苦。为应用程序中的每个状态设计简单的视图，并反应…

reactjs.org](https://reactjs.org) 

今天到此为止。