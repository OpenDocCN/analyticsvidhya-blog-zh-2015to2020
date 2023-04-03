# React 挂钩入门

> 原文：<https://medium.com/analytics-vidhya/getting-started-with-react-hooks-a21cfb09dc5d?source=collection_archive---------17----------------------->

## 如何使用 useState()、useEffect()和 useContext()

![](img/63fb93e2d41fb6a608b7ccbaf966b388.png)

Joshua Aragon 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

钩子是特殊的函数，它允许我们在不写类的情况下做各种事情。在钩子发挥作用之前，我们只能在类组件内部使用状态和生命周期方法。但是有了 React 钩子，现在我们可以用功能组件来做几乎所有的事情，从呈现 UI 到以一种非常简洁的方式处理状态和逻辑。

我们还应该记住，不要在循环、条件或嵌套函数中使用钩子。钩子应该用在功能组件的顶层。

在这篇博文中，我将指导你如何使用 3 个最常用的 React 钩子。

让我们开始吧。

# 1.使用状态()

钩子允许你在你的功能组件中添加和操作状态。为了在组件中使用它，您需要导入它。

```
import {useState} from 'react'
```

然后使用数组析构，我们可以简单地编写如下代码。

```
const [state, setState] = useState(initialState)
```

您可以将初始状态传递给`useState()`，它会返回保存当前状态值的状态变量和一个处理该状态值的 setState 函数。

# 2.useEffect()

`useEffect()` hook 让我们处理生命周期方法中的逻辑。这意味着无论什么时候影响到你的组件，钩子都会被调用。这些影响在 React hooks 文档中被称为副作用。

首先，您需要导入如下内容。

```
import {useEffect} from 'react'
```

对于`useEffect()`我们应该传递一个函数，我们称它为效果。我们可以提供一个数组作为第二个参数。但是如果我们不向`useEffect()`传递一个数组，组件将会重复重新加载。

```
useEffect(() => {
    console.log("Function is being called repeatedly.");
})
```

当您将一个空数组作为第二个参数传递时，arrow 函数将只在第一次呈现时被调用。

```
useEffect(() => {
    console.log("Function has been called only once.");
},[])
```

当我们为第二个参数传递一个数组时，`useEffect()`检查数组值是否已经被改变。所以只有当值不同时，才会调用 arrow 函数。

# 3.useContext()

`useContext()` hook 提供了在功能组件中轻松访问上下文的方法。

`useContext()` hook 与上下文 API 协同工作。上下文 API 用于 React 中的状态管理。它被设计成在我们必须处理需要在我们的应用程序中全局使用的数据时使用，例如主题和首选语言。通过使用上下文 API，我们可以在组件间共享数据时避免钻取。

可以使用`React.createContext`创建上下文。

```
const MyContext = React.createContext(defaultValue)
```

提供者组件用于允许消费组件订阅上下文中的更改。消费组件应该由提供者组件包装。

```
<MyContext.Provider value = {anyValue}></MyContext.Provider>
```

然后在消费组件中使用`useContext()`。`useContext()`可以导入如下。

```
import {useContext} from 'react'
```

`useContext()`接受一个上下文对象并返回该对象的上下文值。所以作为参数，将上下文对象传递给`useContext()`，然后我们可以使用组件中的值。

```
const value = useContext(MyContext)
```

现在让我们试着用一个简单的例子使用`useContext()`来演示如何将颜色主题应用到应用程序中。首先，让我们创建上下文文件。

然后，由于我们希望为整个应用程序添加颜色主题，我们应该在应用程序组件中使用上下文提供程序。

这里我们可以通过提供主题作为`ThemeContext.Provider`的值来改变主题。如果想要深色主题，可以应用`value={themes.dark}`。

最后，我们可以在组件中使用`useContext()`来消费上下文。

我希望你能很好地理解如何在 React 中使用`useState()`、`useEffect()`和`useContext()`钩子。

编码快乐！