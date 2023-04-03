# 在效果挂钩中使用 React 的状态挂钩，而无需无休止的重新渲染

> 原文：<https://medium.com/analytics-vidhya/using-reacts-state-hook-inside-the-effect-hook-without-an-endless-re-render-e9dbf09ec105?source=collection_archive---------8----------------------->

![](img/7274d726fbd9c76639af0adce22fd840.png)

照片由[Tine ivani](https://unsplash.com/@tine999?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

React 现在有钩子，我们可以用它来处理功能组件，而不是类组件。两个主要钩子`useState()`和`useEffect()`反映了类组件的状态和生命周期方法。

为了解释我们如何在一起使用这两个钩子时遇到问题，我将首先总结一下每个钩子是如何工作的。