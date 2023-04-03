# React Redux 解释道

> 原文：<https://medium.com/analytics-vidhya/react-redux-explained-ce8a981f8629?source=collection_archive---------14----------------------->

![](img/26d28f13a4da13d0707af4b62b7618b2.png)

反应还原

edux 是一个用 JavaScript 实现的状态管理库。此外，它可以被认为是一种全局方式的应用程序状态管理模式。它确保可预测的全局管理的应用程序状态。

> " Redux 不仅仅适用于 JavaScript . "

Redux 兼容 React、Angular、vanilla JavaScript 应用程序。但是将 Redux 与 React 一起使用有一个特殊的优势，因为它允许将用户界面(ui)实现为函数。在现代 React 应用程序中，有两种组件类型。

*   表象成分
*   容器组件。

由于表示组件从容器组件获得状态信息，所以我们对容器组件使用 Redux。容器组件中的状态与 redux 存储连接。

在深入了解 Redux 的细节之前，需要理解以下主要概念。

1.单个**存储**用于保存与应用程序相关的所有状态。

2.要修改状态:

a.**行动**应派遣到商店。

b.然后 **REDUCER** 应该根据动作中包含的类型和其他值更新状态。

3.要使用新状态，需要订阅(这是通过**回调函数**实现的)。

因此，让我们进入与 Redux 实现相关的解释。

## **动作**

Action 描述应该如何更新商店。这是一个物体。动作有两个特征，

*   类型:
*   其他属性(有效负载):

## 还原剂

减速器是功能。他们根据动作提供的细节更新商店。Flux 架构(REDUX 继承自 FLUX)支持不可变状态。因此，reducers 采用两个参数。第一个参数与当前状态相关，第二个参数是调度的操作。

*   **状态:**状态可以是任何东西。它也可以是一个对象。
*   **动作:**如上所述，这是一个对象。

成功插入上述两个参数将返回新状态。

```
Const reducerName = (state = initialStateValue, actionPassed){Switch (actionPassed.actionType){case condition1:return (actionToBePerformedBasedOnCase1)case condition2:return (actionToBePerformedBasedOnCase2)default:return (actionToBePerformedBasedUnknownActionValue)}}
```

## 商店

存储以全局方式保存整个应用程序的状态。如上所述，store 调度动作来改变/变更状态。订阅者可以收到有关状态更改的通知。此外，Redux 中还有检索整个状态树的功能。

```
import { createStore } from ‘redux’;
let store = createStore(nameOfTheCreatedReducer);
```

然后您需要使用 Redux Store 来启用您的应用程序。为此，您需要将主应用程序组件包装在提供者组件中，如下所示。以前创建的商店需要被传递到提供者的商店道具中。

```
render(
     <Provider store={store}> <App /> </Provider>, document.getElementById(‘root’));
```

那么为了从 React Redux 获得真正的功能，connect 函数应该使用。这将容器组件添加到 Redux 存储中。然后，它可以在整个应用程序中使用 Redux 存储中定义的状态。

但是在此之前，我们需要定义 connect 函数所需的两个参数。那些是 **mapStateToProps** 和 **mapDispatchProps。**

## **mapStateToProps**

它将 Redux 商店的状态映射到应用程序中使用的道具。实际上，这些道具是通过容器组件在应用程序中向下游传递的。简单地说，mapStaesToProps 将 Redux 存储中定义的状态与我们在应用程序中使用的状态进行映射。

```
const mapStateToProps = state => {
    return ({
       propOne : state.valueRelevantToProp1,
       propTwo : state.valueRelevantToProp2,
    });
}
```

然后我们可以创建启用 Redux 的容器组件，如下所示，

```
const container = connect(mapStateToProps)(containerComponent);
```

## **mapDispatchProps**

这允许我们指定组件打算分派哪些动作。在这里，我们在一个地方定义与组件相关的所有分派，然后根据需求将它们添加到我们的元素中。有两种方法定义 mapDispatchProps。

1.函数形式(允许更多定制)

2.对象速记形式(易于使用)

有一种替代方法可以做到这一点，而无需在 connect 函数中指定 mapDispatchProps。( [Alternativ](https://react-redux.js.org/6.x/using-react-redux/connect-mapdispatch#default-dispatch-as-a-prop) e)

但是这里我是用 mapDispatchProps 来解释的。可以定义如下。

```
const mapDispatchProps = dispatch => {
return ({
   actionToBeIncluded : () => dipatch({type: actionType}),
   actionToBeIncluded2 : () => dipatch ({type: actionType2}),
});
}
```

然后我们可以更新连接函数如下:

```
const container = connect(mapStateToProps, mapDispatchProps)(containerComponent);
```

作为最后一步，现在我们可以向组件中包含的元素添加已定义的操作:

```
<element event={ actionToBeIncluded }> Text </element>
```

这样，我们就成功地在应用程序中启用了 React Redux。在这里，我只是演示了与 Redux 集成相关的基础知识。如果你想深入研究，你可以参考下面的资源来获取更多的信息。

[https://redux.js.org/advanced/advanced-tutorial](https://redux.js.org/advanced/advanced-tutorial)

**编码快乐+谢谢！**