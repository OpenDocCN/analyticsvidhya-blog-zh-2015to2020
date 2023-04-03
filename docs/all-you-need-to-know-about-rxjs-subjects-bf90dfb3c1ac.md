# 关于 RxJS 主题你需要知道的

> 原文：<https://medium.com/analytics-vidhya/all-you-need-to-know-about-rxjs-subjects-bf90dfb3c1ac?source=collection_archive---------6----------------------->

![](img/997091274d1f0782634138f82f9ac95e.png)

主体是一种特殊类型的可观察对象，它在观察者之间共享一条执行路径。

把这想象成一个人在满是人的房间里对着麦克风说话。他们的信息(主题)被同时传送给许多人(观察者)。这是多角色扮演的基础。典型的单播可观察性相当于一对一的对话。

默认情况下，可观察对象是单播的，这实质上意味着发出值的源可观察对象只能被一个订阅者订阅，这不是很有用，因为对于实时 UI 应用程序，我们希望通过同时订阅值来在不同的地方显示相同的值。主题通过启用多播解决了这个问题。

每当一个主题被一个新的观察者订阅时，该观察者就在该主题的注册表中注册，这样，每当它发出下一个值时，就会从该主题接收新的值。如果一个订阅者订阅了一个主题，它不会影响有多少其他订阅者连接到同一个主题。它类似于本地的 addEventListener()方法，该方法不知道有多少事件侦听器正在侦听单个 DOM 元素上触发的同一事件。

主语是宾语有**下一个(v)** 、**错误(e)** 和**完成()**的方法。调用 next()方法来设置 Subject 的下一个值，在发生错误时调用 error()，而 complete()方法用于通知源没有更多的值要发送给观察者。

下面列出了 4 种类型的主题。

1 .主题

2 .行为主体

3 .重播主题

4 .异步主题

主题的上述变体都属于可观察类型，除了它们发出的数据之外，非常相似。

下一节将给出一个非常简单的例子。

```
const subject = new Subject<number>();subject.subscribe({next: (val) => console.log( `observerA: ${ v } ` )});subject.subscribe({next: (val) => console.log( `observerB: ${ v } ` )});const observable = from([ 1 , 2 , 3 ]);observable.subscribe(subject);
```

上述代码片段的输出将是

```
// observerA: 1// observerB: 1// observerA: 2// observerB: 2// observerA: 3// observerB: 3
```

在上面的代码片段中，我们在第一行实例化了一个新主题。然后我们通过两个订阅者两次订阅主题。在 subscription 函数中，我们使用了 next()方法来设置 subject 的下一个值，并简单地将其打印到开发人员的控制台。

在随后的几行中，我们用 from 操作符创建了一个可观察对象，它一个接一个地发出值。那么我们同意上面的观察。

在输出中，我们可以看到，对于从可观察对象发出的每个值，两个订阅者分别接收值，并按顺序打印发出的值。

******* 如果不调用 next()方法，Subject 不返回值。

这是一个非常简单的主题如何工作的概述。

考特西:[http://code2stepup.com/](http://code2stepup.com/)

萨特雅普里雅·米什拉是 http://code2stepup.com/的创始人。他是一名软件顾问、企业培训师和作家。