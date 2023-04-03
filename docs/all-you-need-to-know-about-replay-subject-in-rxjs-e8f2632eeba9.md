# 关于 RxJS 中的重放主题，您需要知道的是

> 原文：<https://medium.com/analytics-vidhya/all-you-need-to-know-about-replay-subject-in-rxjs-e8f2632eeba9?source=collection_archive---------17----------------------->

![](img/997091274d1f0782634138f82f9ac95e.png)

ReplaySubject 是 Subject 的一个特殊变体，它向观察者发出旧值，即使在执行这些值的时候观察者还没有被创建。它提供了旧值数量的设置，一旦新订户向其注册，它将记住这些旧值的数量。

下面的代码片段简单演示了 replay subject。

```
import { ReplaySubject } from 'rxjs' ;const rs$ = new ReplaySubject( 2 ); // (1)// subscriber Ars$.subscribe((data) => { // (2)console.log( 'subscriber A: ' + data);});rs$.next(Math.random()); // (3)rs$.next(Math.random());rs$.next(Math.random());// Subscriber Brs$.subscribe((data) => { // (4)console.log( 'subscriber B: ' + data);});rs$.next(Math.random());
```

在步骤 1 中，我们实例化一个新的 ReplaySubject，并将内存值的数量设置为 2。这意味着它将在内存中存储至少 2 个值，因此当它被订阅时，它至少有 2 个值要发出。在步骤 2 中，我们已经订阅了在步骤 1 中创建的 replay 主题，这里我们只是将值打印到控制台。在步骤 3 中，我们使用 next()方法向 replay subject 传递新值。我们将这个步骤做 3 次，以检查我们是否真的可以在内存中存储至少 2 个值。在下一个步骤 4 中，我们创建了另一个名为 subscriberB 的订户，就像前面的订户一样，它将值打印到控制台。在最后一步中，我们再次调用 next()方法，并将值设置为某个随机数。

这里的关键观察是，在 subscriberB 出现之前，replay subject 已经发出了三个值，并且通过我们的配置，我们已经在内存中保存了两个过去的值。现在来看看输出。

```
// subscriber A: 0.8107208546492104 // (1)// subscriber A: 0.04243867985866512 // (2)// subscriber A: 0.5350443486512133 // (3)// subscriber B: 0.04243867985866512 // (4)// subscriber B: 0.5350443486512133 // (5)// subscriber A: 0.07003469196276346 // (6)// subscriber B: 0.07003469196276346 // (7)
```

在这里的输出中，我们有 7 个值，查看输出值，我们可以得出以下推论。

*   值 1、2 和 3 来自我们在代码片段的步骤 3 中实现的 next()方法。此时，subscriberB 不存在。因此没有从 subscriberB 中打印出任何值。
*   现在创建了 subscriberB，按照配置，内存中有两个过去的值，订户打印出来。
*   这里要注意的一点是，值 4 和 2 以及值 5 和 3 是相同的，因为它们是重放主题的内存值。
*   接下来的值 6 和 7 来自代码片段的最后一条语句。

上面提到的几点是你开始使用 ReplaySubject 时需要了解的所有基础知识。