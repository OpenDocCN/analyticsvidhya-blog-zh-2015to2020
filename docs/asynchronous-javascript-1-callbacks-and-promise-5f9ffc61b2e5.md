# 异步 Javascript (1)回调和承诺

> 原文：<https://medium.com/analytics-vidhya/asynchronous-javascript-1-callbacks-and-promise-5f9ffc61b2e5?source=collection_archive---------10----------------------->

![](img/145994b3ea31518c8a1238056ebdc663.png)

# **1。回调**

首先，我们必须理解 Javascript 中的一些关键概念——同步/异步和阻塞/非阻塞

考虑以下片段

```
//simple callbacks exampleconst greetings = () => {
 console.log('Hello');
};const greetingsLater = () => {
 console.log('Hi');
}setTimeout(greetingsLater, 5000);
greetings();
```

像大多数编程语言一样，Javascript 从上到下一行一行地执行代码——在上面的代码片段中，它创建了两个函数——“greetings”和“greetingsLater”，然后通过内置函数“setTimeout”执行函数“greetingsLater”，最后执行 greetings

其他内置的异步函数包括:setInterval()、requestAnimationFrame()、navigator . geolocation . getcurrentposition…

它将输出

```
Hello
Hi //after 5 seconds later
```

代替

```
//after 5 seconds later
Hi
Hello
```

为什么？

因为 setTimeout()是一个内置 Javascript 函数，它有一个特殊的机制将操作交给浏览器。每当 Javascript 看到这种异步函数时，它会切换到浏览器并进入下一行，而不是等待计时器结束— **非阻塞**。
浏览器接管它并监控定时器，当定时器结束时，浏览器将任务“greetingsLater”放入浏览器中的“待办事项”——**消息队列，一旦 Javascript 引擎——**栈**空闲，Javascript 将执行函数“greetings la ter”——**异步****

```
//blocking exampleconst greetings = () => {
 console.log(‘Hello’);
};
const greetingsLater = () => {
 console.log(‘Hi’);
};
//a for loop which takes time
for (i=1; i<999999999; i++) {
}
greetingsLater();
greetings();
```

阻塞

上面的代码片段是**阻塞**的例子——函数“greetingsLater”和“greetings”在 for 循环中只能执行一次。通常，我们不希望它出现在 Javascript 中，因为它会冻结应用程序，同时我们什么也做不了，这会导致糟糕的用户体验。

回调总是稍后执行

```
//simple callbacks example 2const greetings = () => {
 console.log('Hello');
};const greetingsLater = () => {
 console.log('Hi');
}setTimeout(greetingsLater, 0);
greetings();
```

上面的代码片段与第一个代码片段相同，只是我们在“setTimeout”中放了零，这意味着以 0 毫秒的延迟执行。尽管如此，因为这是一个回调，所以仍然需要等待，如果 Javascript engine-stack 得到了其他任务(在大多数情况下是指不是回调的任务)，那么它将首先执行其他任务。因此，在这种情况下，它仍然先执行“greetings()”，然后执行“greetingsLater”

```
Hello
Hi //immediately display after Hello, as provided 0 in setTimeout
```

如果我们有多个异步任务，会发生什么？

我们可以通过使用嵌套回调来控制它们的顺序，但是很难读取，不是吗？

```
function gettingLocation() {
  setTimeout(function() {
    console.log('Some task required before retrieving location');
      setTimeout(
        navigator.geolocation.getCurrentPosition (
            posData => {
                console.log(posData);
                setTimeout(() => { 
                    setTimeout(console.log('Some task after getting the location'), 0) 
                    }, 0);},
            error => {
                console.log(Error);
                setTimeout(() => {
                    setTimeout(console.log('Some task after fail getting the location'), 0) 
                }, 0);}
       ), 0);
  }, 0);
};
```

这就是为什么 Javascript 引入了 Promise 和 async await

# **2。承诺**

Promise 是一种语法，允许我们以同步方式执行异步代码。简而言之，我们可以控制异步代码的流程，尤其是那些相互依赖的任务。只有单一的嵌套层，没有更多的回调地狱

大多数现代 Javascript API 都支持 promise，但是有些函数如 setTimeout()和 navigator . geolocation . getcurrentposition 不支持 promise，我们可以手动或使用一些库如 [bluebird、](http://bluebirdjs.com/docs/api/promise.promisify.html)将它“包装”成 Promise，这个过程称为 **Promisfy**

## **许诺**

通常，我们会这样设置超时。假设我们想在 3 秒钟后打印一些东西

```
setTimeout(() => console.log('3 seconds Done!'), 3000)
```

现在，我们将创建一个函数来“包装”setTimeout，以便返回 Promise 对象。第一个‘resolve’和‘reject’是我们要传递给 new Promise()的参数，你可以随意命名。然后我们将使用它们来包装成功或失败的结果。

```
function delay(ms) {
   return new Promise((resolve, reject) => setTimeout(
() => (resolve(console.log('3 seconds Done!')), () => reject(console.log('fail')) ), ms));
}
```

通常，我们会在“then”语句中使用结果，而对于 setTimeout，我们没有任何失败。所以，它会是这样的

```
function delay(msg, ms) {
   return new Promise(resolve => setTimeout(
() => resolve(msg), ms));
}//in arrow function
const delay = (msg, ms) => new Promise(resolve => setTimeout(
() => resolve(msg), ms));
```

你可能会觉得奇怪，为什么我们要把一个函数放在 argument 里，然后马上被 JavaScript 执行。这就是所谓“揭示构造器模式”。如果你熟悉 Java，你可能会认为它是一个私有函数，但是因为 JavaScript 没有“私有/公共”的概念，所以他们这样设计是为了限制访问。
有关更多信息，请参考以下链接

1.  [https://www . I-programmer . info/programming/JavaScript/11379-JavaScript-async-promises-the-revealing-constructor-pattern . html？start=1](https://www.i-programmer.info/programming/javascript/11379-javascript-async-promises-the-revealing-constructor-pattern.html?start=1)
2.  [https://stack overflow . com/questions/37651780/why-the-promise-constructor-need-a-executor](https://stackoverflow.com/questions/37651780/why-does-the-promise-constructor-need-an-executor)

我们现在有了返回 promise 对象的函数，我们如何使用它。只需在“then”语句中声明一个以“data”为参数的函数。“数据”是承诺“解决”的结果(对于“拒绝”的返回，我们将在下一节讨论)，即“3 秒钟完成！”或者上面的“失败”

```
delay(3000).then(data => console.log('data:', data));
```

Promisfy 的另一个例子—navigator . geolocation . getcurrentposition()

与上面的例子相似。基于它的 [API](https://developer.mozilla.org/en-US/docs/Web/API/Geolocation/getCurrentPosition) ，我们必须传递“成功”和“错误”的回调，以及一个用于选项的变量。所以我们只是将 resolve 和 reject 分别放在“成功”和“错误”回调中

```
const geoLocation = (opts) => new Promise((resolve, reject) => navigator.geolocation.getCurrentPosition(
posData=>{resolve(posData)}, error=>{reject(error)}, opts));
```

这次我们可以测试“拒绝”,因为我们可以阻止浏览器的位置访问来触发这种情况。为了得到错误，只需在消费上述承诺时，在“then”语句中传递第二个参数

```
geoLocation().then(
data => console.log('data', data), 
err => console.log(err)
);
```

您也可以使用“catch”语句

```
geoLocation()
.then(data => console.log('data', data))
.catch(err => console.log(err));
```

**承诺链** 参考第 1 节中的 callhell 示例，我们可以通过 more then 和 catch 将这段代码转换为承诺链。除了使用上一次的结果——“解决”或“拒绝”，它还可以通过返回另一个承诺来处理另一个异步任务。

```
//Promisfied setTimeout
const delay = (msg, ms) => new Promise(resolve => setTimeout(
() => resolve(msg), ms));//Promisfied navigator.geolocation.getCurrentPosition
const geoLocation = (opts) => new Promise((resolve, reject) => navigator.geolocation.getCurrentPosition(
posData=>{resolve(posData)}, error=>{reject(error)}, opts));delay('Some task required before retrieving location', 0)
.then(data => console.log(data))
.then(() => geoLocation())
.then(data => {console.log(data); return delay('Some task after getting the location', 0)})
.catch(err => {console.log(err); return delay('Some task after fail getting the location', 0)})
.then(data => console.log(data));
```

如果成功检索位置，它打印

```
Some task required before retrieving location
GeolocationPosition {coords: GeolocationCoordinates, timestamp: 1607081342802}
Some task after getting the location
```

如果检索位置失败，它将打印

```
Some task required before retrieving location
GeolocationPositionError {code: 1, message: "User denied Geolocation"}
Some task after fail getting the location
```

关于“catch”语句的更多信息

“catch”语句可以放在第一行之后，它将捕捉“catch”语句之前的任何错误。
考虑下面的代码片段

这里的“catch”将捕获来自 geoLocation()的错误，并恢复“then”语句

```
//assume fail to get location
geoLocation()
.catch(err => console.log(err))
.then(data => console.log('data', data));//Output
//GeolocationPositionError {code: 1, message: "User denied Geolocation"}
//data undefined
```

在下面的例子中，因为第一个“then”语句有错误，所以它跳过其他“then”语句，直到“catch”语句，然后继续。

```
//assume fail to get location
delay("msg", 1000)
.then(data => console.log('data', data))
.then(()=>geoLocation())
.then(data => console.log('data', data))
.then(()=> {return delay("msg2", 1000)})
.then(data => console.log('data', data))
.catch(err => console.log(err))
.then(()=> {return delay("msg3", 1000)})
.then(data => console.log('data', data));//Output
//data msg
//GeolocationPositionError {code: 1, message: "User denied Geolocation"}
//data msg3
```

* *在“then”语句中的第二个参数— reject，与上面提到的“catch”语句具有相同的功能。就我个人而言，我更喜欢“catch”语句，因为它提供了更多的灵活性。

尝试下面的工作示例，查看控制台中的输出