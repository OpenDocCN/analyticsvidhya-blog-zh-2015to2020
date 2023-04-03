# 使用“fetch()”获取 API

> 原文：<https://medium.com/analytics-vidhya/fetching-an-api-with-fetch-e8d902811afa?source=collection_archive---------14----------------------->

![](img/7d0ecd6adc295da865fc22cbc3098be3.png)

伊利亚·巴甫洛夫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

API 或应用程序接口可以是编程世界中的许多东西。本质上，它们是一种可以从客户机传递到服务器的数据形式，反之亦然。这些数据可以是任何东西，从定义的函数到布尔值等等。它们之所以如此重要，是因为它们是你今天访问的许多网站的组成部分。在谈论 API 时，“消费”这个词经常被传来传去。这是获取 API 并对其进行解码以实现其目的的过程，因为否则它将毫无用处。当您阅读本文时，请记住 API 只是数据的一种形式，仅此而已。以最简单的形式，它们可以与您的第一个 JavaScript 应用程序相提并论，或者像您希望的那样复杂。这完全取决于使用它的目的，以及它是否达到了目的。

## JSON

API 的一种流行形式是以“JSON”格式或 JavaScript 对象符号组成的。这是一种当今许多 API 都遵循的格式，因为它非常容易阅读。让我们看看它是什么样子的。

```
{
  "puppies": ["golden retriever", "beagle", "pug"]
}
```

JSON 基本上只是一个 JavaScript 对象，它的键值对有一些关键的不同。第一个也是最引人注目的是“钥匙”，在这种情况下是“小狗”。它有括号，而在普通的 JavaScript 对象中,“key”上没有括号。这是因为编译后的 JSON 是一个巨型字符串。第二个区别是包含对象没有变量名。这是因为当我们消费它时，我们将它存储在一个变量中。

想想看，JSON 真的很酷，因为字符串是最简单的数据形式之一，这使得它成为一种非常有效的数据传输形式。JSON 数据被消费后，它采用常规 JavaScript 对象的形式，允许开发人员对其进行操作。很聪明吧？

## 强烈的

现在使用 JSON 数据非常简单，但过去要复杂得多。今天，我们使用一种基于承诺的方法，称为“获取”。如果你不知道承诺是什么，它基本上只是一个函数，当它解析为真时抛出一个值，当为假时抛出另一个值。我们来看看“取”吧。

```
fetch("API URL or the directory location")
```

Fetch 将 API 的位置作为参数，并对其进行处理。然而，要使用它的数据，我们需要“处理”它。承诺需要处理，因为它们依赖于它。一旦“fetch”被激发，它将吐出一个值。为了消耗这个值，使用了“then”方法。

```
fetch("API URL or the directory location").then(data => console.log(data))
```

然后，当承诺解析为真时，就向其传递数据。“data”变量是接收到的未处理的 JSON 数据。在处理这些数据之前，我们必须履行另一端的承诺。这可以通过“catch”方法来实现。“catch”方法可以链接到“then”方法上。当承诺被证明为假时，触发“catch”。使用这种方法很重要，因为在使用 API 时可能会出现错误，因为服务器并不总是可靠的。这允许开发人员通知客户端他们的错误。错误处理是另一天的另一个主题。

```
fetch("API URL or the directory location").then(data => console.log(data)).catch(error => console.log(error))
```

## 处理 JSON

为了使用 JSON 数据，我们需要将其转换成 JavaScript 对象。这就是我们使用“json”方法的时候。这个方法附加到 JSON 对象上，将它转换成可以使用的常规 JavaScript 对象。

```
fetch("API URL or the directory location")
.then(data => {
  data.json()
})
.catch(error => console.log(error))
```

但是你猜怎么着？“json”方法是另一个可以解决或拒绝的承诺。如果数据是有效的 JSON 数据，它将解析为 true，否则为 false。对于要使用的数据，另一个“then”方法被链接到“json”方法上。

```
fetch("API URL or the directory location")
.then(data => {
  data.json()
.then(data => {
sendTheData(data)
})
})
.catch(error => console.log(error))
```

就是这样！一旦数据被处理成一个常规的 JavaScript 对象，开发人员就能够操作它。在本例中，我使用了一个名为“sendTheData”的函数，并将“Data”作为参数，这样我就可以在我的脚本中使用它。一旦承诺被灌输到你的头脑中，使用 fetch 来消费你的 API 将变得非常容易！

希望这有助于您更好地理解 API！