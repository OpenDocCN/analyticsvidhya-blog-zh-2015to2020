# 在 JavaScript 中填充和过滤数组

> 原文：<https://medium.com/analytics-vidhya/fill-and-filter-in-array-in-javascript-3e7d23814000?source=collection_archive---------15----------------------->

数组方法:

![](img/a2d6c0902fd586157c3700347cf73ac6.png)

照片由[塞尔吉奥·罗德里格斯-波图格斯·德尔奥尔莫](https://unsplash.com/@srpo?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

对于数组方法这个词，我指的是内置的数组函数，它可能在很多方面对我们有帮助。那么，为什么不去探索和利用它们，来提高我们的生产力呢？

让我们一起用一些惊人的例子来一一看看。

## **Array.fill():**

> `fill()`方法将数组中的所有元素改为静态值，从起始索引(默认为`0`)到结束索引(默认为`array.length`)。它返回修改后的数组。

简单地说，它会用你传入的参数集填充数组的元素。通常我们会传递三个参数，每个参数都有一定的含义。第一个参数值:要填充的值，第二个值:索引的起始范围(包括)，第三个值:索引的结束范围(不包括)。假设你要在某个日期应用这个方法，那么它看起来会像这样:array.fill('某个日期'，开始日期，结束日期)。

```
NOTE: Start range is inclusive and end range is exclusive.
```

让我们在下面的例子中理解这一点-

```
//declare array
var testArray = [2,4,6,8,10,12,14];console.log(testArray.fill("A"));
```

当你运行这段代码时，你会看到所有的`testArray`元素都被`'A'`取代，就像`[“A”,"A","A","A","A","A","A"]`一样。

让我们再看一个例子，包括它的范围参数，这样我们可以更清楚地理解它的定义。

```
var testArray = [2,4,6,8,10,12,14,16];console.log(testArray.fill("A",2));
```

在您运行这段代码之前，请记住在开始时我已经提到过您，`fill()`方法的第二个参数代表 index 的开始范围(包含)。从技术上来说，我在这里传递一个参数 2，这意味着，索引 0，1，2 和 2(索引值)应该是包含的，因为它是索引的开始范围，因此元素从索引 2 开始，直到数组的结尾将被替换为`testArray`中的`"A”`。这就是为什么，如果你运行这段代码，你会在屏幕上看到类似`[2,4,"A","A","A","A","A","A"]`的输出。

这里我们只是以一个开始范围索引的例子结束，但是它的结束范围索引呢？让我们用另一个`fill()`的例子来找出这一点，看看它到底会如何表现。

```
var testArray = [2,4,6,8,10,12,14,16];console.log(testArray.fill("A",2,5));
```

在这段代码中，2 是起始范围索引，5 是结束范围索引。意味着在您将看到的输出中，所有元素从索引值 2 开始，直到索引值 4 将被替换为`"A”`，因为开始范围是包含的，结束范围是不包含的。因此，如果你运行这个，你会在屏幕上看到类似于`[2,4,"A","A","A",12,14,16]`的输出。

## Array.filter()。

> `**filter()**`方法**创建一个新数组**，其中所有通过测试的元素都由提供的函数实现。

简单地说，它只是希望你传递一个回调，过滤你的输入，并把它保存在一个新的数组中。让我们看一个例子。

```
const myNumbers = [11,22,33,44,55,66,77];const result = myNumber.filter((num) => num != 55);console.log(result);
```

你可以在这里看到我从`myNumbers`数组中过滤出数字 55，并将它存储到 const `result`中，这将是它的一个新数组。如果你运行这个，你可以在屏幕上看到`[11,22,33,44,66,77]`。

我们可以再看一个例子，以便更清楚地了解`array.filter()`方法。

```
const words = ['spray', 'limit', 'elite', 'exuberant', 'destruction', 'present'];const result = words.filter(word => word.length > 6);console.log(result);
```

这里我们从长度大于 6 的数组`words`中过滤单词。因此，如果您运行这段代码，您可以在输出屏幕上看到`[“exuberant",destruction","present"]`。

这就是我这边的伙计们，我想现在你们已经可以使用数组中的 fill()和 filter()方法了。如果你喜欢它，那么请随时点击拍手，跟随按钮和反馈是最受欢迎的。

*谢谢大家，让我们补上新的。*