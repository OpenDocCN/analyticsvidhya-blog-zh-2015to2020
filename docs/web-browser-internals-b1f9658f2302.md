# 网络浏览器内部

> 原文：<https://medium.com/analytics-vidhya/web-browser-internals-b1f9658f2302?source=collection_archive---------8----------------------->

## 为什么这应该是你走向 web 开发的第一步？

如果你希望在 **web 开发**领域开始职业生涯，了解 web 浏览器的内部原理可能会成为一个很好的救命稻草。

> 浏览器是主人——等待反应，vue，节点一点

Learn React，learn Vue，experiment Ember——如果你掌握了浏览器的内部原理，你会感受到与这些酷框架的联系，以及在未来适应任何框架的能力。

但是，用浏览器应该掌握什么呢？除了数百万行的 **C++** 为它提供动力之外，肯定还有很多。

![](img/ae4ea63771d90e73ebe0a789574d41e8.png)

常用浏览器

什么是浏览器？它是如何工作的？最常用的浏览器有哪些？

有数以百万计的文章，参考资料，维基百科对这些的见解。很有可能你也已经知道这些了。因此，让我们试着找出一些非常重要但我们很难抓住的细节。

你遇到过这种类型的 CSS 代码吗？

```
-webkit-border-before: 1px;
-webkit-border-before: 2px dotted;
-webkit-border-before: medium dashed blue;
```

是的，你有没有想过它们与网络浏览器有什么关系，它们对兼容性有什么帮助。

> 几乎所有流行的浏览器都基于 [webkit](https://webkit.org/) 引擎

WebKit 基本上是一个[网络浏览器引擎](https://webkit.org/project)，供 [Safari](https://www.apple.com/in/safari/) 、Mail、App Store 以及 macOS、iOS 和 Linux 上的许多其他应用程序使用。

让我们看看什么样的引擎实际上驱动着我们喜欢使用的主浏览器，

*   Chrome 和 Opera(从 15 版开始)——[眨眼](https://www.chromium.org/blink)
*   火狐— [壁虎](https://developer.mozilla.org/en-US/docs/Mozilla/Gecko)
*   Safari — [Webkit](https://developer.apple.com/documentation/webkit)
*   边缘— [闪烁](https://www.chromium.org/blink)【网络浏览器— [三叉戟](https://en.wikipedia.org/wiki/Trident_(software))

好了，现在让我们来看看浏览器的高层结构是怎样的。

普通的 web 浏览器应该具有以下组件，

*   UI(用户界面)——这包括我们与之交互的一切，如地址栏、按钮(前、后、搜索等)
*   浏览器引擎——它作为浏览器和渲染引擎之间的接口工作(我们将在接下来介绍)
*   渲染引擎——它是解析 HTML、CSS 等的引擎。这就是负责显示请求内容的部分。
*   网络—用于 HTTP 请求
*   UI 后端——用于绘制基本的小部件，如组合框和窗口。
*   Javascript 解释器—解析并执行 javascript 代码
*   数据存储——浏览器中的所有数据功能，即 cookies、IndexedBD、localStorage 等。

有了这个背景，我再跟大家分享一件有趣的事。

> Chrome 为不同的进程(比如标签)使用不同的渲染引擎实例

这是正常情况下的流动方式，

![](img/964e61667a1b5f5ffe7f3d1f6d481076.png)

信用——html5rocks.com

这里我们可能对 DOM 很熟悉，但是流程中的其他部分呢，

*   在构建 DOM 树时，浏览器会构建另一棵树，称为渲染树。
*   它只不过是按照显示顺序排列的视觉元素。
*   它的主要目的是使内容能够按照正确的顺序绘制。
*   [Firefox](https://www.mozilla.org/en-US/) 称它们为框架，而基于 WebKit 的称它们为渲染器或渲染器对象。

哦，是的，让我分享一个与上述背景一致的重要注意事项，

非可视 DOM 元素将不会插入到呈现树中。听起来很有说服力，对吧？

非视觉的意思就是说**头**元素。

*   渲染器对应于 DOM 元素，但这种关系不是一对一的。有对应于几个可视对象的 DOM 元素。

例如:“select”元素有三个呈现器:一个用于显示区域，一个用于下拉列表框，一个用于按钮。

*   当渲染器被创建并添加到树中时，它没有位置和大小。这就是**布局**出现的地方。

## 风格计算

这是最有趣的部分之一，

*   构建渲染树需要计算每个渲染对象的视觉属性。这可能比我们实际想象的更复杂。
*   样式表的来源是浏览器的默认样式表。

如果你想了解更多，请点击此[链接](https://www.html5rocks.com/en/tutorials/internals/howbrowserswork/#Style_Computation)。

## 样式表级联顺序

你们中有多少人在调试 CSS 行为时遇到过困难？我想这可能会让人望而生畏。如果我们理解样式表级联顺序以及如何在我们的项目中正确使用它们，这绝不容易。

> 浏览器吸入不同的风格——相信我

顺序是这样的，

*   浏览器声明
*   用户普通声明
*   创作普通声明
*   创作重要声明
*   用户重要声明

等等，我们说的是哪个用户或者作者？

**作者**根据文档语言的约定为源文档指定样式表。它基本上是网站创建者(或)的代码。

**用户**可能能够指定特定文档的样式信息。在浏览器中与它互动的是我们！

这是关于 CSS 层叠的详细文档，可以帮助你更深入地了解这个问题。

去吧，接触一些不同层次的 CSS 声明，以便对此有一个清晰的认识。

那么如何处理脚本呢？那也很重要。

让我们打破这个循环，

## **脚本执行顺序**

网络的本质通常是同步的。比方说，你有一个 HTML 文档，

```
<!DOCTYPE html>
<html><head>
<script>
function myFunction() {
 document.getElementById(“demo”).innerHTML = “Paragraph changed.”;
}
</script>
</head>
<body><h1>A Web Page</h1>
<p id=”demo”>A Paragraph</p>
<button type=”button” onclick=”myFunction()”>Try it</button></body>
</html>
```

当浏览器读取它时会发生什么？

它首先开始解析 html，就像 HTML 标签，head 标签…uff，

然后，它注意到脚本标记，并开始处理脚本。只有在完成之后，它才返回到 HTML 解析。在这种情况下，它是可以接受的，只是一个 innerHTML 语句。但是，如果它必须处理数百行代码，进行少量 API 调用，等等，那该怎么办呢？这可能真的令人生畏，对不对？因此，在 HTML 文档中应该把脚本放在哪里一直存在争议。这是一个非常不同的话题。让我把它留给你。

由于现代 HTML 规范，现在您可以明确指定浏览器是应该等待还是异步处理。

看看[异步](https://javascript.info/script-async-defer)和[延迟](https://javascript.info/script-async-defer)属性。这让工作变得更容易。

现在让我们讨论浏览器如何优化布局创建过程中发生的变化。如果你深入研究这个问题，听起来会非常有趣。不要担心，我会在最后分享一个美丽的资源给你做实验:)

## 最佳化

当布局由“调整大小”或渲染器位置(而不是大小)的变化触发时，渲染大小从缓存中获取，而不会重新计算。酷吧？

想象一下，如果浏览器重新计算每一个变化，体验将会变得多么耗时和缺乏交互性。

在某些情况下，只修改一个子树，布局不是从根开始的。如果更改是局部的，并且不影响其周围环境，例如文本插入到文本字段中，就会发生这种情况。

> 浏览器试图做尽可能少的动作来响应变化。

关于浏览器，你还可以探索和学习更多的东西。我只是计划以非常简短的方式分享一些概念，这样就不会太复杂，并鼓励你深入研究。

如果这些让你着迷，是时候做更多的实验了。请务必看看这篇由我的一位导师分享给我的深入的[文章](https://www.html5rocks.com/en/tutorials/internals/howbrowserswork/)。我敢打赌你看完绝对不会后悔。

**信用**:[https://www . html 5 rocks . com/en/tutorials/internals/howsbrowserswork/](https://www.html5rocks.com/en/tutorials/internals/howbrowserswork/)

**由** [程昕婷](https://www.html5rocks.com/profiles/#taligarsiel)和[保罗爱尔兰](https://www.html5rocks.com/profiles/#paulirish)