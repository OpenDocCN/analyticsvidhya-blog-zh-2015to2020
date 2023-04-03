# 是什么让萨斯如此有趣？

> 原文：<https://medium.com/analytics-vidhya/what-makes-sass-so-interesting-d8928fc0f25e?source=collection_archive---------21----------------------->

![](img/00aa91f73e8e08b98e75ba3f39407465.png)

我们有很多 CSS 预处理器，包括 LESS，STYL，当然还有 SASS，但是是什么让 SASS 如此有趣呢？为什么人们更喜欢使用 SASS？

答案就在 SASS 属性和人们已经熟悉的 CSS 基础之间。这里是我们在 SASS 中的概念的基本纲要。

**萨斯变量:**

与编程语言类似，SASS 提供了在变量中存储通用 CSS 属性的能力。最常见的用例是将十六进制颜色值存储在一个变量中，以便进一步使用。

```
// DEFINE
$color-primary: #f908h7;//USE
.div{
background-color: $color-primary;
}
```

**预定义的变亮/变暗功能:**

当悬停在任何元素上时，我们总是希望将颜色变暗或变亮一定程度，以获得良好的用户体验。SASS 为我们提供了一种方法，通过使用一个预定义的函数，并避免对每个阴影使用十六进制值。

```
background-color: darken($color-primary, 10%); // will darken it by 10% and similarly for lighten
```

**Mixins:**

最受欢迎和喜爱的萨斯社区。SASS Mixins 就像一组自定义的 CSS 属性行，以实现可重用性。例如，那些熟悉传统 CSS 编写方式的人，他们知道为了避免代码中的不确定性而写很多次“clearfix”是多么痛苦。但是如果有一个黑客可以把它写在一个地方，并在我们需要的时候调用这个函数呢？所以，萨斯 Mixins 都是关于最后一个坑的！这就像定义一个自定义的 CSS“函数”,当我们需要这段代码时就调用它。

```
// Define a mixin called clearfix@mixin clearfix {
 &::after {
 content: “”;
 clear: both;
 display: table;
 }
}//Then call it at bunch of place to use it
nav {
 @include clearfix;
}
```

**SASS 中的功能:**

非常简单，很少使用。它确实有助于在函数中明确定义数学函数。但是问题是，我们可以在 SASS 文件中的任何地方编写数学运算。

```
//we can define custom functions called divide in this case by passing any arguments such as a,b@function divide($a,$b){
 @return $a/$b;
}//and later use the function
nav{
 margin: divide(60,2) * 1px; //to convert the unit to px we have to           multiply by 1px.
}
```

**在 SASS 中扩展:**

最真诚的是，我们可以用一大堆属性编写一个占位符，并让其他元素“扩展”这个占位符。哦，但是等等！！我们在 Mixins 不就是这么做的吗？？是的，它非常类似于 Mixins，因此在开发人员中造成了混乱。Extends 的唯一一个小问题是，我们应该只在元素实际上继承了父元素的情况下使用 extend，比如“button”和“button: hover ”,并且不应该完全不同，在这种情况下，只需使用 Mixins！

```
%btn-placeholder {
 //write styles here common to parent-button
 padding: 10px;
 margin: 8px;
}btn-placeholder:link{
 @extend %btn-placeholder;
  //define styles only for custom link button here
}
```

这基本上总结了与 SASS 相关的所有重要概念。

*下次见:)*