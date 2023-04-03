# Javascript 是什么？一个非常简单的解释。

> 原文：<https://medium.com/analytics-vidhya/what-is-javascript-a-really-brief-explanation-3cd111dd87d9?source=collection_archive---------25----------------------->

![](img/bb6e03475b0363d1619bee9ca90c5acd.png)

塞尔吉·维拉德索在 [Unsplash](https://unsplash.com/s/photos/explosion?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

1.  Javascript 是什么？
2.  脚本语言到底是什么？
3.  JavaScript 是如何工作的？
4.  当今世界用什么 Javascript？
5.  什么是香草 JavaScript？
6.  Javascript 中为什么需要框架和库？
7.  什么是 EcmaScript？

# Javascript 是什么？

JavaScript 是一种脚本(或编程)语言，可以让你在网页上实现复杂的功能，“让网页活起来”。网页基本上由三部分组成:HTML、CSS 和 Javascript。

HTML 是网页的骨架。骨架上的 CSS 皮肤决定了骨架的外观。所以我们可以说 HTML 是网站的结构，CSS 是这个结构的风格。Javascript 能对这种结构和风格做些什么？它可以:

*   更新结构中的内容。
*   更改颜色和字体大小。
*   动画图像。

# 脚本语言到底是什么？

Javascript 有时被称为脚本语言。脚本语言是一种静态编程语言。但是一个用 C 编程语言写的程序需要编译后才能运行，而 javascript 是不需要编译的。Javascript 代码包含一系列命令，这些命令在运行时被逐个解释(而不是在编译后运行)。

# JavaScript 是如何工作的？

JavaScript 要么嵌入到网页中，要么包含在. js 文件中。Javascript 在页面上加载 HTML 和 CSS 之后工作更有意义，所以它被放在 HTML 中 body 标签的结尾。基本 Html 模板:

```
<html>
    <head>
       <style>/*I am Style of The Website*/</style>
    </head>
    <body>
       <!--I am body of Website. Everything is inside me--> </body>
</html>
```

**带 Javascript:**

```
<html>
    <head>
       <style>/*I am Style of The Website*/</style>
    </head>
    <body>
       <!--I am body of Website. Everything is inside me--> **<!--After Everything-->**
       **<script type="text/javascript">
           //I am Javascript. I make website funny place.
       </script>** **<!-- OR With External .js File-->
       <script type="text/javascript" src="./myFile/myScript.js">
       </script>**
    </body>
</html>
```

# 当今世界用什么 Javascript？

尽管 javascript 是为了使网站具有交互性而创建的，但它目前被用于:

*   开发移动应用程序(React Native、Native Script、Ionic)
*   开发基于浏览器的游戏
*   后端 web 开发(Nodejs)

# 什么是香草 JavaScript？

普通 javascript 是没有任何“简化 Javascript”工具的 Javascript。

# Javascript 中为什么需要框架和库？

框架和库都是别人写的代码，帮助你以一种简单的方式完成一些常见的任务。图书馆把 javascript 变成乐高积木给你。框架让你填满乐高房子的内部。无论如何，目标是让你的工作更容易。一些流行的框架/库有:

*   [做出反应](reactjs.org/)
*   [Vue](https://vuejs.org/)
*   [Jquery](https://api.jquery.com/)
*   [角度](https://angular.io/)

Javascript 外部工具不是必需的，但是很有用。为什么人们使用 Javascript 框架或库:

*   加速开发过程
*   使项目更有条理
*   提高性能，尤其是对于大型项目

# 什么是 ECMAScript？

JavaScript 是 ECMAScript 的子集。ECMAScript 是 JavaScript 的核心。ECMAScript 是 JavaScript 和 JScript 等脚本语言的标准。虽然 JavaScript 旨在与 ECMAScript 兼容，但它还提供了 ECMA 规范中没有描述的附加特性。JavaScript 被认为是 ECMAScript 最流行的实现之一。

看看我的另一篇文章:

[](/@ogzkgnlmz2/how-to-make-money-with-coding-what-was-my-grave-mistake-d3dde31fd408) [## 怎么用编码赚钱？我犯了什么严重的错误？

### 即使你不是专业人士，我也分享我自己的 5 个从编码中赚钱的最佳选择的经验。

medium.com](/@ogzkgnlmz2/how-to-make-money-with-coding-what-was-my-grave-mistake-d3dde31fd408)