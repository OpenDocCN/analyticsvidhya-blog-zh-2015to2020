# Python 开发人员的类型脚本

> 原文：<https://medium.com/analytics-vidhya/typescript-for-python-developers-a16e50a5acb2?source=collection_archive---------1----------------------->

![](img/017dff8365e9542aa6c363419fc8b00b.png)

照片由[菲利贝托·桑蒂兰](https://unsplash.com/@filisantillan?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/programmer?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

## 面向有经验的 Python 开发人员的 TypeScript 快速入门

如果您已经了解 Python，您可能会对学习 TypeScript 感兴趣。它越来越受欢迎。它可以在你的浏览器中工作。在许多方面，它是一种非常相似的语言。

然而，大多数学习编程语言的地方都集中在新开发人员身上。很少有文章或教程只涉及新语言的本质，而忽略了编程的基础。本文将简要而深入地概述 TypeScript 如何工作，以及如何轻松入门。它包含了基本的语法，循环，类型，简写更多。

# 基础知识

TypeScript 是 Javascript 的超集。所以任何 Javascript 代码都是有效的类型脚本代码，而不是相反。TypeScript 编译成 JavaScript 代码，然后由执行器、浏览器或 NodeJS 解释。

TypeScript 可以用作前端语言，通常作为框架或库的一部分。或者作为 node.js 的后端，通常作为 web 服务器，但理论上也作为纯后端。

# 句法

像大多数其他语言一样，TypeScript 使用括号来创建范围，而不是 Python 基于缩进的风格。`CamelCase` TypeScript 也是相对于 Python 的`snake_case`的约定。

## 基础

下面是入门的基础知识列表，首先是 Python 代码，然后是 TypeScript 代码。我将使用 Python 的类型与 TypeScripts 类型进行比较。TypeScript 的打字其实也是可选的，可以很好的推断类型。

基本语法

## 平等

JavaScript 和 TypeScript 中有两个相等运算符。`*==*`检查值是否相等。这将尝试将值进行类型转换。`===`将检查值的相等性和类型。

这种转换使得 JavaScript 有时会做一些奇怪的事情。很多 [WtfJs 项目](https://github.com/denysdovhan/wtfjs)都是关于项目的平等铸造。

## 分配

首先，TypeScript 中的赋值是用一个关键字完成的。关键词有三个:`var`、`let`、`const`。学习这个很容易:如果值是一个常数并且不会改变，使用`const`，如果值将要改变，使用`let`，不要使用`var`。不使用`var`的原因与[变量作用域](https://www.freecodecamp.org/news/var-let-and-const-whats-the-difference/)有关，此处不涉及。

由于这种作用域和显式赋值，您还需要在适当的时候定义变量。例如，如果您想让一个新变量在 if 块中是`large`，在 else 块中是`small`。然后，您必须确保在 if else 块之前定义变量。如果不这样做，编译器会抱怨，因为变量不存在于 if else 块之外。

## 循环和列表

TypeScript 有两种类型 for 循环。再来一个经典，再来一个现代。接下来，数组原型(也称为列表类型)有几个很好的内置函数，比如`forEach` *、* `map`、`filter`、 *…* 、 [Most](https://developer.mozilla.org/tr/docs/Web/JavaScript/Reference/Global_Objects/Array/prototype) (但不是`forEach`、 *)* 返回一个新数组，不修改旧数组。这对于那些试图不改变任何变量而只是制造新变量的函数型、纯函数型的人来说非常好。

## 类型

这些比较 *mypy* 类型和 *TypeScript* 类型。格式为 *Python — TypeScript* 。完整的类型列表可以在[这里](https://www.typescriptlang.org/docs/handbook/basic-types.html)找到。

*   `int, float, double, long`*—* `number`
*   `str` *—* `string`
*   `List[T]` *—* `Array<T>` 或`T[]`，`T`为任何其他类型
*   `Dict` *—* `Object`
*   `Tuple[str, int, dict]`*——*`[string, number, object]`
*   `TypedDict` *—* 自定义界面

TypeScript 和 JavaScript 只有一种没有具体引用整数或浮点数的数字类型。从某种意义上说，这是有意义的，但是将数据转换成整数很烦人。

在谈论类型时，有人不能跳过 TypeScript 中的两个不同的空值:`null` 和`undefined` *。*

*   `null`表示不存在的**对象**或没有输入的事实。可以比作 Python 的`None`。
*   当引用不存在的属性时，返回`undefined`。类似于 Python 的`KeyError`或者`AttributeError`。

如果引用对象上不存在的值，TypeScript 不会出错。然而，在 undefined 上引用任何东西都会抛出一个`TypeError`。两个都是假的，`*undefined*`我不等于`null`。

类型的 bool 转换的一个小区别是，空列表和空对象在 TypeScript 中是真的。

## 目标

TypeScript 中的对象与 Python 中的字典非常相似。

尽管有一些不同:

*   每个键也成为一个属性。`someObject.key`和`someObject['key']`之间没有真正的区别，除了 TypeScript 编译器可能对第二个版本抱怨较少。
*   您不能在创建中使用变量，因为键在赋值中被视为字符串。
*   如上所述，引用不存在的字段会返回`undefined`。

## 功能

函数的工作方式很像 Python，只是有一些小的不同。函数可以有默认值。但是，函数调用必须从左到右填充，并且不支持关键字参数。它们支持随机数量的带有分布参数的参数。

为了弥补没有`kwargs`的事实，TypeScript 实现通常提供一个`options/props/…` last 参数，它包含所有只包含关键字的参数。例如， [AWS CDK 打字稿](https://docs.aws.amazon.com/cdk/api/latest/docs/@aws-cdk_aws-events.Rule.html)有一个*道具*参数，而等价的 [Python CDK](https://docs.aws.amazon.com/cdk/api/latest/python/aws_cdk.aws_events/Rule.html) 提供了单独的关键字参数。

## 类和接口

类是标准 OOP 类的运行，而接口提供了一个结构，但没有实现。

与 Python 不同，一个类只能扩展另一个类。一个类可以实现多个其他类或接口。一个接口只能扩展另一个接口，不能实现另一个接口。

这也显示了 TypeScript 中的一个很好的简写。构造函数可以定义`public name: type`，编译器会理解它必须在初始化中做`this.name = name`。

## 例外

异常在 TypeScript 中的行为与在 Python 中的行为非常相似。关键词略有不同，但大多数事情都延续了下来。然而，TypeScript 不允许像 Python 一样同时捕捉多种错误类型。

# 人手不足

现代 JavaScript 和 TypeScript 的一些优点是简写。它们有助于加快你的开发速度，尽管它们比纯循环实现要慢一些。

该列表包括 Python 中的基本布尔推理，或缺省的短路。但是 TypeScript 有更多这样的东西。

[这篇文章](https://www.sitepoint.com/shorthand-javascript-techniques/)提供了一个关于人手不足的更详细的概述，但是上面的文章涵盖了最重要的内容。

# 生态系统

和 Python 的 pip 一样，TypeScript(和 JavaScript)有一个名为[节点包管理器(npm)](https://www.npmjs.com/) 的包管理器。它将所有的包安装在本地的一个文件夹 *node_modules* 中，这个文件夹因为变得非常大而臭名昭著。

这也是一个普遍的想法，即在 npm 中确实有一个包来处理最琐碎的事情。就我个人而言，我觉得 pip 包质量更高，维护更好，但这只是我的看法。但是 PyPi 包可能会少一些。

主要的重要库有:

*   HTTP 调用(类似请求):axios、rxjs/ajax、request(已弃用)
*   前端:反应 vs 角度 vs Vue 等等
*   后端:Node.js，express(路由)
*   时间和日期:moment.js
*   无功数据:RxJs。

关于 RxJs，因为这是我用过的最酷的库。它可以帮助你处理流数据。它在角度上是积分的，几乎所有的数据都是可观测的。这是开始学习函数式编程的好方法。

还有一个 [RxPy 库](https://github.com/ReactiveX/RxPY)，如果你想在 Python 中这样工作的话。

# 结论

## 好人

*   TypeScript 编译器和类型检查器感觉比 mypy 好很多。对于 VS 代码，它似乎也有更好的兼容性
*   空值检查非常有用，尤其是在处理用户数据时。
*   人手不足可以节省很多时间
*   它基本上是浏览器唯一接受的语言，也可以在服务器上使用。这允许完整的 JS/TS 堆栈
*   我从 RxJS 中获得了很多乐趣，这迫使我以一种功能性的方式进行思考
*   它是一种非常灵活的语言(像 Python 一样)，所以你总能找到适合自己的风格
*   许多图书馆和一个巨大的社区

## 坏事

*   处理日期真的很麻烦，尤其是来自 datetime 和 dateutil 的数据
*   运行脚本稍微有点烦人，因为你不能只对 *python script.py*
*   有些库不如 Python 中的一些库质量高

## 丑陋的

*   [还是 JavaScript](https://github.com/denysdovhan/wtfjs)
*   [npm_modules](https://i.redd.it/tfugj4n3l6ez.png)
*   错误处理通常非常混乱，需要深入到堆栈跟踪中才能找到任何有趣的东西
*   兼容性是令人讨厌的，因为你不能控制环境。是的，仍然有人在使用 IE7。你是否想让你的网站在上面运行，这是你的决定

## 与众不同

*   如前所述，风格不同。它比 OOP 更倾向于函数式编程。需要一些时间来适应
*   人们实际上为任何琐碎的功能安装了一个库

显然还有很多我没有提到的。但是作为有经验的开发人员，您知道答案是一个 Google 搜索或 StackExchange 解决方案。

如果你有任何问题，给我发信息，或者在评论中提问。感谢您阅读我的第一篇技术媒体文章！