# 正则表达式:什么和为什么？

> 原文：<https://medium.com/analytics-vidhya/how-to-write-a-regular-expression-ebc89d0b0258?source=collection_archive---------14----------------------->

![](img/e5d9c392c4d0dd1873a14e51418d7419.png)

图片来源:[马库斯·斯皮斯克](https://unsplash.com/@markusspiske?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

> 正则表达式是一种强大的搜索和替换技术，你可能在不知情的情况下使用过。无论是文本编辑器的“查找和替换”功能，还是使用第三方 npm 模块验证 http 请求体，或者是终端基于某种模式返回文件列表的能力，它们都以某种方式使用正则表达式。这不是一个程序员必须明确学习的概念，但是通过了解它，你可以在某些情况下降低代码的复杂性。
> 
> *在本教程中，我们将学习* ***javascript 中正则表达式的关键概念和一些用例。***

# 如何编写正则表达式？

用 Javascript 编写正则表达式有两种方法。一种是通过创建一个**文字**，另一种是使用 **RegExp** 构造函数。

虽然在对特定字符串进行测试时，两种类型的表达式将返回相同的输出，但是使用`RegExp`构造函数的好处是它在运行时被评估，因此允许对动态正则表达式使用 javascript 变量。此外，正如在这个[基准测试](https://www.measurethat.net/Benchmarks/Show/1734/1/regexp-constructor-vs-literal#latest_results_block)中所看到的，在模式匹配中`RegExp`构造函数比字面正则表达式表现得更好。

任何一种表达式的语法都由两部分组成:

*   **模式**:字符串中需要匹配的模式。
*   **标志**:这些是修饰符，是描述如何执行模式匹配的规则。

# **方法**

下面描述了正则表达式常用的一些方法:

**测试**:对象`RegExp`的`test`方法允许你测试你的字符串或者它的一部分是否匹配正则表达式。

这里的`i`标志确保模式匹配不区分大小写。所以在使用这个标志时,“cat”和“Cat”被认为是相同的。

**exec :** 另一个常用的方法是`exec`方法，它带有一个`g`标志，允许我们循环遍历我们的字符串并获得关于每个匹配的详细信息。

在上面的代码中，使用了`i`和`g`标志的组合，使其不区分大小写并允许多个匹配。正则表达式的`lastIndex`属性是字符串中在前一次匹配后进行匹配的位置。

**match :** 最后，我们还有 String.prototype 的`match`方法，当与`g`标志一起使用时，它返回特定正则表达式模式的所有匹配的数组。

> 注意:如果没有使用 `*g*` *标志*，match r *只返回第一个匹配*

# 旗帜

如上所示的标志定义了模式匹配的行为。下面介绍了六种最常用的标志:

*   ***i*** :模式匹配不区分大小写
*   ***g*** :如果使用此标志，模式匹配将返回特定模式的所有匹配，否则将只返回第一个匹配。
*   ***m*** :多行模式匹配使用此标志
*   ***s*** :点匹配一个换行符\n。
*   ***u*** :匹配全 unicode
*   ***y*** :从上次匹配结束的地方继续匹配

通过简单地将它们一个接一个地连接起来，所有这些标志可以彼此结合使用。

# 速记班

Javascript 附带了各种速记字符类，为编写正则表达式提供了一种便捷的方式:

*   **\s** (匹配空格)
*   **\S** (匹配非空白字符)
*   **\d** (匹配[0–9])
*   **\D** (匹配非数字字符)
*   **\w** (匹配所有 ascii 字符)
*   **\W** (匹配非文字字符)

# 常见使用案例

*   *仅匹配字母*

*   *仅匹配数字*

*   *匹配 beginning(^)*

*   *比赛结束($)*

*   *匹配一个或多个(+)*

*   *匹配零个或多个字符(*)*

*   *匹配一定长度({minlength，maxlength})*

*   *零和一之间的匹配(？)*

> 注意，当使用 RegExp 构造函数时，速记类的反斜杠必须用\\而不是\进行转义。因此，最后一个用例中的正则表达式与下面给出的文字正则表达式相同:
> 
> `const reg=/\w+\.{1}docx?$/g`

# **结论**

本教程只是解释正则表达式及其功能的冰山一角。我有意保留了分组、懒惰(。*？)vs 贪心(。*)匹配和前瞻出本教程，确保内容不至于铺天盖地。任何想深入了解这些概念的人都可以通过下面给出的链接:

# 有用的链接:

*   [使用正则表达式与新的 JS 正则表达式构造函数的基准测试结果](https://www.measurethat.net/Benchmarks/Show/1734/1/regexp-constructor-vs-literal#latest_results_block)
*   [在线正则测试仪](https://regex101.com/)
*   [Debuggex](https://www.debuggex.com/)
*   [密码强度验证](https://www.thepolyglotdeveloper.com/2015/05/use-regex-to-test-password-strength-in-javascript/)
*   [Sampson 关于“在正则表达式的上下文中‘懒惰’和‘贪婪’是什么意思？”](https://stackoverflow.com/questions/2301285/what-do-lazy-and-greedy-mean-in-the-context-of-regular-expressions)
*   [正则表达式信息](https://www.regular-expressions.info/tutorial.html)
*   [正则表达式备忘单](https://www.rexegg.com/regex-quickstart.html)