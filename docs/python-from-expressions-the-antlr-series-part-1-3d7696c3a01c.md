# 来自表达式的 python—ANTLR 系列(第 1 部分)

> 原文：<https://medium.com/analytics-vidhya/python-from-expressions-the-antlr-series-part-1-3d7696c3a01c?source=collection_archive---------7----------------------->

所以，在最后一个故事中，我对 ANTLR(或它的树，双关)进行了一次小小的了解，并概述了它是什么以及它真正能做什么和不能做什么？我提到过**不做**的部分吗？如果您还没有，请在开始之前检查一下。

[](/@hrajpal96/antlr-and-code-generation-a71ead442005) [## ANTLR 和代码生成

### 在这一点上，驱动每一种现代编程语言的写作风格的思想是使用…

medium.com](/@hrajpal96/antlr-and-code-generation-a71ead442005) 

# **在同一页上？**

**没有？**好的，概括地说，你有一些特定格式的输入文本，基于这些文本你想要创建你自己的领域特定语言，简称为 **DSL** ，那么以下是你需要记住的:

*   **Lexer** :将来自**输入文本**的一串字符转换成一串标记。
*   **解析器**:处理令牌，可能创建 AST。
*   **抽象语法树(AST)** :解析输入的中间树表示，比令牌流更容易处理。它也可以被多次处理。
*   **树解析器**:它处理一个 AST。
*   **String Template** :一个支持使用带有占位符的模板来输出文本的库(这是 ANTLR 特有的)。

在这里，我们更进一步，扩展了解析器的概念，以转换基本的布尔表达式，如下所示，并在一个文件中生成单独的 python 函数，然后可以使用上面的字符串模板思想将这些函数用作 python 模块。

![](img/f8889105ae05997940e1a07b2f4e3c3c.png)

# 建立 ANTLR 项目

我在这里使用的设置将是一个在 VSCode 上创建的 Java-Maven 项目。
[用于 VSCode 的 ANTLR 插件](https://marketplace.visualstudio.com/items?itemName=mike-lischke.vscode-antlr4)提供了各种选项(甚至比 Intellij 更多)来调试你的语法文件，并创建了视觉上漂亮的解析树来轻松调试特定配置的用户输入语句。

![](img/225978f208232e5d6449bef9131f5203.png)

来源:[https://github . com/Mike-lischke/vs code-antlr 4/blob/master/doc/grammar-debugging . MD # setup](https://github.com/mike-lischke/vscode-antlr4/blob/master/doc/grammar-debugging.md#setup)

要生成这些可视化效果，需要使用 vscode 的 ANTLR 启动配置，在调试模式下运行带有语法的指定输入文件的语法文件。下面快速浏览一下 VS 代码上 ANTLR 的 launch.json 配置文件:

让我们首先为我们的解析器创建一个基本语法或 BooleanExpr.g4 文件。

注意解析器文件是如何以**语法 BooleanExpr 开始的；**。这可以通过将词法分析器标记(用大写字母表示的标记)和解析器标记(**所有其他的**)保存在两个不同的文件中来分解:

一个用于解析器，另一个用于词法分析器，这样会更方便，因为更容易维护。接下来，我们开始定义一个头和包名，它将被放在生成的解析器类的开头。这将允许我们在 java 代码中指定一个要导入的包。

从我们的 lexer 开始，我们将标识符定义为 Lexer 规则，并提供与之匹配的描述:

Lexer 规则总是以大写字母**开始。这些规则是解析器的基本构建块，重点是构建解析器规则的基础。对于任何对正则表达式有所了解的人来说，这应该有点熟悉。**

这里，`A-Z`表示`A`和`Z`之间的字母，`a-z`表示`a`和`z`之间的字母。类似地，`0-9`表示数字`0`和`9`。类似于正则表达式🤔？因为，规则可能包含也可能不包含这些字母的多次出现，这些字母可能会以(*/+)操作符作为后缀，指示它们出现的频率。这里，`*`表示可能根本不存在(零次或多次)。这意味着我们的`IDENTIFIER`规则将匹配大写、小写的任意组合，总是以大写/小写字母和整数字符开始，而不是空字符。

通常，所有的空格都由 lexer 标记。因此，您必须定义空白，以及在解析器规则中可以使用空白的所有可能位置。然而，由于我们的源布尔表达式在某些地方不需要空间敏感，我们可以编写一个 lexer 规则来处理这个问题。

请注意空白标记的定义是如何编写的，以标识一个或多个空格、制表符和换行符，并让 ANTLR 跳过它们。箭头(-->)操作符定义了遇到令牌时要执行的操作(在本例中为 skip)。接下来是为包含多个操作符和操作数的布尔表达式定义标记。这包括以下令牌:

## 嵌入动作

规则 **GT，GE，LT，LE** 和 **EQ** 包含代码块，允许它们在遇到各自的令牌时执行某些操作。这允许在语法文件本身中定义某些动作，但是一般要注意的是只定义小而简单的动作，而不要定义复杂的代码块。

> 如果我们不想要构建解析树的开销，我们可以在解析过程中计算值或者实时打印出来。另一方面，这意味着在表达式语法中嵌入任意代码，难度较大；我们必须理解动作对解析器的影响，以及在哪里放置这些动作。
> [**权威 ANTLR 4 参考**](http://media.pragprog.com/titles/tpantlr2/listener.pdf)**——**[**特伦斯帕尔**](https://parrt.cs.usfca.edu/)**(ANTLR 背后的男人)**

注意每个规则是如何由空格分隔的字母组成的。这些被称为**片段**。他们的主要目标是减少每个令牌的混乱定义，这本质上需要处理区分大小写的用例😵。这样，用户不需要书写所有可能的文本组合来识别相同的标记。这些定义如下

虽然大多数字母数字标记可以通过使用片段来构建，但其他标记可以通过自定义正则表达式定义或使用引号中的普通字符串来构建，如， **LPARENTHESIS** 和 **DECIMAL_NUMBER。**

另一方面，解析器规则(所有其他规则)从小写字母开始。这些规则的主要目标是在 DSL 中定义布尔表达式的上下文，并帮助从生成的 lexer 标记中构建解析树或抽象语法树。

## 基本构建模块- ✔️

让我们开始定义我们的规则。首先，我们定义我们的根节点(或者，通常所说的**解析**节点)，它本身应该只指向一个规则(这里是 **basicBooleanExpression** )。这从一个 return 语句开始，该语句包含它应该返回的变量(可选，但在我们的例子中是必需的),以及它的返回类型。

该规则指向另一个名为 basicBooleanExpression 的规则，后面跟有`EOF`(或者，**文件结尾**)字符。不包含这实质上意味着您试图解析整个输入，只解析输入的一部分是可以接受的，这样可以避免任何语法错误。

## 感到惊奇的🤔？简化版如下…

使用`EOF`字符是因为如果在解析`basicbooleanExpression`规则时出现语法错误，那么解析规则*将*尝试从中恢复，并报告收集到的语法错误并继续，因为`EOF`是完成规则所必需的，但解析器尚未达到。

因为，我们已经定义了语法并把它分成了两个独立的文件，所以我们使用了包含词法分析器的选项，作为我们在解析器文件中定义规则的词汇表:

```
options {
tokenVocab = BooleanExprLexer;
}
```

回到我们的解析器，第一个规则或***basicbooleanexpression***规则是用三个选项定义的，它们应该总是向我们的 python 目标代码评估返回一个布尔值。第一个是后两个 having 规则的组合，两个布尔表达式与一个**逻辑 and/or** 运算符的组合，第二个是另一个**三元**表达式，使用比较器(like、less or lt)比较两个基本表达式返回的某个值，最后，第三个是一元**(只是一个布尔值，比如真或假)。**

**这些规则由 **'|'** 运算符分隔。这意味着在识别输入字符串时，basicBooleanExpression 可以基于从左到右的文本识别递归地引用它的任何一个子规则。**

**在 **basicBooleanExpression** 中的每条规则要么被赋予一个变量名，如 left、right(表达式中的左和右操作数)和 op(操作符的简称)，要么是一条单标记规则。`$str`变量用于分配当前表达式解析的结果，在规则的开始使用`returns [String str]`返回，并与每个规则链接使用，直到达到一个 lexer 规则。**

**`#`用于标记每个规则，以便它在目标语言解析器(在我们的例子中是 Java 解析器类)中有专用的监听器方法。**

**以下是语法文件的完整链接:**

**[](https://github.com/hrajpal96/BooleanParser/blob/master/src/main/antlr/BooleanExprLexer.g4) [## hrajpal96/BooleanParser

### 在 GitHub 上创建一个帐户，为 hrajpal96/BooleanParser 开发做贡献。

github.com](https://github.com/hrajpal96/BooleanParser/blob/master/src/main/antlr/BooleanExprLexer.g4) [](https://github.com/hrajpal96/BooleanParser/blob/master/src/main/antlr/BooleanExprParser.g4) [## hrajpal96/BooleanParser

### 在 GitHub 上创建一个帐户，为 hrajpal96/BooleanParser 开发做贡献。

github.com](https://github.com/hrajpal96/BooleanParser/blob/master/src/main/antlr/BooleanExprParser.g4) 

在下一节中，我将为语法文件生成目标语言解析器，并进一步探究语法树的生成及其遍历模式和类型。

[](/@hrajpal96/python-from-expressions-the-antlr-series-part-2-5436ef00bcf) [## 来自表达式的 python-ANTLR 系列(第 2 部分)

### 在前一部分中，我创建了两个 ANTLR 语法文件，用于将基本布尔表达式转换为 python 表达式…

medium.com](/@hrajpal96/python-from-expressions-the-antlr-series-part-2-5436ef00bcf)**