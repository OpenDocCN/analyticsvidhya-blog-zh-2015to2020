# 用令人敬畏的“lobstr”软件包成为一名更好的 R 程序员

> 原文：<https://medium.com/analytics-vidhya/become-a-better-r-programmer-with-the-awesome-lobstr-package-af97fcd22602?source=collection_archive---------0----------------------->

![](img/38a4013ba695d5dc3e92c6b5947eb565.png)

图片提供:[https://pix abay . com/en/coding-programming-working-macbook-926242/](https://pixabay.com/en/coding-programming-working-macbook-926242/)

> “工具会放大你的才能。你的工具越好，你越知道如何使用它们，你的工作效率就越高。”—安德鲁·亨特，务实的程序员

程序员的主要工具是他或她对编程语言的选择。说到数据科学，R 一直是我构建模型的首选语言。

出于各种原因，r 是最流行的编程语言之一(至少在数据科学中是这样):

*   易于使用的语法
*   优雅的可视化/绘图系统
*   丰富的包装生态系统，以及
*   丰富的社区支持

是一个直观的 R 包，有潜力让你成为更好的程序员。当我在搜索 [R Infrastructure](https://github.com/r-lib) 的 GitHub 页面寻找新的 R 包时，我偶然发现了`lobstr`，它被证明是一个非常有用的包！

# 关于 lobstr

`lobstr`由令人惊叹的 Hadley Wickham 设计，试图帮助普通开发者更好地理解 R。用他自己的话来说，`lobstr`提供了深入挖掘 R 对象细节的工具。

![](img/e35dfc5da46fab583d8b8ef030be6e12.png)

`lobstr`也可以被认为是`str` base-R 函数的改进版本。

# 安装软件包

`lobstr`尚未在 CRAN 上发布，所以目前只能从 GitHub 安装。在使用下面的安装代码之前，请确保您已经安装了`devtools`包。

```
# install.packages(“devtools”)
devtools::install_github(“r-lib/lobstr”)
```

# lobstr 中的不同函数

`lobstr`提供三个简单的功能:

*   `ref()` —参考文献
*   `ast()` —抽象语法树
*   `cst()` —调用堆栈树

这三个函数服务于三个不同的目的，我们会了解`ref()`和`ast()`的细节。我们现在将省去`cst()`,因为它仍在进行初始开发和测试。

## **参考文献— ref()**

你有没有想过当你把一个现有的 R 对象赋给一个新的对象名会发生什么？它是否创建了一个新的对象，使内存翻倍，或者它只是创建了一个引用？

会帮助你理解这一点。为了回答上述问题，让我们创建一个简单的数字向量。我们就叫它`simple_vector`。现在，让我们从同一个`simple_vector`创建一个新的列表，我们称之为`double_vector`。

我们在列表`double_vector`中两次使用`simple_vector` 的原因是为了检查 R 是否分配了两个不同的内存空间，或者它是否只是引用了原来的`simple_vector`。

最后，我们将使用`simple_vector`和`double_vector`创建另一个列表`triple_vector`。请注意，对象`double_vector`和`triple_vector`的类型为`list`(非向量)。

```
**library**(lobstr)

simple_vector <- c(2.0,3.0,4.0)

double_vector <- list(simple_vector,simple_vector)

triple_vector <- list(double_vector,simple_vector)

ref(simple_vector)
*#> [1:0x7f9ba555aa58] <dbl>*

ref(double_vector)
*#> █ [1:0x7f9ba26be4c8] <list>* 
*#> ├─[2:0x7f9ba555aa58] <dbl>* 
*#> └─[2:0x7f9ba555aa58]*

ref(triple_vector)
*#> █ [1:0x7f9ba13fcf08] <list>* 
*#> ├─█ [2:0x7f9ba26be4c8] <list>* 
*#> │ ├─[3:0x7f9ba555aa58] <dbl>* 
*#> │ └─[3:0x7f9ba555aa58]* 
*#> └─[3:0x7f9ba555aa58]*
```

现在使用函数`ref()`，我们可以找到 R 对象的内存引用，正如你在上面看到的，`*0x7f9ba555aa58*` 是`simple_vector`的内存引用，当你`ref(double_vector)`时，你可以看到列表对象两次引用回`simple_vector`的内存引用，同样的情况也发生在`ref(triple_vector)`上，其中`triple_vector`引用两次，一次是引用回`simple_vector`的`double_vector`的地址，另一次是引用到`simple_vector`本身。

出色地绘制了一个树形结构，帮助我们可视化内存引用。这样，我们就可以确定我们是在 R 中创建新的内存分配，还是引用现有的内存对象，从而在编写代码时进行更好的内存管理。

## 抽象语法树— ast()

正如维基百科上提到的，

> 在计算机科学中，**抽象语法树** ( **AST** )，或者仅仅是**语法树**，是用编程语言编写的源代码的抽象语法结构的树表示。

像每一种编程语言一样，R 中的任何表达式都可以用语法树的形式来表达。在开发和测试复杂的表达式时，以 ASTs 的形式可视化表达式非常有帮助。

让我们测试一下，看看它能为我们做些什么。首先，使用`ast(1+2)`来可视化一个简单的加法表达式，显示出`+`是将`1`和`2`传递给的操作符。这导致了加法操作，但是当我们必须将加法操作的输出分配给一个新对象时，那么`<-`就变成了根节点。

```
library(lobstr)#ast#simple additionast(1 + 2)
#> █─`+` 
#> ├─1 
#> └─2#simple addition with result assignmentast(x <- 1 + 2)
#> █─`<-` 
#> ├─x 
#> └─█─`+` 
#> ├─1 
#> └─2
```

虽然我们可以继续使用`ast()`来理解复杂的表达式，但是对于另一个琐碎的(但令人困惑的)操作来说，也很方便，即操作符优先。

表达式`y <- 2 + 3 * 5 / 9 ^ 2`很难在几秒钟内手工完成，即使它包含简单的算术运算符。这是因为在我们的头脑中使用运算符优先级并不总是容易的。但是这里有`ast()`在做同样的事情:

```
#operator precedenceast(y <- 2 + 3 * 5 / 9 ^ 2)
#> █─`<-` 
#> ├─y 
#> └─█─`+` 
#> ├─2 
#> └─█─`/` 
#> ├─█─`*` 
#> │ ├─3 
#> │ └─5 
#> └─█─`^` 
#> ├─9 
#> └─2
```

很神奇，对吧？

# 摘要

因此，通过使用`lobstr`函数`ref()`和`ast()`，我们可以更好地进行 R 编程——编写内存高效的代码，并以更好的方式理解表达式求值。你在上面看到的完整代码可以在[这里](https://github.com/amrrs/blogpost_codes/blob/master/lobstr_intro.R)获得，而`lobstr` 文档可以在这里[获得。](http://lobstr.r-lib.org/)

您对此套餐有什么体验？请在下面的评论中告诉我们！