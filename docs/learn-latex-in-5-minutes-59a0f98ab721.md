# 5 分钟学会乳胶

> 原文：<https://medium.com/analytics-vidhya/learn-latex-in-5-minutes-59a0f98ab721?source=collection_archive---------18----------------------->

将你的工作提升到专业水平

![](img/3d95d8d66a848b04b4f5b281763385a1.png)

尼克·莫里森在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

今年之前，我从未听说过乳胶。直到一位新老师明智地建议我们应该开始为我们的研究生论文学习 LaTeX。“为什么不能直接用 MS Word？”我真诚地问道，他回答道:

> “如果你想让你的作品看起来专业，你可以使用乳胶”

他推荐我们使用背面的，一个不错的免费 LaTeX 编辑器。讽刺的是他的学科期末考试被一个项目代替了，我利用这个机会学习了 LaTeX，用专业的标准来呈现我的作品。

在本教程中，我想与你分享我对 LaTeX 的了解，这样你也可以将你的工作提升到一个全新的水平。我将指导您创建以下元素:

*   涉及
*   索引
*   标题和字幕
*   粗体、斜体和下划线
*   项目符号列表
*   形象
*   数学和方程
*   语法突出显示

我推荐你跟随背页的这个教程，因为它很简单，但是任何其他的文本/ LaTeX 编辑器都足够了！

# 涉及

你的读者从你的作品中获得的第一印象无疑来自它的封面。使用 LaTeX 为您的作品制作专业外观的封面再简单不过了。封面必须在`\begin{document}`标签之前定义，但是稍后我们将从文档内部调用封面的构造。盖子通常由以下元件构成:

## 标题

很难想象没有标题的封面，要定义它，只需键入`\title{My First Article in LaTeX!\vfill}`并给你的作品起一个适合它的标题。另外，你应该在头衔的末尾加上`\vfill`。这个标签将推动标题后面的所有内容，使其适合同一页面。这样，我们可以确保标题在页面的顶部，而像作者或日期这样的东西留在底部。

## 作者

该表扬就表扬！使用`\author{author1}`添加任意多的作者。你可以添加更多的作者，并把他们显示在同一高度，可能避免伤害自尊，就像`\author{auhtor1 \and author2}`。要垂直添加它们，将`\and`与`\\`交换，在作者之间换行。

## 加入

你的工作发生在一个机构？一定要用这个简单的标签`\affil{Universidad Carlos III de Madrid, Spain}`来提及/肯定它。

## 日期

最后但并非最不重要的一点是，详细说明你的工作发生的时间总是好的。只需添加`\date{Once upon a time}`就可以了！

正如我们之前所预期的，我们必须用`\maketitle`从文档内部构造封面，看起来应该如下所示:

```
\begin{document}\maketitle\thispagestyle{empty}\newpage\section{Introduction}\end{document}
```

注意在`\maketitle`后面有一个`\newpage`标签，确保只有封面出现在首页。我们使用`\thispagestyle{empty}`删除页面底部的页面计数器。

# 索引

在作品的开头添加一个索引是至关重要的，因为它可以帮助你的读者浏览它。我们只需要构建一次索引，然后忘记它，因为它会在我们每次编译工作时自动刷新。可以通过以下方式构建索引:

```
\begin{document}
[...]\tableofcontents{}
\setcounter{page}{1}
\newpage[...]
\enddocument 
```

请注意，我们必须将索引的页码设置为 1，否则它仍然会计算封面(虽然我们没有显示)并将索引标记为第 2 页。

# 标题和字幕

任何结构良好的论文的基础，标题和副标题在 LaTeX 中就像你已经预料到的那样简单！只需给你的头衔加上`\section{Title}`，从`\subsection{Subtitle}`一直到`\subsubsection{Sub-subtitle}`。

# 粗体、斜体和下划线

强调关键概念，这对于确保你的作品被你的读者所接受是至关重要的。作为一切与乳胶，这是相当直观的！

## 大胆的

`\textbf{Bold}`

## 意大利语族的

`\textit{Italic}`

## 强调

`\underline{Underline}`

以及以上三者的所有可能组合！例如，我们可能希望同时将**加粗**和*斜体*，这可以通过如下方式实现:

```
\textbf{\textit{Bold and Italic!}}
```

# 项目符号列表

就个人而言，我离不开项目清单，它们让一切看起来整洁有序。以下是如何用乳胶制作一个:

```
\begin{itemize}
    \item First!
    \item Second! [...]
    \item Last!                      
\end{itemize}
```

# 形象

图片为你的作品增添了活力，让它更容易被理解，所以别忘了添加一些！为了在 LaTeX 中添加图像，我们必须在文档的开头添加下面的包:`\usepackage{graphicx}`。

现在添加图像很容易，只需使用`\includegraphics{img.jpg}`。但是编辑它们要比我们习惯的花费更多的精力。居中和成像的最简单方法是用`center`包围它:

```
\begin{center}
        \includegraphics{img.jpg}
\end{center}
```

改变它的尺寸有点复杂。为了让您有所了解，让我们来看看如何更改图像的宽度，使其与文本的宽度相匹配:

```
\includegraphics[width=\textwidth]{img.jpg}
```

# 数学和方程

作为一名计算机系的学生，我发现自己在论文中加入方程的时间比我希望的要多。幸运的是，在 LaTeX 中添加方程和其他事情一样简单。

要在文本中添加公式，我们必须用美元符号将公式括起来:

```
Newton's second law: $F=ma$
```

如果你想在另一行显示你的方程，我们必须用`\[ ... \]`把方程括起来

```
Newton's second law: 
\[F=ma\]
```

如果你想用 LaTeX 更深入地研究数学和方程，我推荐你看看下面这篇文章，作者是 [Andre Ye](https://medium.com/u/be743a65b006?source=post_page-----59a0f98ab721--------------------------------)

[](https://towardsdatascience.com/latex-for-data-scientists-in-under-6-minutes-3815d973c05c) [## 面向数据科学家的乳胶，不到 6 分钟

### 任何数据科学简历的必备技能

towardsdatascience.com](https://towardsdatascience.com/latex-for-data-scientists-in-under-6-minutes-3815d973c05c) 

# 语法突出显示

语法突出显示对我来说是必须的，因为我一直围绕代码工作，我必须说，与其他流行的文本编辑器相比，LaTeX 在这一领域的性能给我留下了深刻的印象。为了突出显示用户语法，我们必须在文档的开头添加另一个包`\usepackage{minted}`。

要使用它，您必须遵循以下结构:

```
\begin{minted}{python}while(True):
   print("Hello World!")\end{minted}
```

就我而言，它也支持`java`和`c`，只需将`python`换成你喜欢的语言，就万事俱备了！

# 最后的想法

LaTeX 是一个强大的工具，可以让你的工作看起来更专业。我很高兴能够分享我的知识，因为我知道如何使用它，希望你会觉得有帮助！关于 LaTeX，您可以随时问我任何问题，或者查看背页的 LaTeX 文档，这是我学习的主要资源。

编辑愉快！

感谢您的阅读。