# 正则表达式——数据科学的重要工具

> 原文：<https://medium.com/analytics-vidhya/regular-expression-a-very-important-tool-for-data-science-6806110b5e43?source=collection_archive---------18----------------------->

2020 年 6 月 5 日

![](img/5f374f2a905e8a8f51712458d48a19c1.png)

由[马库斯·斯皮斯克](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

写这篇博客的灵感:fastai 课程(Jeremy 说正则表达式是考虑学习的重要工具。完成第一部分课程后，我想写一篇关于这个的博客，但是忘记了。我应该早点写这篇博客，但是在我浏览 fastai v2 时才想起这个话题)

检查 [fastai v2](https://dev.fast.ai/)

正则表达式是一个字符序列，主要用于查找和替换字符串或文件中的模式。

让我们讨论使用正则表达式可以解决的问题。(示例来自 fastai 课程)

在解决深度学习问题时，我们有数据集，可能会有标签存储在文件名中的时候。这样我们就有了路径，并需要从中提取标签。或者你可能需要从网站上提取信息。在这些或类似情况下，正则表达式是重要的工具。

让我们先从简单的例子开始。

假设你有一个文档，你想搜索名字为‘Kiran’(姓氏可以是任何东西)的所有人的名字，该怎么做呢？？
这里正则表达式开始发挥作用。

正则表达式:' **Kiran\s\w+\s'**

这里的 s 表示一个空格，\w 表示字符+表示一个或多个字符。
这将提取名字为 Kiran、姓氏为 Kiran 的所有姓名。

让我们看看标签位于文件名路径中的示例:

`data/oxford-iiit-pet/images/american_bulldog_146.jpg`

`data/oxford-iiit-pet/images/german_shorthaired_137.jpg`

美国牛头犬是这个形象的标签。
但是怎么提取呢？？？

编写正则表达式类似于我们处理问题的方式。看到上面的例子，我们可以知道标签是在最后一个正斜杠(/)后找到的，在标签后我们有数字，路径以`.jpg`格式结束

正则表达式是 **/([^/]+)_\d+.jpg$**

我会一步步解释。

**$** 表示我们正在解释的文本结束

。确保在文本结束之前，我们有正确格式的 jpg。

**\d** 表示数字位数，+表示多位数。

数字前有下划线吗

**(【^/]+】**是为了寻找一组不含正斜杠的字符，【】表示我们感兴趣的字符。**^**‘是否定。

**开始时的正斜杠**是告诉我们当我们点击正斜杠时搜索结束。

**/([^/]+)_\d+.jpg$** 给出了我们想要的标签，即本例中的`american_bulldog`。

`python code`

```
import re string = 'data/oxford-iiit-pet/images/american_bulldog_146.jpg' 
pat = r'([^/]+)_\d+.jpg$' 
pat = re.compile(pat) 
print(pat.search(string).group(1)) >american_bulldog
```

重要的正则表达式备忘单:

```
^ Start of string 
$ End of string 
\b Word boundary 
* 0 or more 
+ 1 or more 
? 0 or 1 
\s White space 
\S Not white space 
\d Digit 
\D Not digit 
\w Word 
\W Not word 
\ Escape following character 
\n New line 
\t Tab 
. Any character except new line (\n) 
\[a-z] Lower case letter from a to z
\[A-Z] Upper case letter from A to Z
(a|b) a or b 
\[abc] Range (a or b or c) 
\[^abc] Not (a or b or c) 
\[0-7] Digit from 0 to 7
```

我已经用两个例子解释了正则表达式，但是目的是向你介绍正则表达式以及它能做什么。这篇博客旨在向您介绍正则表达式的强大功能。如果学会了如何使用正则表达式，它会成为你数据科学工具箱中的重要工具。

感谢您阅读博客。

*最初发布于*[*https://kirankamath . netlify . app*](https://kirankamath.netlify.app/blog/regular-expression-a-very-important-tool/)*。*