# Python 乍一看

> 原文：<https://medium.com/analytics-vidhya/python-at-first-glance-a73bb8c7107e?source=collection_archive---------18----------------------->

![](img/34b2ef9a94ef3ea764de7a5b7e8b4a4b.png)

在 [Unsplash](https://unsplash.com/s/photos/python-programming?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上由 [Hitesh Choudhary](https://unsplash.com/@hiteshchoudhary?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

本周，我决定通过 LinkedIn Learning 的入门课程来了解一下 Python。为什么是 Python？Python 一直是大多数编程语言排名网站上许多类别的顶级语言之一，包括对该语言的热爱、使用它的程序员数量、学习的容易程度等等。它是数据科学和非编程领域中最常用的语言。在本文中，我将讨论一些我对 Python 的初步了解。

# 格式化

我发现 Python 的一个独特之处是它用于函数、类等的格式化结构..Python 不像 Javascript、C#、Java 和其他语言那样使用普通的花括号，而是使用缩进系统。在声明一个函数或给出 if 语句的条件后，使用冒号来打开函数，然后对其中包含的每一行使用缩进。这里有一个比较的例子:

```
// Javascript
if (a > b) {
    console.log(a)
}# Python
if a > b:
    print(a)
```

我们在这里可以看到的另一个格式差异是 if 条件没有括号。像 Ruby 一样，在 Python 中，括号对于条件或参数通常是可选的。此外，Python 中的注释以#开始，而不是 Javascript 中使用的//。

# 条件句和循环

Python 不同于大多数其他主要语言的另一个方面是在条件和循环领域。以 if/else 语句为例。你可能会认为大多数语言在这一点上几乎是一样的，但是下面是 Python 与 Javascript 和 Ruby 的区别:

```
# Python
if a > b:
    print(“a is greater”)
elif b > a:
    print(“b is greater”)
else:
    print(“a and b are equal”)// Javascript
if (a > b) {
    console.log(“a is greater”)
} else if (b > a) {
    console.log(“b is greater”)
} else {
    console.log(“a and b are equal”)
}# Ruby
if a > b
    puts “a is greater”
elsif b > a
    puts “b is greater”
else
    puts “a and b are equal”
end
```

所以在三种不同的语言中，有三种写 else if 的方法。Python 中的条件语句与其他语言的另一个不同之处是，在 Python 中不能使用 switch 语句，而只能使用 if/elif。

除了条件语句，Python 还利用了循环。虽然许多语言都有一些不同的循环类型，但是 Python 保持了简单，只有两种类型；while 循环和 for 循环。

# 建筑风格

像大多数现代语言一样，Python 可以使用面向对象的编程，其特性包括类、方法和继承。许多人更喜欢使用 OOP 编程，因为它带来了良好的编码实践的好处。

Python 也可以与过程式编程风格一起使用。这是一种在 C 和 Basic 等老语言中常见的老风格。它基于例程或函数，通常被认为不如 OOP 安全。

# 导入库

Python 的另一个特性是导入定制库，这几乎是所有语言的标准。此功能可以通过以下方式实现:

```
#Python
from datetime import time
import os
```

这个特性对于简化工作流非常重要，因为许多常见的函数已经编写在库中，开发人员可以随时使用。

# 附加功能

在 Python 中，我还发现了一些有趣的特性。首先是通过 datetime 库访问和处理日期和时间的能力。通过特定的命令，我可以查看和操作年、月、周、时间等信息，甚至可以创建日历。使用 datetime 时出现的另一个特性是能够通过日历在 Python 中生成 HTML 编码的日历。call 日历调用。

另一个很棒的特性是通过代码处理文件的能力。使用 Python 可以创建、编辑、重命名文件，甚至将文件存档为 zip 文件。

最后，我了解了如何从 web 上检索和处理数据，这是 Python 最伟大的特性之一，可能需要另一整篇文章来描述。它非常适合直接利用 JSON、XML 甚至 HTML，使得从网上获取和处理数据变得容易。它还允许通过与 JavaScript 非常相似的命令进行 DOM 操作。

到目前为止，我只是开始接触 Python 的一些潜在用法。在我过去几天短暂的一瞥中，不难看出为什么 Python 是当今最常见和最受欢迎的编程语言之一。