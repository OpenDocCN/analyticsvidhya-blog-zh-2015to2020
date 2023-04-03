# Python 新手？让我们开始吧…

> 原文：<https://medium.com/analytics-vidhya/python-newbies-lets-get-started-c9f9be72e1dd?source=collection_archive---------22----------------------->

![](img/e9cdae8811d91cd84330edf592ed7dbf.png)

弗朗西斯科·加拉罗蒂在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 第一天

你是程序员吗？不，没关系，我们将从头开始，让你清楚 Python 及其语法的基本概念。

当我在大学二年级时，我决定现在我必须接触 python，但是我没有找到合适的导师来帮助我从哪里开始成为 python 专家。所以我开始靠自己，但我花了很长时间来理清我的概念，因为我当时感到害羞，如果有人帮助我，这是很好的，否则就让它去吧。然后，随着时间的推移，我意识到，当你使用“Google Ustad”时，你可以学习并理清你对每一件事情的概念。Ustad 是一个乌尔都语单词，意思是老师。但是如果你为某样东西奋斗，那么它一定会变得有价值。就我所拥有的而言，当你有合适的路线图可以遵循时，我不想让任何人挣扎得那么厉害。

所以现在轮到你通过弄清楚 python 的一些基本概念来接触 python 了。我会分享我在 python 中所学到的知识，从最基础的到高级的。一旦你知道了语言的正确语法，那么接下来的一切都依赖于你的大脑根据它来制定逻辑和程序。构建逻辑是一种艺术，每个程序都可以用不同的方式编写。我相信，如果某个问题被交给一个有十个学生的班级，每个学生都有不同的想法，那么这个问题也有十种可能的解决方案。

# 输入输出语法:

所以每种语言的新手都要从输入输出语法这样的基础开始。所以，我们也从那里开始。

```
variable1=input("Enter your Name :")
Output:
Enter your Name :Adnan khan
```

正如你在上面的图中看到的，我们用变量 1 的名字声明变量，并简单的传递输入给它。在 python 中，您可以通过简单地调用 input 函数来获取输入。这里你应该记住一件事，默认情况下每一个输入都是字符串。稍后我们将简要讨论字符串数据类型。现在，如果你想接受其他类型的输入，让我们看看你会怎么做

```
variable_int=int(input("Enter the number of your wish :"))
print(variable_int)
type(variable_int)Output:
Enter the number of your wish :15
15
int
```

在上面的代码中，你可以看到我们以整数的形式接受了用户的输入。同样，您可以接受用户的所有其他类型的输入。我们可以简单地使用函数 type()来检查变量的数据类型。type()函数的目的是返回对象的类型。这就是输入函数。

现在让我们讨论如何从程序中获得输出。Python 的语法很容易被人类理解。对于输出语句，我们简单地使用 Print()语句。

```
print("Hello python!")Output:
Hello Python!
```

你可以注意到 print 语句，它打印了括号内的语句。你可以通过简单地写在括号内和引号内来写你想写的任何东西。同样参见图 2，我们使用 print 语句来打印我们已经声明的 variable_int。这就是关于输入和输出语法的全部内容。

# 数据类型:

你不知道什么是数据类型？所以这不是一个大问题，我们会详细讨论每一个数据类型，你会清楚你的概念。

## Int:

它通常被称为整数或 int，包含没有小数的实数，如 1，9，110 等。整数可以是任意长度。这是定义整数数据类型最简单的方法。

## 浮动:

浮点数类似于整数，但它们之间的主要区别是浮点数包含小数，如 1.5、2.3、65.7 等。Float 只精确到小数点后 15 位。

## 十进制:

Decimal 类似于 float 数据类型，但 float 的精度仅限于 15 个小数点。所以，有时它会导致一个错误，它是如何产生错误的，让我通过代码展示一下

```
(1.1+2.2)==3.3Output:
False
```

根据人类思维的计算，这两个数字是相等的，但是 python 拒绝了它。由于计算机只理解二进制(0 或 1)，浮点以二进制形式存储在内存中，这些数字没有得到正确存储，这就是它拒绝的原因。

我们通常在下列情况下使用十进制。

*   当我们制作需要精确十进制表示的应用程序时。
*   当我们想要控制所需的精度时。

## 复杂:

python 中很少使用复杂数据类型。如果你读过《学校生活中的基础数学》,你会理解类似 3+5j 这样的术语。它包含两个值，一个是实数，另一个是虚数。带有“j”项的值是虚数。

```
var=3+5j
type(var)Output:
complex
```

这就是 python 中输入输出和数据类型的全部内容。这是开始的第一天，我将尝试在每天的基础上讨论一点我的知识，到目前为止我已经得到了什么。我不是专业人士，但我相信与人分享你的知识对我来说肯定也是有益的。我将分享一些免费的资源，在那里你可以免费学习 python。

[](https://www.datacamp.com/courses/intro-to-python-for-data-science) [## Python 简介

### 掌握 Python 中数据分析的基础知识。通过使用 numpy 学习科学计算来扩展您的技能。

www.datacamp.com](https://www.datacamp.com/courses/intro-to-python-for-data-science) [](https://cognitiveclass.ai/blog/event/introduction-python-data-science) [## 面向数据科学的 Python 介绍(2016-11-10)

### Python 是一种免费的开源编程语言，近年来广受欢迎。在这次聚会上…

cognitiveclass.ai](https://cognitiveclass.ai/blog/event/introduction-python-data-science) 

如果你有阅读书籍的毅力，那么你应该跟着书，通过解决一些基本的例子，你会学到很多东西。

Nichola Lacey 的 Python 示例

你可以从谷歌上免费下载。现在去吧，把你的手弄脏。解决尽可能多的例子。

# 小贴士:

明确你对问题的概念。

找到可能的解决方法。

不要因为错误而生气。从这些小错误中，你会学到很多东西。

尝试模拟运行您的代码并检查输出。

教别人强自己的基础。

请给我宝贵的反馈。从你们那里学到一些知识对我来说太好了。