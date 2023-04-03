# 使用 Python 构建一个密码生成器

> 原文：<https://medium.com/analytics-vidhya/learning-python-with-passwords-54e63a1d983b?source=collection_archive---------17----------------------->

从一个实际项目开始您的 Python 之旅

![](img/3814b3bd8f7ddc91b79a2791659c2ba6.png)

图片来自 [RealPython](https://realpython.com/python-basics/)

Python 是世界上最受初学者和专业人士欢迎的编程语言之一。这是有充分理由的。了解 Python 将打开 web 开发、机器学习和数据科学的大门。凭借其易读的语法和广阔的社区，开始您的 Python 之旅变得前所未有的容易。

**工具**

很多课程和教程都是通过去[python.org](https://www.python.org/)下载最新版本的 Python 来介绍 Python 的。从那里，你必须下载一个叫做“IDE”(集成开发环境)的应用程序，比如 IDLE、PyCharm 或 VSCode。尽管这些年来获得一个 IDE 已经变得非常容易，但是对于初学者来说，设置一个 IDE 还是有些麻烦。

现在，如果我告诉你，你已经有了一个很棒的 Python IDE，实际上不需要任何设置，会怎么样？输入[谷歌 Colab](https://colab.research.google.com/) 。这个基于网络的 Python IDE 非常适合初学者，每个拥有谷歌账户的人都可以免费使用。

在本文中，我们将使用 Colab 来教授您 Python 的核心要点，并应用它来制作一个随机密码生成器。

**基础呢？**

当谈到任何新技能时——无论是烹饪、写作还是编码——很容易陷入试图记住每个细节的困境，从而试图爬上学习曲线。然而，正如任何职业厨师、作家或程序员会告诉你的，学习基础知识的最好方法是通过小项目。我们将直接做我们的第一道菜，看看我们需要什么样的基础知识，而不是先学习刀术和如何烧水。

当然，在你开始你的第一个项目之前，你需要了解一些核心要点，但是我保证我们会很快完成。

**类型、变量、循环，天啊！**

Python 中的每个对象都有一个特定的“*类型*”与大多数语言不同，当你创建一个对象时，Python 通常会理解它是什么类型，而不需要你显式地写出来。一些基本类型是整数(非十进制数，如-1、0 或 5)、字符串(由字母和数字组成的字符串，如“Hello World”)和列表(在其他语言中也称为数组，只是其他对象的有序组)。如果您不确定对象的类型，行`print(type(yourObject))`将打印出类型。还有许多其他类型，如布尔值、浮点数和元组，但提到的三种是我们将在这个项目中使用的类型。

*变量*就像是对象的容器。类似于`name = "Alice"`的一行“保存”了变量`name`中的字符串对象`"Alice"`。通过像这样声明一个变量，Python 能够记住`name`以备后用。

*循环*是整个编码的一个巨大部分。循环允许程序员自动迭代重复的动作，而不是键入每一行代码。在大多数编码语言中有两种主要类型的循环:T4 循环和 T5 循环。当某个条件为真时，`while`循环重复指定的动作。通常，当我们不确定代码的某一部分会重复多少次时，while 循环非常有用。`for`循环用于迭代或系统地遍历一个序列，如字符串、列表或元组。将某一行代码执行指定次数的一个非常有用的“公式”是`for i in range(n):`，其中“I”代表总共“n”次重复中的一次迭代。

我跳过了许多其他重要的东西，比如 if-else 语句、函数、面向对象编程等等。如果你喜欢这个项目，我鼓励你通过围绕你想学的技能创建你自己的项目来学习我跳过的东西。

**创建密码生成器**

现在你已经学习了一些基本知识，我们将开始创建我们的随机密码生成器。我们的生成器将询问用户在密码中需要多少个字母、数字和特殊字符，并打印出符合这些规范的密码。

进入 [Google Colab](https://colab.research.google.com/) 点击右下角的“新笔记本”。这将在你的 [Google Drive](https://drive.google.com/) 中创建一个名为“Colab Notebooks”的文件夹，你所有的笔记本都将保存在这里。点击中间的单元格；您可以在这里键入代码，并通过单击左侧的播放按钮来运行代码。当您阅读教程时，我建议您自己键入代码，并在继续下一步之前尝试理解它为什么有效。

首先，我们需要导入两个“模块”模块是赋予 Python 更多特性的函数和对象的包。

```
import random
import string
```

您可以使用#后跟一些文本来注释代码。Python 会跳过任何跟在#后面的内容，所以这对于让代码更容易理解非常有帮助。

```
# Imports
import random # Used for random generation
import string # Added string object functions
```

接下来，我们将为每个可能的字母分配一个变量，为每个数字分配一个变量，为每个特殊字符分配一个变量。我们将简单地使用字符串模块，而不是手动输入字符串中的每个字符(像`"0123456789"`)。

```
letters = string.ascii_lettersnumbers = string.digits # Same as "0123456789"special_chars = string.punctuation
```

现在我们已经定义了什么是字母、数字和特殊字符，我们可以要求用户输入。内置函数`input()`为用户打印出指定的提示，然后我们可以将答案保存为变量。我们将使用`input()`来询问用户在最终密码中需要多少个字符。因为用户输入的类型是字符串，所以我们必须使用“强制转换”将输入转换成整数我们可以通过用`int()`包装`input()`将字符串转换成整数。注意，如果 Python 不能将输入转换成整数，它将抛出一个错误。有一些方法可以确保用户使用 try-except 语句键入整数，但这超出了本课的范围。

```
letter_num = int(input('How many letters would you like in your password? '))number_num = int(input('How many numbers would you like in your password? '))special_num = int(input('How many special characters would you like in your password? '))
```

是时候使用`random`模块了！首先，我们必须定义一个空白密码，以便以后添加。这是通过定义一个空白字符串来实现的，比如:`password = “”`。为了从我们之前制作的字符串中随机选取一个字符，我们可以使用函数`random.choice()`从字符串中随机选取一个字符。然后使用`+=`操作符，我们可以将新字符添加到现有密码中。我们必须对每种类型的字符(字母、数字、特殊字符)都这样做，并重复用户想要的次数。这是一个很好的例子，说明 for 循环可以帮助我们解决问题。下面是一个字母循环的例子(请记住，您必须在 for 循环中缩进代码):

```
for i in range(letter_num):password += random.choice(letters)
```

通过简单地对其他字符重复这种格式，并包括我们的密码定义，我们得到:

```
password = ''for i in range(letter_num):password += random.choice(letters)for i in range(number_num):password += random.choice(numbers)for i in range(special_num):password += random.choice(special_chars)
```

现在密码的问题是，它的所有字母后面都是数字，后面是特殊字符，如“XYZ123$@#”我们希望随机混合字符串，使密码看起来更像“1X@Y$Z23#”只需要一行代码，但是要做的事情很多:

```
password = ''.join(random.sample(password, len(password)))
```

行`random.sample(password, len(password))`接受我们的密码字符串，随机选取指定数量的字符，不重复(我们将其设置为密码的长度)。该函数输出一个类似于`['1', 'X', '@', 'Y', ..., '#']`的列表，但是如果我们打印出来，用户将无法使用它。用语句`''.join()`包装会将列表的每个部分合并成一个字符串，我们将它保存在变量 password 中。然后我们可以打印出`password`，这样用户就可以看到了。

```
print('Password: ' + password)
```

恭喜你！您刚刚完成了您的第一个 Python 项目，它已经拥有了比自动生成的密码更多的功能。在真实的应用程序中，这个密码将被正确地加密和保存，而不是简单地打印出来，但是这是理解 Python 如何工作的一个很好的生成器。

完整代码

**现在怎么办？**

既然您已经有了一些经验，那么如何使用 Python 就完全取决于您了。这种语言多功能性的好处意味着你可以用它解决很多问题。几乎每一个可以用另一种语言解决的问题都可以用 Python 解决，而且网上有很多免费资源。

如果你对如何使用 Python 自动化任务感兴趣，我推荐 Al Sweigart 的[用 Python 自动化枯燥的东西](https://automatetheboringstuff.com/)。如果你想学习机器学习和 AI 能做什么，Google 有很多资源有 [TensorFlow](https://www.tensorflow.org/) 。或者如果你只是想做自己的项目，有像 [StackOverflow](https://stackoverflow.com/) 和 [GitHub](https://github.com/) 这样的网站可以在你不可避免地面临某个问题时为你提供指导。

我希望这个简短的项目给了你信心，你也可以学习如何编码，此外，我希望它向你表明，编码并不总是很难。最后，编码与其说是关于语法和调试，不如说是关于逻辑思维和解决问题。成为一个更好的程序员会帮助你成为一个更好的问题解决者。