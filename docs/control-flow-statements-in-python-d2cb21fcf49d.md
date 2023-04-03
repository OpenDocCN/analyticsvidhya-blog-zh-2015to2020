# Python 中的控制流语句

> 原文：<https://medium.com/analytics-vidhya/control-flow-statements-in-python-d2cb21fcf49d?source=collection_archive---------8----------------------->

## 理解控制流

![](img/c506afc50bb4abd96beb56a118b37e7a.png)

照片由[韦斯·希克斯](https://unsplash.com/@sickhews?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

在本文中，您将了解控制流语句，以及它们如何构成编写您一直梦想的复杂动态 python 程序的基本构件。

你我都知道路并不总是笔直的，python 程序也是如此。Python 程序并不总是堆叠在一起并顺序执行的单行语句列表。在解决问题时，您会经常遇到需要您改变通常的执行流程的情况，这可以形成编写复杂程序的非常基本的构件。

例如，您希望您的程序能够根据用户输入的语言说“hello”、“holla”或“namaskar”。是的，你猜对了，这可以通过使用控制流来实现。

我们在控制流下的重点将是**条件语句** ( `if`、`elif`、`else`)**循环语句** ( `for`、`while`)`break` 和`continue` 语句。

## **条件语句**

条件语句也称为`if-then`语句。条件语句允许程序员根据布尔条件的结果是`True` 还是`False`来执行特定的代码块。也就是说，如果条件为真，则执行一行或一块代码，否则为假。

**语法:**

```
if condition:
    statement
       …
elif condition:
    statement
       …
else:
    statement
       …
```

看看实际中的条件语句:

```
default = enif language == ‘sp’: print(‘Holla’)elif language == ‘ind’: print(‘Namaskaar’)else: print(default)
```

如果语言是西班牙语，上面的程序打印 holla，如果是印地语，打印 namaskar，否则打印英语 hello。

```
**Note:** be mindful of the colon (:) after each condition and the indentation denoting a block of code. You should check out pythons [PEP 8](https://www.python.org/dev/peps/pep-0008/) for best practices and guidance on how to write a well-structured code for readability.
```

> 对于 python 来说，`elif` 关键字是惟一的关键字，您可以根据需要选择包含或多或少的关键字，`if` 和`else` 经常在其他语言中使用。

## **循环**

编程中的循环也称为迭代。这是一种根据特定条件重复/迭代执行某些代码语句的方式。通常，这用于迭代可迭代对象，如 list、set 等。我们的重点是`for` 和`while` 循环:

*   `for` **循环**

**语法:**

```
for item in iterable:
    statements
```

迭代项目集合:

```
countries = [‘Nigeria’, ‘France’,’Ghana’, ‘India’]
for country in countries:
    print(country)
```

*   `while` **循环**

**语法:**

```
while expression:
    statement(s)
```

打印“Hello Jane ”,直到满足条件:

```
num = 0
while (num < 3):
    num = num + 1
    print(‘Hello Jane’)
```

> 循环一直执行，直到传递的参数的值为`False`。

*   `break` 和`continue`

两者都是重要而有用的语句，可以包含在循环(for 和 while)中，以进一步增强执行流。

> **中断** —这允许你退出整个循环，即使循环条件仍然为真。
> 
> **继续** —这允许您跳过当前循环中尚未执行的语句，并继续下一次迭代。

下面的代码说明了如何在一个循环中使用`continue` 来打印出一串偶数:

```
for n in range(15):
# if the remainder of n / 2 is not 0, skip the rest of the loop
if n % 2 != 0:
    continue
    print(n, end=’ ‘)
```

下面的代码说明了如果遇到特殊字符，如何在循环中使用`break`来突破:

```
for letter in ‘Programming’:
    if letter == ‘g’:
        break
    print( ‘Current Letter :’, letter)
```

我们已经探索了 Python 中控制流语句的基础。现在是你继续冒险的时候了，记住:

> 成功既不神奇也不神秘。成功是坚持应用基本原则的自然结果
> ― **E .詹姆斯·罗恩**

像往常一样，不要忘记 [**关注我**](/@ezekiel.olugbami) 上的帖子。鼓掌分享，以示对本帖的支持。

你也可以看看我的另一篇关于如何设置你的 Python 开发环境[](/analytics-vidhya/setting-up-your-python-3-development-environment-on-windows-26d912da9d2f?source=friends_link&sk=e53c5da824c0b618e0881a6e9cbd219a)**和 [**Python 数据类型**](/analytics-vidhya/python-data-types-beginners-guide-2966d907597f?source=friends_link&sk=9e3fb1f3b735aeed43d3dd44a2dc2972) 的文章，这些文章旨在让你的 Python 编程初学者之旅变得简单。**

## ****快乐编码**:)**