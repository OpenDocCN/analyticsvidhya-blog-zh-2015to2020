# Python Lambda 函数

> 原文：<https://medium.com/analytics-vidhya/python-lambda-functions-78d3673aebfd?source=collection_archive---------17----------------------->

![](img/3f38283d8bd29ab5bc4868dbc1f3e279.png)

Python lambda 函数是一个非常有价值的特性。使用 Python 的好处之一是，与其他编程语言相比，它简化了代码结构。这意味着用更少的代码行做更多的事情。

然而，情况并非总是如此。如果您不知道 python 提供了哪些工具来实现这一点，那么编写冗长的代码是非常容易的，因为它的“紧凑性”使选择 Python 变得毫无意义。

一个这样的工具是 lambda 函数。这些允许我们在 python 中创建匿名函数。如果您以前读过优秀的 python 程序员的代码，您可能会遇到类似这样的情况:

```
add = lambda a,b : a + b
```

上面的代码可能看起来有点混乱，让我通过将 lambda 函数分解成组件来阐明它的语法。Lambda 函数有 3 个组成部分:lambda 关键字(当然)、一组参数和一个表达式。

```
lambda [(optional) arguments] : [expression]
```

# 规则

在我们详细讨论 lambda 函数的实现之前，让我们先讨论一下它们的规则:

1.  Lambda 函数是无名的(除非赋给变量)。不能像对“def”关键字那样，在“lambda”关键字后分配函数名。
2.  参数是可选的。Lambda 函数可以在不传递任何参数的情况下定义，就像常规函数一样。
3.  Lambda 函数只能有一个表达式。
4.  就像常规函数一样，你可以从 lambda 函数返回任何东西。也有可能什么都不回。
5.  Lambda 函数可以赋给变量重复使用。这是命名 lambda 函数的唯一方法。
6.  Lambda 函数可以从常规函数返回。
7.  Lambda 函数可以返回其他 lambda 函数。

# 为什么使用 Lambda 函数？

如果你以前从未使用过 lambda 函数，你可能认为没有它们你也能做得很好，你是对的。然而，正如我之前所说的，它们在确保我们维护干净、紧凑和 pythonic 式的代码方面起着重要的作用。

Lambda 函数在需要匿名函数一次性使用或重复使用的情况下最有用。

# 履行

如前所述，我们可以使用 lambda 函数和常规函数来生成函数。让我们创建一个例子，我们想要创建 3 个函数。一个是给定数字的两倍，一个是给定数字的三倍，一个是给定数字的四倍。我们可以通过创建 3 个不同的函数来实现这一点:

```
def doubler(n): return n * 2 def tripler(n): return n * 3 def quadroupler(n): return n * 4
```

我们可以不写上面的代码，而是创建一个函数，返回一个带有我们想要的功能的 lambda 函数:

```
def function_generator(n): return lambda a : a * n
```

现在我们创建我们想要的函数:

```
doubler = function_generator(2) tripler = function_generator(3) quadroupler = function_generator(4)
```

function_generator 函数返回一个函数，该函数将其参数与传递给 function_generator 的任何参数相乘。我们可以用下面的语句对此进行测试:

```
print(doubler(4), tripler(4), quadroupler(4)) # Output 8 12 16
```

通过在一行中嵌套两个 lambda 函数，可以进一步简化函数生成器:

```
function_generator = lambda n : lambda a : a * n
```

之前的测试行应该产生与之前相同的输出。

记住，赋给变量的 lambda 函数使得变量可以作为函数调用。在引入话题时，我举了以下例子:

```
add = lambda a,b : a + b
```

这使得“add”可调用。因此，要将两个数字相加，我们将使用以下语句:

```
add(4, 6) # Returns 10
```

这也适用于返回 lambdas 的函数。只要 lambda 函数被返回，它就是可调用的，不管嵌套有多深。返回的函数也没有名字(因此是“匿名的”)，除非它被赋给一个变量。

我们也可以在 lambda 函数定义后立即调用它:

```
(lambda x : x * 2)(4) # Returns 8
```

确保存储/打印返回的值，因为以后在您的模块代码中将无法再访问它。这种格式对于理解列表非常有用。假设我们有一个数字列表(num ),我们想要生成一个新的数字列表，这样对于 num 中的每个数字 n，n * 2 > 50 应该返回 True。如何才能实现这一点？

使用常规函数和列表理解:

```
def validate(n): 
    return n * 2 > 50 nums = [3, 1, 4, 22, 45, 1, 245, 6, 34, 4, 8, 14] 
nums2 = [n for n in nums if validate(n)] # Returns [45, 245, 34]
```

但是，我们可以完全删除函数定义，并在列表理解中包含验证表达式:

```
nums = [3, 1, 4, 22, 45, 1, 245, 6, 34, 4, 8, 14] 
nums2 = [n for n in nums if (lambda x : x * 2 > 50)(n)] 
# Returns [45, 245, 34]
```

使用 Python lambda 函数使我们的代码比前一个例子更紧凑。

**如果你喜欢这篇文章，可以考虑关注我的** [**个人网站**](https://kelvinmwinuka.com/) **，以便在我的内容在媒体上发布之前提前获得(别担心，它仍然是免费的，没有烦人的弹出广告！).另外，请随意评论这篇文章。我很想听听你们的想法！**