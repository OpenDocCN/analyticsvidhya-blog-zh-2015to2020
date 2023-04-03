# 使用 Python 编程语言介绍数据科学。(第 1 条:列表、字符串和字典)

> 原文：<https://medium.com/analytics-vidhya/introductory-note-on-data-science-using-python-programming-language-36d6ed167a9?source=collection_archive---------40----------------------->

在开始本系列“使用 Python 编程语言介绍数据科学”之前，我们有必要知道什么是数据科学。因此，数据科学融合了数据推理、算法、开发和技术，以解决复杂的问题。因此，我们可以说，数据科学是一个需要编程技能、数学和统计知识的专业知识的研究领域。现在，数据科学有了一个更简单的定义，它基本上处理原始数据或信息，并从中提取有意义的结论或推论，这使我们人类更容易理解和研究数据。因此，数据科学家处理原始数据，提取有意义的数据推断，并将机器学习算法应用于数据推断，提取的数据可能是图像、视频、文本等。来制作一个人工智能模型。现在，这些模型按照程序员的训练执行特定的任务，以各种方式造福人类。
你可能会想为什么我选择 python 来编码，数据科学家喜欢 python 有很多原因，尽管你可能会认为 Ruby 非常出色，可以执行数据清理和数据抢劫等任务，但 python 具有更多的机器学习库，这也是选择 python 的一个重要原因。现在，如果我们从语法的角度来看，我发现 python 语言的语法非常简单，易于理解和学习，但我认为这因人而异，所以你会看到凡事都有利弊，你必须知道他/她最适合哪里。

最后，在简要介绍了数据科学之后，我们就可以开始我们的系列了。现在，如果你对 python 语言一点都不熟悉，我强烈建议你首先了解 Python 的基本语法，以及数据类型和 Python 类型&序列的基本概念，因为你需要先了解上述主题的基本知识，才能继续。
让我们开始吧，检查你所有的先决条件，首先我们将工作清单。

# 列表。

列表是有序的、可改变的或可变的集合。在 Python 中，列表用方括号[ ]书写。在下面的代码片段中，我声明了一个列表“x”和，你可以在列表上执行的两个基本操作也被显示出来，它们是“append()”和“remove()”。现在，为什么我提到这些操作是我们在列表上执行的“基本操作”呢？这是因为列表是可变的，这意味着您可以改变列表的内容，比如添加元素或删除元素。元组不能访问这些操作，因此它们是不可变的。所以你看，python 为我们提供了列表和元组，在列表中你可以改变列表的内容，而元组可以用来存储常量或任何类型的数据，这些数据不应该以任何方式改变。下面是给你的代码片段:

因此，最初，列表‘x’有元素[1，2，‘c’]，但是在使用 append()函数时，我们向列表中添加了整数 99。为了说明 append()函数对各种数据类型都有效，在代码中，我们还有 append('abc ')，它是一个字符串，添加起来没有任何麻烦。然后，我们在代码片段中使用了 remove()函数，该函数删除括号中提到的任何特定项/元素。这就是关于如何在列表中添加或删除元素的全部内容。
人们肯定希望在一个列表中单独迭代或访问数据，所以让我们这样做吧。下面的代码片段向我们展示了如何使用循环方法和索引操作符方法遍历列表中的项目的两种方法:

我们甚至可以在这里看到 len()函数，它简单地返回列表的长度，在上面的例子中，
end= "，帮助我们打印同一行中的元素。

接下来，我们将发现我们还可以用列表做什么。
**1。让我们使用'+'来连接列表。**

```
print([1,2] + [3,4])
#prints ->[1, 2, 3, 4]
```

**2。元素重复**
使用' * '重复列表

```
print([1,2,3]*3)
#prints ->[1, 2, 3, 1, 2, 3, 1, 2, 3]
```

**3。找到一个特定的元素**
让我们使用‘in’操作符

```
x3= 1 in x1
print(x3)
#prints -> True
# As the 'in' operator checked for us whether 1 is in the list x1 or not
```

我想强调的另一件事是，我发现它很酷，它是关于解包一个序列，可以通过运行下面给出的代码来执行。

```
'''Let us now see Sequence unpacking '''
x6 = ('Ayush', 'Singh', '[singh@x.com](mailto:singh@x.com)')
first, last, email = x6
print(first)
#prints ->Ayush
print(last)
#prints -> Singh
print(email)
#prints -> [singh@x.com](mailto:singh@x.com)
```

好了，这就是我想展示的关于列表的全部内容，但是有很多操作你可以在列表上执行，并且可以使用它们，因为我认为这是学习更多的最好方法。

# 用线串

接下来，让我们研究 python 中的字符串。如果您熟悉数据类型，或者有相当具体的编码背景，那么您肯定对这种数据类型了如指掌。在 python 中，可以对字符串执行很多操作，即使在数据科学领域，对字符串的操作也起着重要作用。我们将首先看看如何分割一个字符串，从中获得一个子串。字符串切片非常简单，您可能已经猜到了，它提取字符串的一部分，其返回类型是字符串本身。看看下面的代码片段。

```
x4 = 'Dr. Abhishek Singh'
print(x4[0]) 
#first character, prints -> D 
```

这有助于我们打印字符串“x4”的第一个字符。

现在，看一下这段代码:

```
print(x4[0:1])
#prints ->D
```

这也打印了与上面的代码相同的输出，但是我们已经显式地设置了结束字符，就像在字符串遍历的终点一样。所以，切掉一个字符串就这么简单。看一下下面的代码片段，我在那里执行了一些操作，注释将帮助您了解所执行的操作的动作以及所提到的输出将增强您的理解。

注意:split 返回一个字符串中所有单词的列表，或者一个按特定字符拆分的列表。

关于字符串，在连接时，我总是特别强调一些东西。请确保在连接之前将对象转换为字符串。
Chris' + 2，将显示一个错误为- > TypeError:只能将 str(不是“int”)连接到 str。这是做这件事的正确方法。

```
print('Python' + str(2))
#prints -> Python2
```

这是对字符串的总结，因为我发现这些操作是必不可少的，值得强调，但是如果你想了解更多，强烈建议你阅读更多的文档。

# 字典

在 Python 中，Dictionary 是包含在{ }中的数据值的无序集合，每个值都与一个键相关联，因此，它就像一个地图。字典保存键:值对。
因此，字典将键与值相关联，因此，如果我们想要访问字典中的特定值，我们必须知道该特定值的键。关于字典中的键，还有一个信息，它们不允许多态。现在让我们用字典工作。在下面提供的代码片段中，x5 是一个字典，它保存两个值-'[ayush@x.com](mailto:ayush@x.com)'和'[abhishek@x.com](mailto:abhishek@x.com)'，现在正如我们前面提到的，字典保存 key: value 对，现在很明显' Ayush '和' Abhishek '是键。
那么这里是声明一个字典的一般格式:
var _ name = { '<Key>':'<Value>' }
随代码一起写的注释会帮助你更好的理解。

```
x5 = {'Ayush': '[ayush@x.com](mailto:ayush@x.com)', 'Abhishek': 'abhishek.com'}
print(x5['Ayush']) # Retrieve a value by using the indexing operator
#prints->[ayush@x.com](mailto:ayush@x.com)
print(x5['Abhishek'])
#prints ->abhishek.com
```

让我们对字典执行一些很酷的操作。
1。**迭代所有的键和值:**

```
for n in x5.keys():
 print(n)
#prints->Ayush Abhishek

for n in x5.values():
 print(n)
#prints->[ayush@x.com](mailto:ayush@x.com) abhishek.com
```

**2。一起访问键和值:**

```
for name, email in x5.items():
    print(name)
    print(email)
#here we can access the keys as well as value, together
''' 
prints-> Ayush
[ayush@x.com](mailto:ayush@x.com)
Abhishek
abhishek.com
'''
```

这对于本文来说已经足够了，但是肯定会有更多的文章发表，关注使用 Python 的数据科学。您可以在:
[https://github.com/ayush-670/PythonDataScience_basics](https://github.com/ayush-670/PythonDataScience_basics)
找到完整的源代码。如果您对数据科学感兴趣，并且喜欢这篇文章，请继续关注。如果您有任何疑问，我很乐意与您互动。
非常感谢你阅读这篇文章！我希望它对你有所帮助。