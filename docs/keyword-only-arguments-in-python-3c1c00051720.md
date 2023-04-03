# Python 中的纯关键字参数

> 原文：<https://medium.com/analytics-vidhya/keyword-only-arguments-in-python-3c1c00051720?source=collection_archive---------0----------------------->

## Python 中星号的魔力—第二部分

![](img/a2f353e66c0db0369c27b67699387c17.png)

照片由[塞德里克 X](https://unsplash.com/@cedericx?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

所以，你已经阅读了 python 中星号的[基础魔法，现在想要更多？当然可以。该进城堡了。如果您还没有准备好，请随意访问上一篇文章，当您觉得准备好了时再回来。或者至少，你已经试过你自己的了。](/analytics-vidhya/the-magic-of-asterisks-in-python-aed3538deef9)

在本文中，我们将主要讨论为 python 提出了**关键字** **参数**的 [PEP3102](https://www.python.org/dev/peps/pep-3102/) 。

# 关键字参数

在 [python 词汇表](https://docs.python.org/3.8/glossary.html)中，你会找到一个定义*参数*在 python 中的含义的部分。就像任何其他编程语言一样，参数是在进行函数调用时传递给函数(或方法)的值。在 python 中，有 **2 种参数**:

1.  关键字参数
    关键字参数是前面带有标识符的参数(命名参数)。注意这里的值是在函数调用中传递的，而不是定义函数参数时的默认值。
2.  位置参数
    简单地说，位置参数是所有不是关键字参数的参数。

一个例子是当使用`requests.get`时，我们经常直接在字符串中传递 URL。在这个场景中，我们只是给函数一个参数`url`的位置参数。但是，如果我们传递一个类似于`requests.get(url='https://your-url.com/')`的值，那么我们传递的是一个关键字参数。

> 不要把自己和**必选**和**可选**的争论搞混了。这两个是因为当我们用它的参数定义一个函数的时候。如果我们包含一个缺省值的参数，以后这个参数就变成可选的了。否则，我们需要传递一个参数，因此需要位置参数。

现在，当我们调用一个函数时，由我们来决定是传递位置参数还是关键字参数。但是，因为 python 维护可读性，所以考虑的是，如果参数显然是要传入的，那么我们只是将值作为位置参数传递。相比之下，同样最有可能的是，我们希望其他用户知道我们将什么值传递给什么参数。所以，关键字论点。

# 仅关键字参数

在定义函数时使用星号，给了我们更高层次的魔力。它叫做**纯关键字参数**。这种类型的参数与普通的关键字参数有相似的行为，但是它迫使用户传递关键字参数，而不是位置或直接参数。

```
def generate_username(*users, separator, length=None):
    users = [user.lower().split() for user in users]
    username = [separator.join(user) for user in users]
    if length:
        username = [user[:length] for user in username]
    return username
```

让我们以上面的函数为例。该功能将以简单的方式生成用户名，并带有用户自定义的分隔符。注意，这里使用了打包参数。所以，我们可以称之为

```
>>> generate_username("John", "Mr Smith", "Mrs Smith", separator="_")
```

并将结果`['john', 'mr_smith', 'mrs_smith']`。

现在，你问的魔法在哪里？尝试将`"_"`作为位置参数传递，如下所示。

```
>>> generate_username("John", "Mr Smith", "Mrs Smith", "_")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: get_username() missing 1 required keyword-only argument: 'separator'
```

接受那个错误！您缺少 1 个**必需的仅关键字参数:分隔符**。那是什么？实际上，这就是在函数中使用星号后跟一个必需的参数`separator`的效果。

正如 PEP 3102 所述，函数中星号后面定义的每个参数都需要传递**仅关键字参数**。也就是说，通过指定参数名。这允许我们在星号后面定义不会打包在一起的附加参数。因为我们用参数`users`嵌入了星号，所以它变成了一个**可选的位置打包参数**。

> **可选位置打包参数**是一个打包参数，就像在[上一篇文章](/analytics-vidhya/the-magic-of-asterisks-in-python-aed3538deef9)中描述的一样。

在我们为`separator`传递一个关键字参数之前，上面的函数将吸收任何位置参数。在上面的例子中，如果我们不想为参数`users`捕获任何位置变量，该怎么办呢？我们可以使用魔法，只要一个星号。让我们重新定义上面的函数。

```
def generate_username(users, *, separator, length=None):
    users = [user.lower().split() for user in users]
    username = [separator.join(user) for user in users]
    if length:
        username = [user[:length] for user in username]
    return username
```

这里的区别是我们不能传递任何数量的位置参数，而是将它作为集合(列表或元组)传递给参数`users`。

```
>>> list_users = ["jhonny", "Jadon Smith", "Mr Smith", "Mrs Smith"]
>>> generate_username(list_users, separator="_")
['jhonny', 'jadon_smith', 'mr_smith', 'mrs_smith']
```

如果我们传递一个`"_"`作为位置参数，它将抛出如下所示的错误。

```
>>> generate_username(list_users, "_")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: generate_username() takes 1 positional argument but 2 were given
```

这是因为星号`*`后面的任何剩余参数都需要一个关键字来传递参数，如下面的 PEP 3102 所述。

> 由于 Python 要求将所有参数绑定到一个值，并且将一个值绑定到一个只有关键字的参数的唯一方法是通过关键字，因此这样的参数是“必需的关键字”参数。

# 外卖食品

使用 python 中的星号，我们有 [2 个选项](https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/#Positional_arguments_with_keyword-only_arguments)来定义只有关键字参数的函数:

1.  带有位置参数的仅关键字参数
2.  没有位置参数的仅关键字参数

这两个场景是在 PEP 3102 中首次提出的，你可以阅读更多关于这个魔法存在的理由。

# 参考

[1]特雷·亨纳，[Python 中的星号:它们是什么以及如何使用它们](https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/#Positional_arguments_with_keyword-only_arguments) (2018)

[2] [人教版 3102](https://www.python.org/dev/peps/pep-3102/)