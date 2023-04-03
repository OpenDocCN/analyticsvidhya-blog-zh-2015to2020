# 测试 Python 装饰器

> 原文：<https://medium.com/analytics-vidhya/testing-python-decorators-3c5777e4524d?source=collection_archive---------1----------------------->

这篇文章的灵感来自我的同事，他最近问了我一个关于如何为装饰者编写测试的问题。目的是为刚接触装饰、测试或两者兼有的人提供一个温和的推动力来掌握这个主题。

如果你对装饰者不熟悉——在 [*的人们，realpython*](https://realpython.com/) 有一个很好的开始指南:[Python 装饰者入门](https://realpython.com/primer-on-python-decorators/)

如果您完全不熟悉编写测试(并且有大约 1 小时的空闲时间)，我强烈建议您首先阅读这些精彩的文章:

1.  [Python 测试入门](https://realpython.com/python-testing/)
2.  [使用 Pytest 进行有效的 Python 测试](https://realpython.com/pytest-python-testing/#what-makes-pytest-so-useful)

你可以在我的 [github](https://github.com/captain-fox/testing_python_decorators_sample_code) 中找到本文没有提到的所有代码样本和额外的测试用例。

每当我们为一个常规函数编写一个测试场景时,(通常)对于如何做没有困惑:取决于我们如何调用一个函数(参数、上下文、应用程序的状态等)。)其活动的产物被认为是成功或失败。

测试装饰者有一点点不同。您需要将 ***修饰的*** 函数排除在等式之外，而是验证 ***修饰器*** 是否做了它应该做的事情。

巧合的是，我有*刚刚*的例子(我过度简化了)来演示你应该如何以及为什么用测试用例覆盖装饰者。

假设我们正在开发一个有两类用户的应用程序:

1.  经理
2.  普通用户

有时，作为对某种触发的反应，应用程序会向他们发送电子邮件。类型为 ***经理*** 的用户从地址 **management@ecorp 接收电子邮件。 *com*** *，*所有其他用户都会收到来自**internal@ecorp.com**的邮件。地址规则适用于所有发出的电子邮件。

将苹果和橙子分开是一个经验法则，这就是为什么我们希望有一个函数负责建立与电子邮件服务器的连接并发送消息，另一个函数根据用户类型确定发送电子邮件的地址。

`notify_by_email` 函数负责与邮件服务器建立连接并发送消息(我们一会儿会讲到`using_email_address`装饰器)。

为了简单起见，`notify_by_email`接受两个参数:`**user**` (应该接收消息的人)和`**from_email**`(应该发送消息的地址)。让我们跳过`**html_message**` 的内容及其如何到达那里，以及与邮件服务器建立连接的过程，因为它们与我们的主题无关。注意`notify_by_email`不返回值。这样的函数通常从异步进程内部调用。如果在执行这样的函数时发生了不可预知的事情，它会被添加到日志文件中，不会导致应用程序崩溃。

装饰者的工作是决定从哪个电子邮件地址发送信息(记住:苹果和橘子！):

装饰器根据用户的类型决定使用哪个电子邮件地址，并将其与`user`一起作为`from_email`参数传递给 ***装饰的*** 函数。

这允许我们在代码中的任何地方调用`notify_by_email`，只提供`user`参数:

`@using_email_address` decorator 将负责从哪个电子邮件地址发送消息。

快进几百行代码，我们很高兴让我们的应用程序做一些有用的技巧，包括给用户发送电子邮件。我们甚至创建了一个模拟来避免发送实际的电子邮件:我们的测试用例检查`notify_by_email`是否试图使用正确的凭证与电子邮件服务器建立连接。

突然，我们的应用程序开始使用错误的地址向一类或两类用户发送电子邮件。为`notify_by_email`函数编写测试用例意味着我们在每个函数调用中都隐式地调用`using_email_address`，但是并不真正检查装饰器内部发生了什么。

让我们试着为 decorator 编写一个测试用例，就像我们为常规函数编写测试一样:我们需要用某些输入来调用`using_email_address`,并断言在它内部发生的任何事情都会产生预期的输出/结果。这种编写测试的技术被称为“黑盒测试”。

`using_email_address`接受一个参数:一个函数。它还有一个返回值:一个函数。因此，要测试一个装饰器，我们需要向它传递一个函数，并期待一个函数作为返回值。记住这一点，让我们开始为`using_email_address`编写测试用例。因为我们必须将函数作为参数传递，所以让我们创建一个函数，并将其命名为`to_be_decorated`:

如果你所看到的还没有让你想起什么——我给你一点提示:我们刚刚做了 decorator 对语法糖所做的事情，以`@`的形式应用于函数。这给了我们理解如何测试 decorators 的第一个技巧——像通常一样，用函数包装它，只是这次使用可以“玻璃盒子”的函数，这意味着您可以完全控制函数内部的事件流。

让我们以更 pythonic 化的方式重新编写测试并运行它(尽管到目前为止我们还没有任何`assert`语句):

我们收到一个错误:

`TypeError: wrapper() missing 1 required positional argument: ‘user’`

这意味着尽管我们已经重新创建了装饰函数关系，用`to_be_decorated`函数替换了`notify_by_email`，我们仍然缺少几个参数和至少一个返回语句。我们快到了！让我们修复这个错误:

正如你已经注意到的，我们已经为我们的测试用例提供了`user` fixture。`to_be_decorated`函数现在需要`user`、`from_email`和`**kwargs`作为参数。最后我们调用`to_be_decorated`，用`using_email_address`包装，除了这次我们可以直接从`to_be_decorated`函数中访问它的活动结果。

如果我们现在运行测试用例，它应该会通过。我们没有任何`assert`语句，因此这仅仅意味着没有任何东西抛出异常，并且所有的参数都就位了。

让我们再一次更新我们的测试用例，但是这次我们让`using_email_address`负责根据它接收的用户类型选择正确的电子邮件地址。我们将要测试的类型是`manager`，所以在用`using_email_address` `to_be_decorated`函数包装之后，应该会收到:

1.  `user`与第 6 行调用的`manager`对象相同的参数
2.  `from_email`带有值`management@ecorp.com`的参数

瞧啊。现在你知道了如何从测试的角度接近装饰者。

我鼓励你看看这篇文章的[样本代码](https://github.com/captain-fox/testing_python_decorators_sample_code)。您会发现更多的测试示例，我在这里没有提到。示例代码是自包含的，所以除了`Python`和`PyTest`之外，您不需要任何额外的库来运行它，并使用您自己的实现。