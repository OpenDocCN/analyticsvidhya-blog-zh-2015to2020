# python——一种更好的脚本语言。

> 原文：<https://medium.com/analytics-vidhya/python-a-better-scripting-language-112c665c7de2?source=collection_archive---------1----------------------->

如果说调试是去除软件 bug 的过程，那么编程一定是把 bug 放进去的过程。—埃德格·迪杰斯特拉

# 脚本面临的挑战

不考虑我们在软件开发中的角色，我们经常使用 Linux/Unix(*nix)。在*nix 上使用命令行(CLI)时，我们需要编写 shell 脚本。我们通过脚本自动完成一些事情，或者使用 shell 命令对一些数据进行数字处理。大多数时候我们已经知道经常使用的命令，否则我们谷歌一下。但是，如果任务很长或很复杂，就会变得很乏味，并且很难找到合适的命令或组合两个以上的命令。

现在，编写脚本并不像使用一两个带有正则表达式和管道的命令那么简单。这变得很痛苦，我们找到一些解决方法或者放弃。但是无论哪种方式都是耗费时间和精力的任务，所以我们倾向于避免它，因为它会浪费你的实际工作时间。除非你是全职命令行脚本编写员。

有时你会抽出一些时间完成剧本，并为自己感到骄傲(这是你应该的)。但是大约一个月后，当你访问同一个脚本并试图找出一种方法来调整它以适应新的/变化的需求时，你会发现自己完全迷失在这个过程中，因为当谈到脚本时，很难理解你的代码(在长时间访问它之后)。现在有人可能会说，这不仅仅局限于脚本，其他编程任务也会发生这种情况。真的！但是我要说的是，使用脚本更有可能发生这种情况，原因是使用脚本构建抽象的过程比使用其他花哨的编程语言更复杂，因此当使用脚本时，我们经常会编写上下文敏感的代码。

因此，让我们来看看如何使用 python 来帮助克服这些问题。

# 为什么 python 是编写脚本的好选择？

*   Python 是一种非常高级的语言，它具有与所有其他典型的高级编程语言相似的特征。

Python 为您处理了大量的复杂性，语法比其他经典脚本语言更加用户友好。

*   Python 还提供了一个很好 REPL(解释器),这对于试错策略来说非常方便。
*   丰富的库支持和巨大的开发人员社区，用于 IRC、邮件列表和堆栈溢出的问答。
*   用 python 编写的与操作系统交互的库是围绕操作系统特定库的包装器。这样，这些库就是通用。

因此，您可以轻松地在一个特定的操作系统上编写 python 脚本，并可以在另一个操作系统上使用，而无需任何更改。

*   要使用 python 编写脚本，你不需要完全成熟的 IDE，你只需要一个编辑器(vim 或任何其他)和一个 python 二进制文件。

所以从这个意义上来说，我可以说它是非常轻的。

*   我认为选择 python 编写脚本而不是传统脚本语言的最重要的原因之一是

python 为您提供了比 shell 更好的编程环境。也就是说，您可以通过 python 代码启动 shell 命令来提取数据，并将输出带到更好的编程环境中，您可以对数据进行建模，从数据中提取一些模式，对数据进行计算，将其写入另一个文件，使用 python 代码通过网络轻松发送数据。所有这些都变得很容易，因为 python 提供了典型编程语言的特性，构建抽象的过程变得更加容易，从而使代码更具可读性、可维护性和可扩展性。

# 让我给你看一个简短的例子。

假设我们有一个文件，其中包含一些包含 IP 地址的文本(类似于“ip addr”命令的输出)。现在的任务是首先从文件中提取所有的 IP 地址，然后逐一检查 IP 地址是上升还是下降。这个任务听起来可能很简单，但是我强烈建议您在这里暂停一下，尝试使用 shell 命令编写相同的脚本，或者至少尝试思考一下如何完成这个任务。

文件:样本文本文件

```
~]# ip address add 192.1.2.223/24 dev eth1
~]# ip address add 192.165.4.223/24 dev eth0
~]# ip addr
3: eth1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP qlen 1000
    link/ether 52:54:00:fb:77:9e brd ff:ff:ff:ff:ff:ff
    inet 192.168.8.1/24 scope global eth1
    inet 192.168.4.23/24 scope global eth1
```

完成了吗？

现在让我们用 python 脚本来做这件事。

```
import sys
import os
import re

*# function to extract all ip address from a file.*
**def** **get_all_ip_from_file**(file_path):

    file_data **=** open(file_path, "r")**.**readlines()
    ip_address_list **=** [] *# List to store ip address*
    **for** line **in** data:
        ips_on_line **=** re**.**findall( r'[0-9]+(?:\.[0-9]+){3}', line ) 
        ip_address_list **=** ip_adress_list **+** ips_on_line
    **return** ip_address_list

*# function to check if the ip is Up or Down*
**def** **check_if_ip_up_or_down**(ip):

    response **=** os**.**system("ping -c 1 " **+** ip) 
    **if** response **==** 0:
        **print** ip, 'is up!'
        **return** True
    **else**:
        **print** ip, 'is down!'
        **return** False

**def** **main**():
    ip_list **=** get_all_ip_from_file('sample_text_file.txt')
    **for** ip **in** ip_list:
        status **=** check_if_ip_up_or_down(ip)
        **print** ip **+** " --> "**+** str(status)

main()
```

我希望你能从这个小演示中明白这一点。请随意在下面的评论区写下你的评论/意见/批评。我很乐意讨论更多。编码快乐！！！

*原载于*[*saurabhkukade . github . io*](https://saurabhkukade.github.io/Python-A-Better-Scripting-Language/)*。*