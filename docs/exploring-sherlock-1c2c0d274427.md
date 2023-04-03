# 探索夏洛克

> 原文：<https://medium.com/analytics-vidhya/exploring-sherlock-1c2c0d274427?source=collection_archive---------13----------------------->

一个搜索社交网络用户名的工具。

这篇文章是我理解一个开源项目的尝试。项目本身和代码属于作者和合作者。我只是想弄清楚这是如何工作的，以及在这个项目中实际使用了哪些框架和库。夏洛克的 Github 回购在这里是。

所以，让我们从夏洛克做什么开始。这是一个简单的工具，可以在 300 多个网站中搜索特定的用户名。我的一个朋友告诉了我这个工具，我们讨论了如何实现它。在阅读了[这个](https://github.com/sherlock-project/sherlock/wiki/Adding-Sites-To-Sherlock)维基之后，我开始知道我们最初的猜测是正确的。也就是说，夏洛克依赖于状态代码和来自点击这些不同网站的 URL 的响应。夏洛克有三种方法可以发现一个网站中是否有注册了特定用户名的用户。

1.  *检查 HTTP 状态代码*:有效的用户名不会在响应中发送 404 错误。
2.  *检查响应 URL* :响应 URL 的存在表明没有用户使用所查询的用户名，网站返回一个重定向 URL 到主网站或注册页面。
3.  *检查错误消息:*检查站点响应中的错误或寻找任何重定向。

关于上述方法的详细讨论以及如何存储站点和查询用户名可以在[这里](https://github.com/sherlock-project/sherlock/wiki/Adding-Sites-To-Sherlock)找到。此外，该项目的 wiki 和被扫描网站的列表可以分别在[这里](https://github.com/sherlock-project/sherlock/wiki)和[这里](https://github.com/sherlock-project/sherlock/blob/master/sites.md)找到。

![](img/f6004d80797cf60a36be9061f17c4e94.png)

照片由 [Hitesh Choudhary](https://unsplash.com/@hiteshchoudhary?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

夏洛克项目的 Github 回购可以在这里找到[。现在，让我们直接进入代码。](https://github.com/sherlock-project/sherlock)

通过阅读 readme，我们可以很容易地看出它是一个命令行应用程序，需要 Python 才能运行，并且可以用 Docker 运行。所以，让我们快速浏览一下`requirements.txt`文件，了解一下是什么在幕后驱动着夏洛克。以下是夏洛克使用的所有软件包的列表。

*   美丽组 4>=4.8.0
*   bs4>=0.0.1
*   认证> =2019.6.16
*   色彩> =0.4.1
*   lxml>=4.4.0
*   比重瓶> =1.7.0
*   请求> =2.22.0
*   请求-未来> =1.0.0
*   汤筛> =1.9.2
*   茎> =1.8.0
*   torrequest>=0.1.0

还有，我们走个小弯路，深入了解一下这些套餐。

*   [Beautiful Soup 4](https://pypi.org/project/beautifulsoup4/):Beautiful Soup 帮助我们解析和修改 HTML 和 XML 文件。它允许我们操作页面，就像 javascript 中的`document`允许我们操作网页一样。它实际上将原始的 HTML 文件转换成 Python 对象。这些对象最常见的类型包括`Tag`、`NavigableString`、`BeautifulSoup`和`Comment`。对象允许我们使用 HTML 或 XML 标签以及它们的名称和属性。`NavigableString`指 HTML 标签对象内部的任何文本。`BeautifulSoup`包含整个解析后的文档。我们可以迭代地在文档中找到`next_`或`previous_`元素，或者声明性地找到`find`或`find_all`元素、字符串、带有正则表达式的标签、列表或属性字典。完整的文档可以在[这里](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)找到。
*   [bs4](https://pypi.org/project/bs4/) :所以 bs4 只是一个安装美汤包的哑包。
*   [certify](https://pypi.org/project/certifi/):certify 拥有验证 TLS 主机身份的根证书。也就是说，它确保所述网站的 SSL 证书是有效的，并且是由其根证书存在于 certificate CA 包中的 CA 之一发布的。
*   [colorama](https://pypi.org/project/colorama/) : [ANSI 控制序列](https://en.wikipedia.org/wiki/ANSI_escape_code)用于控制终端上的颜色、光标位置等操作。Colorama 使这些控制序列在 MS windows 中工作。
*   [lxml](https://pypi.org/project/lxml/) :处理 xml 和 HTML 文档的库。它为 C 库 libxml2 和 libxslt 提供 Pythonic 绑定。它基于 [ElementTree API](http://effbot.org/zone/element-index.htm) ，将 XML 文档定义为[元素](http://effbot.org/zone/element.htm)对象的树。
*   [py SOCKS](https://pypi.org/project/PySocks/):[SOCKS](https://en.wikipedia.org/wiki/SOCKS)协议通过代理服务器帮助客户端和服务器之间交换网络数据包。关于 SOCKS 代理的更多信息可以在[这里](https://securityintelligence.com/posts/socks-proxy-primer-what-is-socks5-and-why-should-you-use-it/)找到。PySocks 允许创建 Socks 代理，作为一个隧道，转发流量而不修改它。
*   [requests](https://pypi.org/project/requests/) : Requests 是一个完整的 Python HTTP 库。此处可找到一份易于遵循的文档[。请求可用于发出各种 HTTP 请求、读取响应、在 URL 中传递参数、处理原始/二进制/JSON 内容、传递自定义头或 JSON 数据、检查响应状态代码、获取或发布 cookies 等。](https://2.python-requests.org/en/master/user/quickstart/)
*   requests-futures:Requests-futures 提供了与 Requests 库完全相同的功能，并增加了异步执行。此外，它返回未来对象而不是响应对象。
*   soupsieve 是一个 CSS 选择器库，使用现代 CSS 选择器提供选择、匹配和过滤。
*   [stem](https://stem.torproject.org/index.html) : Stem 是与 Tor 对话的 Python 库。它用于使用 Tor 控制协议连接到 Tor 进程。
*   [torrequest](https://github.com/erdiaker/torrequest) :一个包装请求和 stem 的包装器，用于通过 Tor 发出 HTTP 请求。

在我们知道使用了什么库以及用于什么目的之后，我们可以开始查看它们在项目本身中是如何使用的。

当我们查看项目文件时，我们可以安全地忽略 docker、travis、code_of_conduct、contributing 文件和 tests 文件夹。让我们先来看看这两个数据文件:`data.json`和`data_bad_site.json`

*   在`data.json` 文件中，`errorType`指的是出错时在响应中查找的一条信息。因此，状态代码的错误类型将查找 HTTP 状态代码，而消息的错误类型将在响应中查找特定的错误文本。此外，用于检查用户名的 URL 格式由作为用户名占位符的`{}`组成。
*   `data_bad_site.json`包含夏洛克目前的检测算法不支持的站点。文件`removed_sites.md`包含了网站不被支持的详细原因。

好了，现在我们来看最后三档:`site_list.py`、`sherlock.py`和`load_proxies.py`。

*   `load_proxies.py`:所以，这个文件中的函数在代码中有很好的记录。它提供了从 CSV 文件中提取代理、根据 wikipedia.org 检查代理以及返回仅工作代理列表的函数。
*   `site_list.py`:这个脚本使用命令行参数-r 来更新`data.json`中出现的站点的 Alexa 排名。每个请求在单独的线程中运行，结果写入`sites.md`文件。它使用 XML ElementTree 对象来查找`REACH`标签，并最终从 Alexa API 返回的响应中提取`RANK`属性。来自 Alexa API 的典型响应如下所示:

```
<!-- Need more Alexa data?  Find our APIs here: [https://aws.amazon.com/alexa/](https://aws.amazon.com/alexa/) -->
<ALEXA VER="0.9" URL="elwo.ru/" HOME="0" AID="=" IDN="elwo.ru/">
<SD><POPULARITY URL="elwo.ru/" TEXT="252350" SOURCE="panel"/><REACH RANK="235507"/><RANK DELTA="-115898"/><COUNTRY CODE="RU" NAME="Russia" RANK="22456"/></SD></ALEXA><?xml version="1.0" encoding="UTF-8"?>
```

*   `sherlock.py`:该项目的核心模块。让我们逐一解释这个文件的主要部分。

1.  `main`功能:初始化 colorama，用于在终端打印彩色文本和背景。为 tor、代理、打印、按等级排序、颜色、输出文件夹、详细程度等定义命令行选项[。在函数内部定义一个要修改的](https://docs.python.org/dev/library/argparse.html)[全局](https://www.geeksforgeeks.org/global-keyword-in-python/) `proxy_list`。检查和验证传递的各种冲突参数，并设置代理，检查它们，或者是否应该使用 tor 来发出请求。检查是否提供了用于加载网站列表的 URL。从 JSON 文件中加载站点列表，或者是完整的，或者只是在`site_list`命令行参数中传递的一个子集。检查`site_list`中通过的站点是否受支持。还检查我们是否需要根据网站的排名排序列表。然后，设置文件夹和文件来存储获取的信息。最后，调用夏洛克函数以及所有必需的参数，它执行搜索用户名的任务。如果用户名被实际检测到并写入输出文件，最终的`results`字典将被搜索。此外，如果指定，结果将写入 CSV 文件。
2.  用于检查超时有效性、打印有效和无效结果、信息、错误等的实用功能。
3.  这个类增加了每个请求的总响应时间。`Future`类似于 Javascript 中的`Promise`，用于异步调用。所以它覆盖了基类的 request 方法:`FutureSession`，定义了一个被钩子调用的方法`timing`，检查是否有其他带有`response`键的钩子，并将`timing`添加为要执行的第一个函数。
4.  `sherlock`功能:如果需要，初始化 tor。将会话中的最大线程数设置为 20。创建扩展的`ElapsedFutureSession`对象。在发出请求之前，在 site_data 中添加 User-Agent 头和额外的头(如果有的话)。如果给定站点的用户名的正则表达式检查失败，则阻止发出请求。检查每个网站的`errorType`,以了解为用户名检测收集信息的范围。是否允许重定向以及是否使用 HEAD 或 GET HTTP 方法。最后，使用上面指定的选项、头、超时、URL 等发出请求。存储请求调用返回的`future`。使用`get_response`函数提取响应时间、错误类型和响应。从响应中提取状态代码和响应文本。检查状态代码是否为 2XX 类型，检查响应中的错误消息是否符合 data.json 文件中的定义，或者是否存在重定向，并相应地更新`exists`状态。最后，将状态、响应文本和经过的时间保存在结果中，并将最终结果作为字典返回。
5.  `get_response`函数:返回响应、错误类型和总响应时间。处理与连接、代理和重试次数相关的各种错误。