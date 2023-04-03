# 理论与世界:如何从谷歌检索链接不是那么容易

> 原文：<https://medium.com/analytics-vidhya/theory-vs-the-world-how-retrieving-links-from-google-is-not-hat-easy-3ab35c8ac92a?source=collection_archive---------22----------------------->

> 你想证明一点编码对人文学科有帮助吗？轻松点。

在我们的研究中，我们都经常使用谷歌，如果你能存储你从搜索结果中得到的链接会怎么样？这看起来是一个超级简单的任务。它需要一秒钟来计算出你需要手动执行*的步骤*:访问谷歌，执行搜索，获得结果，保存数据，移动到下一页，如果需要的话进行迭代。

此外,“提取链接”是各种执行网络抓取的包中非常流行的特性，应该有很多文档和教程。更好的是:我们想要构建的脚本对一些同事有帮助(这里我们将使用 **Python** )。

看起来通过实践来学习一些库的新特性是一件很容易的事情。此外，它证明了**编码对人文学科**有所帮助。

太好了，走吧。不会很久的，对吧？剧透:没那么容易(因此有了这篇文章)。

![](img/ac80bdeca5aef9091311055680941b43.png)

[剧透:但我们会给它一个肯定的答案。图片来自 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2069850) 的[戈登·约翰逊](https://pixabay.com/users/GDJ-1086657/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=2069850)

# 基本思想:要求和美丽的声音

项目大纲很容易绘制，并且与我们手工绘制的内容非常接近:

1.  到达搜索引擎；
2.  查询一下；
3.  获取查询结果；
4.  提取所有链接；
5.  拯救他们；
6.  移到下一页；
7.  冲洗并重复。

第四步看起来是最可怕的一步。我们必须检查 html 并得到正确的标签。但这也是乐趣的一部分。好吧，这里潜伏着一些问题，比如“当我用完结果时，我如何发现？”。但是我们可以同意刮掉一组固定的页面，甚至停止第一页。

带着**请求**和 **BeautifulSoup** 库(如果您没有它们，请分别在此处和[处](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-beautiful-soup)获取安装说明)，我们从一些标准导入开始我们的旅程:

```
**import** requests
**from** bs4 **import** BeautifulSoup **as** bs
```

接下来，我们向搜索引擎(Google here)发出请求。为此，我们注意到谷歌上的所有查询都有这样的 URL:'[【https://www.google.com/search?q=】T21](https://www.google.com/search?q=)'+'要查询的内容'。

由于我们不想一直输入我们的查询，我们将硬编码它，即搜索“高飞”。然后，我们检查请求的状态，以确保在访问页面时一切正常。

```
**import** requests
**from** bs4 **import** BeautifulSoup **as** bs*#search for our term with requests*
searchreq = requests.get('https://www.google.com/search?q=Goofy')
 *#ensure it works*
searchreq.raise_for_status()
```

如果你想每次都输入不同的查询(即不要硬编码)，你可以这样做:

```
**import** requests
**from** bs4 **import** BeautifulSoup **as** bs*# ask the user what to search*
query = input('What do you want to search?')*#search for our term with requests*
searchreq = requests.get('https://www.google.com/search?q=' + query)*#ensure it works*
searchreq.raise_for_status()
```

# 获取链接

我们已经完成了草图中的任务 1、2 和 3。现在到了棘手的部分。我们需要分离出谷歌给我们的链接。这意味着我们需要为返回搜索结果的每个页面创建一个 BeautifulSoup 对象(即我们称之为 **searchreq** )并用 BeautifulSoup 处理它们。

我们按照标准做法，把这个对象叫做‘汤’。我们还指定了我们要解析的 html。然后在“结果”中，我们将使用我们的 soup 对象返回我们需要的内容并打印出来。这就是我们添加到代码中的内容:

```
*# creating the Beautiful Soup object to parse html* soup = bs(searchreq.text, 'html.parser')*#apply a find all method on our soup object to get the result*
results = soup.find_all() *#but wait, what to we have to search?**#print them and be happy (if it works)*
print(results)*#SPOILER: it won't*
```

# 抓取链接

为了抓取链接，我们需要告诉 BeautifulSoup 我们需要它提取什么。为了找到答案，我们在一个搜索结果上调用浏览器中的检查器模式(右键单击并选择 inspect on Chrome)。

从那里我们玩一个游戏:

1.  寻找我们需要的物品；
2.  为我们关心的项目提取模式或规律性(即链接)；
3.  抓住他们。

![](img/e9052a751e8d5b94c7fe9a7753613ccc.png)

[对不起，这是必须发生的——来自 [Pexels](https://www.pexels.com/photo/close-up-photo-of-pokemon-pikachu-figurine-1716861/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 的 [Carolina Castilla Arias](https://www.pexels.com/@carocastilla?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 摄影

我们的第一选择可能是类似“http”的东西，但这将捕捉许多额外的东西，以及像是**而不是**搜索结果的链接。

你必须考虑 HTML 模式和标签。如果你看一下(或者疯狂的谷歌一下)，你会发现有一个叫 **div class="r"** 的好东西似乎有你要找的东西。

在花了几分钟阅读 BeautifulSoup 文档页面之后，我们学习了如何从带有: **Soup.select(')的 soup 中获取它们。**。

所以我们把这些放在一起:

```
**import** requests
**from** bs4 **import** BeautifulSoup **as** bs
*# ask the user for a the search*
query = input('What do you want to search?')*#search for our term with requests*
searchreq = requests.get('https://www.google.com/search?q=' + query)*#ensure it works*
searchreq.raise_for_status()*# creating the Beautiful Soup object*
soup = bs(searchreq.text, 'html.parser')*#apply a find all method on our soup object to get the result*
results = soup.select('.r a')print(results)
```

我们已经做好了尝试的准备！

# 卡住了:世界反击了

**[]**

没错，再看一遍。一对方括号。这是我们的产量。

**[]，即**一个空列表。

这就是我们的结果。这令人失望。这是为什么呢？发生什么事了？让我们看看发生了什么事。

我们做的第一件事是尝试打印我们的 soup 对象(如果您有 Ipython，请使用 shell)。一旦我们打印出了 soup 对象，我们就试图搜索我们钟爱的“r”类，也就是我们试图在没有 soup 对象的情况下选择的那个类。

**不在那里！**

这是:*这个世界在报复我们*。在实践中，理论是不够的。所以，嗯，*现在我们可以慌*了。这是怎么回事？这应该是一个简单的任务。

# 出路

我们开始更多地搜索。我在 Twitter 上询问了 Al Sweigart(《T4》一书的作者，用 Python 自动化枯燥的东西，如果你刚开始使用 Python，你应该看看这本书)。事实上，书中的一个程序讨论了获取链接的任务。

艾尔很友好地告诉我，这是谷歌掩盖其结果的常见做法。这就是为什么汤和我们看到的不一样。他简要地提醒我，谷歌之外还有生活，所以在不同的搜索引擎上搜索可能会更好(他建议 duckduckgo)。

这一点*确实*重要(因此有额外的 *Es* )。现在我们知道问题的原因了:**我们在谷歌上看到的 HTML 与我们通过请求**得到的不一样。我们已经有了一个解决方法的提示:试着询问不同的搜索引擎。

我们可以利用这些新知识来建立替代方法。

# 重新思考这个问题

我们有一个新问题。传递我们搜索结果的 HTML 部分不受我们的控制。我们能做什么？我们能像看到的那样得到它吗？有办法解决吗？这要看我们想怎么打了。

# 1.方法:不同的搜索引擎

第一种选择是避开这个问题:我们选择一个不同的搜索引擎。实际上，我们在维基百科上查询搜索引擎名称。然后，我们弄清楚如何查询，并希望链接提取阶段保持不变。

假设如此，这看起来并不是一个昂贵的选择。我们希望其中一个引擎能给我们同样的 html，我们可以检查。

# 2.没门:我们战斗！

我们知道我们想要什么。尽管 HTML 标签不同，但我们知道链接仍然存在。通过 [**正则表达式**](https://docs.python.org/3/howto/regex.html) 提取它们怎么样？这将是困难的，也许是次优选项，但不要冒险再次与 HTML 混淆等。我们可以一劳永逸地解决这个问题。

我们将编写一个正则表达式来提取所有这些内容。我们可以预测我们会:

*   有两个结果或者更多的副本(我们将通过从结果中创建一个**集合**来排除它们)
*   有一些不好的结果(如链接到你的谷歌帐户；或者额外的非搜索相关链接)。

假设你能识别坏链接，比需要更多的链接可能比我们之前得到的[空列表]更好。

# 3.重建:从美味的汤到硒

也许我们可以绕过 HTML 混淆，以不同的方式获得搜索结果。Selenium 是另一个流行的 Python 库，允许我们自动浏览。

Selenium 将为我们打开浏览器，然后我们将查看 HTML。如果失败，我们可以让 Selenium 检查页面，并复制和粘贴检查过的 html。

这似乎在理论上是可行的*。但是需要额外的努力。*

# *4.以不同的方式下载 HTML*

*我们知道混淆发生，但我们不知道如何和何时。也许我们可以试着下载页面并保存在我们的桌面上，然后在那里操作。*

*这听起来既简单又复杂。保存文件，简单。但是，我们需要正确地访问它…请求是正确的方式吗？这需要一些额外的努力。*

# *要做:*

*好的，仍然有一个问题，但是这个领域看起来更清晰了:*

*   *不同的方式需要探索；*
*   *代码应该会增长，并最终到达 GitHub。*

*这有趣吗？随意在 [Linkedin](https://www.linkedin.com/in/guglielmofeis/) 上联系，或者在 [Twitter](https://twitter.com/endecasillabo) 上加入更广泛的对话(期待一些 fantavolley 的挣扎)。*

*(这是之前出现在这里的一篇帖子的改进和评论版本:[http://www . the GUI . eu/blog/scraping-links-from-Google-part-1 . htm](http://www.thegui.eu/blog/scraping-links-from-google-part-1.htm))。*

*这项工作是作为 **CAS 奖学金**的一部分进行的，即*CAS-参见 Rijeka* 。点击此处查看更多关于伙伴关系的信息。*