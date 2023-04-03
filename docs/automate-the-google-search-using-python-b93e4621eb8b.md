# 使用 Python 实现谷歌搜索的自动化

> 原文：<https://medium.com/analytics-vidhya/automate-the-google-search-using-python-b93e4621eb8b?source=collection_archive---------3----------------------->

![](img/e533938ea61082666c251510d95ee291.png)

每当我们需要帮助时，我们从谷歌那里获得简单的帮助。谷歌每天有 54 亿次搜索，是目前全球最大的搜索引擎。但有时候，手动搜索是相当令人生畏的。如果我们可以使用 python 来自动化整个过程。在这个故事中，我会帮助你。**讲完这个故事，你就会很好的利用 Python 实现谷歌搜索全过程的自动化。**

为此，我们将从名为**谷歌搜索**的 Python 库获得帮助。它是免费的，简单易用。要安装该库，请在命令提示符或 Anaconda 提示符下键入:

```
pip install google
```

安装谷歌后，只需从谷歌搜索中导入搜索模块

```
from googlesearch import search
```

搜索所做的是，它会将你的查询和搜索过谷歌，找出相关的网址，并将其保存在一个 python 列表中。要获得保存的结果，只需迭代列表中的对象并获得 URL。

```
# as an example:query = "iPhone"for i in search (query,  tld='com', lang='en', tbs='0', safe='off', num=2, start=0, stop=2, domains=None, pause=2.0, tpe='', country='', extra_params=None, user_agent=None):
 print(i)
```

此处 tld:域名，即 google.in 或谷歌。in，lang:搜索语言，num:每页结果数，start:检索的第一个结果，stop:检索的最后一个结果，依此类推。在你的 Jupyter 笔记本中，如果你点击 **shift+Tab** 并点击加号，你将获得该模块的所有信息。

让我们创建一个关于股票市场公司分析的简单搜索算法，它会让你对现在的公司状况有一个快速的总体了解

```
# importing the module
from googlesearch import search# stored queries in a list
query_list = ["News","Share price forecast","Technical Analysis"]# save the company name in a variable
company_name = input("Please provide the stock name:")# iterate through different keywords, search and print
for j in query_list:
   for i in search(company_name+j,  tld='com', lang='en', num=1, 
   start=0, stop=1, pause=2.0):
      print (i)
```

在这里，我创建了这样的搜索关键字“**公司名称+查询**”。它接受第一个查询，将公司名称添加到其中，并使用 google 搜索最新的搜索结果。当它完成搜索时，它打印出 URL。

为了简单起见，我只取了搜索中的第一个 URL。如果你愿意，你可以带更多的网址来搜索。**请使用 collab**:[*https://colab . research . Google . com/drive/17 zotfibopojyxantofewucuhtpp3 o 4 ay*](https://colab.research.google.com/drive/17ZOtFIBoPoJYxantOFEWUCUHtpP3O4ay)运行

当前算法仅在默认模式下搜索。如果你喜欢按类别搜索，比如新闻、图片、视频等。等等，你可以用 tpe: parameter 轻松指定。只要选择正确的类别，你就可以了。下面我运行相同的代码，但只使用谷歌新闻搜索

```
# only added tpe="nws" for news only
for j in query_list:
    for i in search(company_name+j,  tld='com', lang='en', num=1,
    start=0, stop=1, pause=2.0, tpe="nws"):
        print (i)
```

同样，对于视频，添加 tpe = " vde 图像，tpe = " isch 书籍，tpe="bks "等。

如果你想用一个更广泛的例子来说明你可以用谷歌搜索做什么，这里有一个例子。

```
from googlesearch import searchimport requests
from lxml.html import fromstring# Link URL Title retriever usin request and formstring
def Link_title(URL):
  x = requests.get(URL)
  tree = fromstring(x.content)
  return tree.findtext('.//title')company_name = input("Please provide the company name:")query = int(input("Please give the appropriate value. 1 for Fundamental Analysis, 2 for News, 3 for Technical Analysis & 4 for Share Price Forecast:"))if query == 1:
  print (company_name+" "+"Fundamental Analysis:")
  print (" ")
  for i in search(company_name,  tld='com', lang='en', num=1, start=0, stop=1, domains=['https://www.tickertape.in/'], pause=2.0):
    print ("\t"+i)elif query == 2:
  print (company_name+" "+"News:")
  print (" ")
  for i in search(company_name+ 'News',  tld='com', lang='en', num=3, start=0, stop=3, pause=2.0, tpe='nws'):
    print ("\t"+"#"+" "+Link_title(i))
    print("\t"+i)
    print(" ")elif query == 3:
  print (company_name+" "+"Technical Analysis:")
  print (" ")
  for i in search(company_name+ 'Technical Analysis',  tld='com', lang='en', num=3, start=0, stop=3, pause=2.0):
    print ("\t"+"#"+" "+Link_title(i))
    print("\t"+i)
    print(" ")else:
  print (company_name+" "+"Share Price Forecast:")
  print (" ")
  for i in search(company_name+ 'share price forecast',  tld='com', lang='en', num=3, start=0, stop=3, pause=2.0):
    print ("\t"+"#"+" "+Link_title(i))
    print("\t"+i)
    print(" ")
```

这里我使用了与上面例子相同的逻辑。我在这里添加了两件事，一是创建函数 Link_title()来显示来自搜索查询的特定链接的标题，二是为了进行基本分析，我使用 domain 选项指定了相关的 URL。

在这个例子中，为了分析一支股票，我考虑了四个类别:基本面分析、新闻、技术分析、股价预测。为了记住这一点，我使用了 if Else 条件来展示适当的类别 URL。所以每当你运行代码，它会问你股票名称和搜索类别，然后它会显示正确的网址和标题。**请使用下面的 collab 运行代码:**

[T3【https://colab . research . Google . com/drive/1 w6 dxfuleahi 7m Hg 2 zslw 7 qfkl-xA6vO](https://colab.research.google.com/drive/1w6dXfUlEahI7M7HG2zSlW7QfkL-xA6vO)

股票分析只是你可以使用它的许多应用之一。使用谷歌，它可以是你想要的任何东西。我希望你喜欢我的故事。你可以在这里使用 **Github** 链接访问代码:[*https://Github . com/soumyabrataroy/Google-stock-search-using-python . git*](https://github.com/soumyabrataroy/Google-stock-search-using-Python.git)

请在下面的评论中告诉我你的想法。可以在 Twitter 上找到我:@ sam1ceagain

原发布于 LinkedIn:https://www . LinkedIn . com/pulse/automate-Google-search-using-python-soumyabrata-Roy/