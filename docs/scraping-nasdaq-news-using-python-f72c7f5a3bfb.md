# 使用 Python 抓取纳斯达克新闻

> 原文：<https://medium.com/analytics-vidhya/scraping-nasdaq-news-using-python-f72c7f5a3bfb?source=collection_archive---------0----------------------->

![](img/b636776bec2dfa89310217085e7cc843.png)

股票交易是当今世界最复杂的动态之一。在今天的时间，多种算法和研究已经产生，以了解股票交易的复杂性。有越来越多的努力来理解股票交易的系统动力学，以预测股票价格的突发行为。

为了充分预测股票价格，需要访问股票价格的历史数据。大多数情况下，你会专注于一只股票，这是一个预测值。为了获得股票价格的历史数据，你可以使用[数据服务提供商](https://blog.datahut.co/datahut-vs-import-io-which-alternative-is-better-for-web-scraping/)或者你可以使用简单的[网络抓取器](https://blog.datahut.co/what-web-scraping-can-and-cant-do-for-you/)来完成这项工作。这项任务可以通过抓取提供股票价格数据的网站来完成。你可以继续抓取纳斯达克新闻网站或[抓取雅虎财经网站](https://blog.datahut.co/scraping-yahoo-finance-data-using-python/)的股票价格数据！

在本文中，我们将着重于搜集纳斯达克新闻网站的股票价格数据。我们将一步一步地演示 web 报废的实现，这样你就可以很容易地理解它。在[扒纳斯达克新闻网站](https://blog.datahut.co/is-web-data-scraping-legal/)之前，让我们先了解下一节的纳斯达克新闻。

# 什么是纳斯达克新闻？

纳斯达克股票市场是美国股票的交易所。它是世界第二大市值证券交易所。纳斯达克公司(Nasdaq Inc .)拥有交易所平台，该平台还拥有纳斯达克北欧和纳斯达克波罗的海股票市场网络，以及几家美国股票和期权交易所。

纳斯达克是一个全球性的证券交易平台。纳斯达克，由全国证券交易商协会(NASD)发起，使交易商能够在一个计算机化、快速和透明的系统上进行证券交易。纳斯达克新闻包括各种股票、商品和指数的每日信息。此外，它还提供所有对股票分析师、商业人士和普通交易者有用的金融和股票市场新闻。

# 用于股票价格预测的 Web 抓取

股票价格预测是 21 世纪最热门的预测任务之一。不同的投资机构都在竞相开发自己的算法来准确预测股价。无论这种情况有多复杂，都有很多底层算法可以帮助你做到这一点。

所有这些算法的首要要求是股票价格数据的可用性。股票数据通常可从不同的数据供应商处获得，但这是有成本的。如果你是一个独立的研究人员，想在股票价格预测方面有所实践，有一种方法可以获得这些股票数据。在这种情况下，网络刮擦来拯救你。使用网络抓取，你可以从不同的股票媒体平台获得股票数据，如纳斯达克新闻，雅虎财经等。手头有了股票数据，您可以在分析股票市场时执行以下任务

1.  **股价预测**
    [网上交易](https://www.thebalance.com/how-to-start-trading-stocks-step-by-step-1031363)涉及通过网上平台进行股票交易。 ***网上交易门户为股票、共同基金、商品等不同金融工具的交易提供便利*** 。在网上股票交易中，一只股票的所有者虚拟地会见不同的买家，并把股票卖给买家。只有当买方和卖方就交换价格达成一致时，出售才会发生。此外，这些价格取决于市场，由雅虎财经提供。此外，股票交易机构可以利用雅虎财经数据来记录不断变化的股票价格和市场趋势。这种分析将有助于金融和投资公司预测市场，买卖股票以获取最大利润。

**2。股市情绪分析**

组织可以对商业和金融领域的博客、新闻、推文和社交媒体帖子进行股票**市场情绪分析**
[***以分析市场趋势。此外，抓取雅虎财经将有助于他们为自然语言处理算法收集数据，以识别市场情绪。通过这种方法，人们可以跟踪对某一特定产品、股票、商品或货币的情绪，并做出正确的投资决定***](https://blog.datahut.co/web-scraping-social-media-e-commerce/)

**3。股票研究**

股票研究是指分析公司的财务数据，对其进行分析，并确定股票买卖的建议。*股权研究的主要目的是为投资者提供关于购买、持有或出售特定投资的财务分析报告和建议*。此外，银行和金融投资机构经常通过提供及时、高质量的信息和分析，为他们的投资和销售&交易客户使用股票研究。

**4。合规性**

商业和金融投资工作是高风险的工作。许多投资决策直接取决于政府的贸易计划和政策。因此，**监管合规** [*跟踪政府网站和其他官方论坛以获取任何与交易相关的政策变化*是至关重要的。](https://blog.datahut.co/gdpr-and-the-consent-epoch-how-to-convince-customers-to-share-data/)主要来说，风险分析师应该抓取新闻媒体和政府网站，以获取与其业务直接相关的事件和决策的实时行动。

# 我们的目标

在这篇博客中，我们的目标是学习搜集纳斯达克新闻的过程。我们将搜集最活跃股票和指数的数据。我们将使用 python 来实现我们的 web scraper。此外，我们将使用 BeautifulSoup 库抓取纳斯达克新闻。BeautifulSoup 是 python 中提供的一个简单的抓取库。

如果你完全不熟悉网页抓取的过程，我们将在这个博客中一步一步地介绍。因此，最后，你将能够很容易地理解整个刮管道。在直接跳到抓取纳斯达克新闻的实现之前，让我们看一下我们将要遵循的抓取管道。

# 抓取纳斯达克新闻的管道

要实现股票价格数据的纳斯达克新闻抓取，我们需要遵循几个步骤一步一步的程序，我们将完成！首先，我们将设置目标 URL，并从目标 URL 下载所有可用的数据。之后，我们的主要任务是在下载的数据中搜索我们需要的信息。

这更像是一个字符串匹配过程，我们在数据中寻找特定的模式，并使用这些模式提取它们。在提取数据后，我们将尝试可视化这些数据以便更好地理解，并与我们一起保存。

# 阶段 1:确定刮削参数

web 抓取中最重要的任务之一是分析目标网页的 HTML 结构。这里，我们希望找到数据的 HTML 结构中的模式。这些模式是从网页中提取数据的基本要素。我们将寻找一些重复出现的 HTML 结构或者 HTML 标签和 id。

让我们试着在我们的案例中找到一些模式。下面是我们将要抓取的目标股票价格表的 HTML 片段。

![](img/fbb552cd4dba4da66cbf2c3cb2691428.png)

在最活跃的股票页面上，您可以使用左键单击并检查页面上的元素。之后，您可以使用悬停功能来查找目标股票表的 HTML 代码。这里，您可以在图像中看到，股票表被映射到代码中名为“genTable”的类。这给了我们一个钩子，可以在抓取的时候在 HTML 代码中查找整个表。因此，这里我们的方法是首先查找指定的表。找到表格后，我们将逐一迭代表格行，并逐一提取股票数据。

# 阶段 2:抓取纳斯达克新闻的 Python 实现

在这一节中，我们将从抓取纳斯达克股票价格新闻的实现开始。我们在这里使用 python 来实现 web scraper。我们的首要任务是首先导入所有的库。

```
import requests from bs4 import BeautifulSoup import csv import pandas as pd
```

导入所有库之后，我们需要设置目标 URL。一旦我们设置了目标 URL，我们的代码将解析整个网页并将所有的 HTML 内容存储在一个变量中。之后，我们使用 BeautifulSoup 库提供的内置函数在 HTML 代码中搜索我们需要的信息。您可以在下面找到完整的实现！

```
mostActiveStocksUrl = &quot;https://www.nasdaq.com/markets/most-active.aspx&quot; r= requests.get(mostActiveStocksUrl) data=r.text soup=BeautifulSoup(data) table=soup.find_all('div', attrs={&quot;class&quot;:&quot;genTable&quot;}) all_rows=table[1].find_all('tr') symbols=[] names=[] last_sales=[] change_nets=[] share_volumes=[] for row in all_rows: cols=row.find_all('td') if(len(cols)): names.append(cols[1].text) last_sales.append(cols[2].text) change_nets.append(cols[3].text) share_volumes.append(cols[4].text) data=pd.DataFrame({&quot;Names&quot;:names,&quot;Last Sale&quot;: last_sales,&quot;Chnange Net&quot;: change_nets,&quot;Share Volume&quot;: share_volumes})
```

# 阶段 3:可视化结果

在这一阶段，我们将在一个表格中组织收集的数据，并查看存储的结果。我们使用 python 中可用的 pandas 库从收集的信息中构建一个简单的数据框架。实现在下面！

```
data=pd.DataFrame({&quot;Names&quot;:names,&quot;Last Sale&quot;: last_sales,&quot;Chnange Net&quot;: change_nets,&quot;Share Volume&quot;: share_volumes})
```

![](img/41840a6ff1bd807640324069d98ab2cb.png)

在本文中，我们了解了如何使用 python 简单地从纳斯达克新闻中获取股票市场数据 。此外，有关股票、商品和货币的数据也是通过抓取纳斯达克新闻网站收集的。 **Beautiful soup 是 python 中一个简单而强大的抓取库**，它让抓取纳斯达克新闻网站的任务变得非常简单。

此外，金融机构通过抓取纳斯达克新闻网站收集的数据来预测股票价格或预测市场趋势，以生成优化的投资计划。除了金融机构，不同垂直行业的许多行业都利用了网络抓取的优势。**将 Datahut 作为您的网络抓取合作伙伴，开始利用网络抓取为您的组织带来的好处。**