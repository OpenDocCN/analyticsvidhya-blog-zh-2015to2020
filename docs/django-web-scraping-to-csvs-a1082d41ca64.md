# Python:网络抓取到 CSV

> 原文：<https://medium.com/analytics-vidhya/django-web-scraping-to-csvs-a1082d41ca64?source=collection_archive---------34----------------------->

如果您不需要将信息存储在数据库中，网络搜集是从来源(如网站)检索信息的一种很好的方式。假设你有自己的新闻网站，你想展示其他新闻网站的最新头条，这将是一个转向网络抓取的好理由。

幸运的是，有很多方法可以搜索所有语言，包括 Python。Python 有几个不同的库，包括 [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) 和 [Scrapy](https://scrapy.org/) 。Python 还有一个用于导入/导出 CSV 的内置特性。我最近有一个项目需要网络抓取，所以很明显我转向了 Python，这非常简单。

Web 抓取需要对 HTML 有一个基本的了解，因为一旦你从实际的页面中获得数据，你将需要搜索 HTML 元素来获得你所需要的特定信息。但是首先，你需要`pip install beautifulsoup4`和`pip install requests`，这将允许你发送 HTTP 请求到网站。然后像这样导入这些库:

```
from bs4 import BeautifulSoup
import requests
```

然后你就可以开始编码了！你所需要的只是你抓取的网站的 URL。请注意，一些网站有更高的安全性，并防止任何形式的网络抓取。

```
page = requests.get(your_url_here)
soup = BeautifulSoup(page.content, 'html-parser')
```

您将需要指定您需要的解析类型，因为 BeautifulSoup 可以适应各种类型的解析。因为我们正在处理 HTML 元素，所以一个 *html 解析器*是合适的。

这就是乐趣所在(乐趣是相对的)。这时你需要检查你的控制台，并找出你需要的元素。一旦找到元素，用 BeautifulSoup 找到它们就非常容易了——可以使用元素标签和/或类名。例如:

```
div = soup.find('div', class_='class_name')
```

然后你甚至可以添加一个`.text`来获得一个元素的内部文本:

```
text = soup.find('div', class_='class_name').text
```

我参与的项目也需要将收集到的数据导出到 CSV 文件中。这就是 Python 内置 CSV 特性的由来。假设您从那个`.find`方法得到的结果产生了多个 div，您需要遍历这些 div，然后将每个 div 中的文本导出到一个 CSV 文件中。首先，在页面顶部，您需要导入该库:

```
import csv
```

然后你就可以开始用它编码了！因此，您可以遍历多个 div，并打开一个包含这些结果的新 CSV:

```
all_text = []
for div in divs:
  get_text = div.text
  text_dict = {'Text': get_text}
  all_text.append(text_dict)open('text.csv', 'w') as out_file:
  headers: [
    'Text'
  ]
  writer = csv.DictWriter(out_file, fieldnames=headers)
  writer.writeheader()
  for text in all_text:
    writer.writerow(text)
```

这段代码将自动在您的目录中打开一个新的`.csv`文件，其中包含您所请求的数据。`'text.csv'`是你的 CSV 文件的名字，想怎么叫都行。您可以随意调用头，借助代码的力量，如果头与原始字典的键匹配，您不需要将整个字典传递给`writerow()`。

TL；DR，web 抓取数据并通过 Python 导出到 CSV 非常简单。医生也很棒。