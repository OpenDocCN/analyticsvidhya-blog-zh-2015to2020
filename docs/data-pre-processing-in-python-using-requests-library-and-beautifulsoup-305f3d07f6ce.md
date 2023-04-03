# 利用请求库和 BeautifulSoup 在 Python 中进行数据预处理

> 原文：<https://medium.com/analytics-vidhya/data-pre-processing-in-python-using-requests-library-and-beautifulsoup-305f3d07f6ce?source=collection_archive---------22----------------------->

![](img/4dc86a8687cafeea16baf91bf2b9e86a.png)

# ***我们在这里要做什么？***

> 我将从 **goodreads** 网站(网址为“[https://www.goodreads.com/quotes](https://www.goodreads.com/quotes)”)读取内容(本质上是引用)，然后对其进行数据预处理。目标是最终将检索到的数据以有组织的格式(提取的报价以及作者姓名&报价上的赞数)保存在数据帧中，然后保存在. csv 文件中。格式非常简单，因为我们将在 dataframe &中有三列“Authorname”、“Likes”和“Quote ”,然后将其写入 csv 文件。

— — — — — — — — — — — — — — — — — — — — — — — — — — — — -

代码:
*(*考虑你那端的缩进)*

*#这里，我们将导入 Python 的库用于查询一个网站的*
**导入请求**

*#我们将在这里导入熊猫来创建一个数据帧最终* **导入熊猫作为 PD**

*#在这里，我们将导入 Beautiful soup 函数来解析从 bs4 导入 Beautiful soup*
**网站返回的数据#从**

*#让我们在这里指定 url，内容(引号)必须从这里读取*
**引号= "**[**https://www.goodreads.com/quotes**](https://www.goodreads.com/quotes)**"**

*#在这里，我们将查询网站/url，并将 html 返回到变量' page '*
**page = requests . get(quotes)**

*#现在将获取网站的内容作为文档*
**soup = beautiful ul soup(page . text，' html.parser')**

*#这将会找到文档中的所有行，它们的类是“quote details”#其中包含所有引号的详细信息— Authorname，quote*
**quote _ text = soup . find _ all(class _ = " quote details ")**

*#我们现在将创建一个空数据帧来存储检索到的细节*
**df1=pd。DataFrame()**

现在让我们运行一个循环来分别获取每个类别——author name，quote #并打印它们。
**for text 1 in quote _ text:**

*#找到类为“authorOrTitle”时取作者名。当发现类为“quoteText”时，获取报价。当它发现类是“正确的”时，获取喜欢。*
**likes = text 1 . find(class _ = " right ")
num _ like = likes . find(' a ')
for num _ like 中的 num _ like new:
author _ name = text 1 . find _ all(class _ = " author orttitle ")
quote text = text 1 . find _ all(class _ = " quote text ")
for quote text 中的屈:
for author _ name 中的人名:** 替换(' \n '，'')，
"Quote" : qu.contents[0]。替换(' \n '，'')，
“喜欢”:num_likenew }，
ignore_index=True)

**print(df1 . head())**
*#将数据存储到一个 csv 文件*
**df1.to_csv(r "提供你的路径。csv 文件此处")**

**确保 csv 文件路径以。csv(例如 C:\Users\my\filename.csv)*