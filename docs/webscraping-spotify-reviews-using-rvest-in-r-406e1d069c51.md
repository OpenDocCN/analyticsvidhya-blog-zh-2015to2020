# 在 R 中使用“rvest”抓取 Spotify 评论

> 原文：<https://medium.com/analytics-vidhya/webscraping-spotify-reviews-using-rvest-in-r-406e1d069c51?source=collection_archive---------1----------------------->

![](img/a9df137eeaca797f65acf6b7a1238afa.png)

R 中的“rvest”库是我工具箱中最新的抓取网站的工具！为了测试它，我决定抓取“Trustpilot”网站，这是一个供评论者评论服务和其他网站的流行平台。在这个搜集练习中，我计划提取评论者对 Spotify 的评分和评论。让我们看看用户对 Spotify 音乐有什么看法！

![](img/cff49e73d94699b3bacdcef9ad279162.png)

带有 Spotify 评论的 Trustpilot 网站

我们从加载所有必要的库开始。

![](img/f5042927f775964a3265c388908de59f.png)

接下来，我们创建一个变量来存储源 url 的 html，在我们的例子中是 Trustpilot 网站上的 Spotify 页面。

![](img/19b5d31afe033e058a2305e5ab019a84.png)

我们的相关 url 存储在变量“content1”中。稍后当我们有一个函数来输入这个变量时，我们会用到它。

我们向下滚动，看到总共有 15 页包含评论。

![](img/7c1d5011461e7268f79b05e9625227f0.png)

所以我们写了一个函数来获取所有的 15 页。在代码中，html_nodes(' . ')表示类标签，html_text()表示我们希望文本不在 html 中。

![](img/5179ebc7f78f3c140aeac53f28ca1dcf.png)

接下来，我们编写 4 个小函数来获取评论、评论者的姓名、星级和评论的日期时间。让我们一个接一个地去做。

1.  函数来提取评论的正文

![](img/906248501cd55bb00068aa13761aeb18.png)

2.提取审阅者姓名的函数

![](img/024a4079c57f413a7407b5fe5fbe886b.png)

3.函数提取每个评论者的星级

![](img/adba3d1d5ca89667406c566e143b273b.png)

4.函数来提取每个评论的日期时间

![](img/02a90099bdb22924a2dc30aa4bcfa036.png)

我们现在将上述四个函数合并成一个通用函数。

![](img/21f89e17cc4db7966a8efd51fcbfa332.png)

为了方便起见，我们缩短了上面的函数。

![](img/14ec1d13ccfc59acd107c2db9558484b.png)

这个函数只抓取一个 url。我们创建了一个组合函数来从多个页面(在我们的例子中是 15 个页面)抓取 URL。

![](img/3c60b1d4aa3bd9efceecdc20d169b630.png)

现在，将我们相关的 url 和公司名称输入到上面创建的函数中。记住，我们的相关 url 位于变量“content1”中！

![](img/af95bee8845131918c24c2ff0dae290d.png)

我们将 spotify 表格转换为. tsv 文件，因为. tsv 文件存储了一个数据表，其中的数据列由制表符分隔。此外，这些文件可以导出的电子表格程序，这使得数据可以查看一个基本的文本编辑器或文字处理程序。我们将把 spotify 表格导出到一个 excel 文件中，以便将所有评论汇总在一起。

在导出之前，我们将显示输出。

![](img/677b74badfeb8b82afd77c312fb5f14c.png)

我们得到一张 5 乘 4 的桌子。

![](img/949d4333051bd56b79ec2d9a3416ad6d.png)

现在，我们将整个表格导出到 excel 文件。

![](img/63b5c9cff70f438a5c8043f7a4b47b21.png)

所以我们终于导出了文件！让我们看看它是什么样子的:)

![](img/660f1009b8ac1fd01e76c05d96dd9751.png)

…

…

…

![](img/309d656c9f1be3f4785b6c28452b699a.png)

哇哦。我们已经为 Spotify 收集了 300 条用户评论:)

如果你喜欢读这篇文章，请给我掌声！

源代码存放在我的 GitHub 上。

[](https://github.com/tanmayeewaghmare/Web-scraping-using-R/blob/master/Spotify%20user%20ratings.R) [## tanmayeewaghmare/Web-scraping-using-R

### 在 GitHub 上创建一个帐户，为 tanmayeewaghmare/Web-scraping-using-R 开发做出贡献。

github.com](https://github.com/tanmayeewaghmare/Web-scraping-using-R/blob/master/Spotify%20user%20ratings.R)