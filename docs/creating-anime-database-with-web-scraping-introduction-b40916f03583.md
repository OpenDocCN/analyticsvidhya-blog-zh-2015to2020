# 用网络抓取创建动漫数据库-简介

> 原文：<https://medium.com/analytics-vidhya/creating-anime-database-with-web-scraping-introduction-b40916f03583?source=collection_archive---------13----------------------->

[](https://github.com/Tejas-Haritsa-vk/Creating_Anime_Database_With_Web_Scraping) [## tejas-Haritsa-vk/Creating _ Anime _ Database _ With _ Web _ Scraping

### 网络抓取是一种用于从网站中提取大量数据的自动化方法。网站上的数据是…

github.com](https://github.com/Tejas-Haritsa-vk/Creating_Anime_Database_With_Web_Scraping) ![](img/f7f934f17a7aeaf0d533aa1bcfa9c0ea.png)

使用 Python 进行 Web 抓取

# 介绍

**什么是网页抓取？**
网页抓取是一种用于从网站提取大量数据的自动化方法。网站上的数据是非结构化的。
网络抓取有助于收集这些非结构化数据，并将其以结构化形式存储。
用于网络搜集的其他一些词有:网络爬行、网络数据提取、网络收获等。

![](img/0681b798bbbc1612f5d1e4d288708a5f.png)

听完这些之后，每个人脑海中的下一个问题会是

**网页抓取合法吗？总的来说，网络抓取本身并不违法，毕竟你可以出于教育目的抓取一个网站。
根据一般经验，从不允许网络抓取的网站上通过网络抓取获得的任何数据都不能用于商业目的，因为这会违反法律，因此是非法的。
想知道一个网站是否允许网页抓取，可以看看网站的“robots.txt”文件。
你可以通过把“/robots.txt”附加到你想要抓取的 URL 来找到这个文件。
例如，“https://....你的网址在这里…/robots.txt "**

现在这个问题解决了，让我们进入正题。

# 概观

在本帖中，我们将浏览
1。如何检查&分析一个网页进行网页抓取
2。如何入门网页抓取
3？以结构化格式收集非结构化数据

在我们完成这篇文章后，我们将会有一个包含目前网站上所有可用动漫的动漫数据库，包括:
1。动漫片名
2。说明
3。当前/最新一季
4。播出的剧集
5。状态
6。首次播出日期
7。流派
8。sub/Dub 9。系列/电影
10。统一资源定位器

# 要求

1.Python 3
2。Jupyter 笔记本
3。urllib
4。漂亮的一组
5。正则表达式
6。琴弦
7。csv
8。时间

## 装置

***对于康达用户:**
1。康达安装 jupyter 笔记本
2。康达安装美容套装 4

***对于 PIP 用户:**
1。pip 安装 jupyter 笔记本
2。pip 安装 beautifulsoup4

# 内容

1.网页抓取教程. ipynb(笔记本)—如何检查和分析网页进行网页抓取
2。动漫网站数据抓取. ipynb(笔记本)—网络抓取入门&以结构化格式收集非结构化数据

**链接到笔记本:**

[](https://github.com/Tejas-Haritsa-vk/Creating_Anime_Database_With_Web_Scraping) [## tejas-Haritsa-vk/Creating _ Anime _ Database _ With _ Web _ Scraping

### 网络抓取是一种用于从网站中提取大量数据的自动化方法。网站上的数据是…

github.com](https://github.com/Tejas-Haritsa-vk/Creating_Anime_Database_With_Web_Scraping) 

## 链接到第二部分的实践教程:

[](/@tejastejatej/creating-anime-database-with-web-scraping-hand-on-tutorial-6f90e2174be1) [## 用 Web Scraping-Hand on 教程创建动漫数据库

### 使用 python 和 urllib & BeautifulSoup 库创建一个动画数据库

medium.com](/@tejastejatej/creating-anime-database-with-web-scraping-hand-on-tutorial-6f90e2174be1) 

希望这对你有用，如果你喜欢这篇文章，请留下你的赞。