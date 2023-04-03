# 用 Selenium 和 Python 抓取亚马逊结果。

> 原文：<https://medium.com/analytics-vidhya/scraping-amazon-results-with-selenium-and-python-547fc6be8bfa?source=collection_archive---------1----------------------->

![](img/2711b17efa18f51517433f91256d2bc6.png)

卢卡·布拉沃在 [Unsplash](https://unsplash.com/s/photos/web-scraper?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

当我第一次用 BeautifulSoup4 开始网页抓取时，我发现最难跨越的障碍是分页。从静态页面获取元素似乎相当简单——但是如果我想要的数据不在我加载到脚本中的初始页面上呢？在这个项目中，我们将尝试使用 Selenium 在亚马逊结果页面中循环分页，并且…