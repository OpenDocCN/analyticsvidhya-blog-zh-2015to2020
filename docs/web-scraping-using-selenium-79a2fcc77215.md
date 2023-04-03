# 使用 Python Selenium 进行 Web 抓取和登录

> 原文：<https://medium.com/analytics-vidhya/web-scraping-using-selenium-79a2fcc77215?source=collection_archive---------4----------------------->

![](img/683d07a5a41520ad99c172a47cddf6be.png)

有一个网页抓取问题，当网站必须先登录？

我们可以用硒来解决这个问题。基本上，selenium 用于自动测试 web 验证，但也可以用于抓取，因为它可以由脚本自动控制，可以轻松地处理 javascript、DOM 或复杂的 html 标签

例如，我们试图从需要先登录的网站上删除新闻，如 www.wsj.com 的[或 www.barrons.com 的](https://www.barrons.com/real-time?mod=hp_LATEST&mod=hp_LATEST)[的](https://www.barrons.com/real-time?mod=hp_LATEST&mod=hp_LATEST)

我们做的第一件事是**安装库，**包括*selenium python* 库， *webdriver manager* 库并在你的文件中导入几个 selenium 函数

图书馆

**为登录**创建您的函数/类 **，代码包括:**

*   放上网址
*   设置 web 驱动程序选项(例如，窗口大小、headless 等。)和
*   使用您的用户名和密码登录

通过 Selenium 登录网站

成功登录后，我们可以**继续代码获取新闻**。我们可以选择我们需要的信息(如标题、文章、日期等)并**存储到 csv**

使用 Selenium 的网页抓取

有时，我们仍然无法从网站上获取数据，因为*验证码*或其他原因。因此，如果发生这种情况，我们可以通过一些方法来防止它，如用户代理或减缓脚本的执行

对于用户代理，我们可以使用 ***fake_useragent 库*** 并在 web 驱动选项中添加一个随机代理。而为了减缓脚本的执行速度，我们可以使用***【time . sleep(秒)***

然而，使用 selenium 进行 web 抓取仍然很棘手，但至少这是另一种从网站获取数据的工具，并且可以很容易地登录到网站。

** *此代码改编自* [*此处*](https://github.com/philippe-heitzmann/WSJ_Web_Scraping_Project-NYCDSA-Project2) *，更多信息请查看* [*此处*](https://nycdatascience.com/blog/student-works/scraping-wall-street-journal-article-data-to-measure-online-reader-engagement-an-nlp-analysis/)