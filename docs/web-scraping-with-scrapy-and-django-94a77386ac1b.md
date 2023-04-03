# Scrapy 和 Django 的网页抓取。

> 原文：<https://medium.com/analytics-vidhya/web-scraping-with-scrapy-and-django-94a77386ac1b?source=collection_archive---------4----------------------->

![](img/92c936a0035c6ad6eb25893f0d983160.png)

网络抓取是从网站提取数据的过程。它模拟人与网页的交互，在抓取器的帮助下检索想要的信息。

当抓取一个网站，你应该确保他们没有违反其条款和条件。大多数网站都有一个 **robots.txt** 文件，指导抓取者可以或不可以从网站请求哪些页面或文件。这主要是用来避免网站请求过载，这很可能会影响用户。

[**Scrapy**](https://scrapy.org/) 是一个免费开源的网页抓取框架。它能够以快速、简单且可扩展的方式从特定网站提取所需的数据。也是有据可查的。

Scrapy 允许我们定义数据结构，编写数据提取器，并带有内置的 CSS 和 XPath 选择器，我们可以用它们来提取数据。抓取器向网站发出 GET 请求，解析 HTML 响应，并使用外部定义的数据提取器提取目标数据。有 JSON、CSV、XML 等多种输出格式。输出也可以保存到数据库中。

Scrapy 对许多网站限制采取了对策，例如默认情况下随机选择请求之间的时间，这有助于避免被禁止访问。

# 我们正在建造的东西

我们将构建一个应用程序，抓取可供出租或出售的房屋属性。这些属性将从[**realestatedatabase**](https://realestatedatabase.net/FindAHouse/houses-for-rent-in-kampala-uganda.aspx?Title=Houses+for+rent+in+kampala)网站中抓取。

## 假设

本文假设读者对网络抓取、Scrapy 和 Django 有所了解。

## 先决条件

*   Python 3(这个应用程序是用 python 3.7 构建和测试的)。
*   PostgreSQL。
*   代码编辑器。

这个应用程序是建立在 Scrapy 2.0 和 Django 3.0 之上的

**安装依赖关系**

创建一个 PostgreSQL 数据库，稍后将使用它的凭据将其连接到应用程序。

创建一个虚拟环境并激活它。

```
$ python3.7 -m venv venv
$ source venv/bin/activate
```

安装 Django 和 Scrapy。

```
pip install django scrapy
```

安装我们将使用的其他依赖项。

```
pip install scrapy-djangoitem django-phone-field django-environ word2number
```

创建 Django 项目并创建我们的属性应用程序。

```
django-admin startproject main .python manage.py startapp properties
```

创建一个 Scrapy 项目，然后将其添加到`settings.py`中的`INSTALLED_APPS`

```
scrapy startproject scraper
```

创建模型以保存抓取的数据。用这个更新 properties 文件夹中的 models.py 文件。

用数据库凭证更新`settings.py`。将`properties` app 和`phone_field`添加到`INSTALLED_APPS`。这些凭证应该添加到`**.**env` 文件中，以便被保密。

然后运行迁移。

```
python manage.py makemigrations
python manage.py migrate
```

## 项目

该项用于构建蜘蛛解析的数据。前面创建的 Django 模型将用于创建该项目。 [**DjangoItem**](https://github.com/scrapy-plugins/scrapy-djangoitem) 是一个从 Django 模型中获取其字段定义的项目类，这里使用的是之前创建的主应用程序中定义的**属性**模型。更新`scraper/scraper/items.py`。

## 管道。

项目管道是从蜘蛛中提取项目后处理数据的地方。管道执行诸如验证和在数据库中存储项之类的任务。更新`scraper/scraper/pipelines.py`。

## 蜘蛛

定义如何抓取网站，包括如何执行抓取(即跟随链接)以及如何从网页中提取结构化数据。创建一个属性蜘蛛`scraper/spiders/properties_spider.py`。

蜘蛛具有如下定义的特征。

*   **名称**:定义它的字符串。
*   **allowed_domains** :一个可选的字符串列表，包含允许这个蜘蛛爬行的域。
*   **start_urls** :蜘蛛开始爬行的 URL 列表。
*   **规则**:一个(或多个)`[**Rule**](https://docs.scrapy.org/en/latest/topics/spiders.html#scrapy.spiders.Rule)`对象的列表。每一个`[**Rule**](https://docs.scrapy.org/en/latest/topics/spiders.html#scrapy.spiders.Rule)`都定义了爬行站点的特定行为。 **link_extractor** 是一个 [Link Extractor](https://docs.scrapy.org/en/latest/topics/link-extractors.html#topics-link-extractors) 对象，它定义了如何从每一个被抓取的页面中提取链接，而 **call_back** 是一个来自蜘蛛对象的方法，它将用于每一个用指定的链接提取器提取的链接。
*   **Item_loaders** :提供一个方便的机制来填充报废的项目。 **add_css** 方法接收一个存储提取数据的项目字段和一个用于从网页提取数据的 css 选择器。 **load_item** 方法用收集到的数据填充该项并返回，通过管道保存到数据库。

用下面的值更新`scraper/scraper/items.py`，激活`pipelines`，添加`user_agent`，更新`spider_modules`和`newspider_modules`。

创建一个 Django 命令，用于启动蜘蛛爬行。这在 scraper 中初始化了 Django，并需要在 spider 中访问 Django。在`scraper`文件夹中创建一个`management`文件夹。在`management`文件夹下创建一个`commands`文件夹。确保所有新创建的文件都有一个`__init__.py`。

在`commands`文件夹中创建一个`crawl.py`文件，如下图。

这将使用 [CrawlerProcess](https://docs.scrapy.org/en/latest/topics/api.html#scrapy.crawler.CrawlerProcess) 来运行 django 项目的内部。

要运行蜘蛛并将所有属性保存到数据库，请运行以下命令。

```
python manage.py crawl
```

在后续文章中，我们将构建一个页面来显示所有属性。

这是一篇很长的文章，感谢阅读。*干杯*！！。

代码可以在这里找到[。](https://github.com/peterwade153/house-bob)

喜欢这篇文章，请联系 twitter @peterwade153。在[peterwade153@gmail.com](mailto:peterwade153@gmail.com)发邮件。

参考文献。我使用了 Henriette Brand 的文章作为参考。这里可以找到[。](https://blog.theodo.com/2019/01/data-scraping-scrapy-django-integration/)