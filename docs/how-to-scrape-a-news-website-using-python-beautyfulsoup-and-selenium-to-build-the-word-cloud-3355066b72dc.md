# 如何使用 Python、BeautyfulSoup 和 Selenium 构建单词云来构建新闻网站

> 原文：<https://medium.com/analytics-vidhya/how-to-scrape-a-news-website-using-python-beautyfulsoup-and-selenium-to-build-the-word-cloud-3355066b72dc?source=collection_archive---------13----------------------->

## 通过使用词云实现更好的可视化

![](img/cbf0f04ba3d902e324244702922cf234.png)

从 2020 年初到现在，香港报纸上最常出现的新闻是什么？香港社会运动，2019 冠状病毒还是中美贸易战？我在香港刮了一份知名报纸，希望用一个好的可视化方法~ **字云**得到我想要的答案。

**什么是网页抓取？**

网络抓取工具专门用于从网站上提取信息。它们也被称为网络收集工具或网络数据提取工具。

**为什么要做网页抓取？**

网络抓取工具可以在各种情况下用于各种目的。例如:

1.  收集市场研究数据
2.  收集股票市场信息
3.  收集联系信息
4.  收集数据以下载供离线阅读或存储
5.  跟踪多个市场的价格等。

**如何在 Python 中刮？**

用 Python 抓取一个网站非常容易，尤其是在 BeautifulSoup 和 Selenium 库的帮助下。Beautiful Soup 是一个 Python 库模块，允许开发人员通过编写少量代码，快速解析网页 HTML 代码并从中提取有用的数据，减少开发时间，加快网页抓取的编程速度。Selenium 是一个自动化测试网页的工具，可以通过它提供的一些方法自动操作浏览器，可以完全模拟真人的操作。

在抓取一个网站之前，必须做一些准备工作，在 Jupiter 笔记本中输入命令，它将安装以下库。

```
!pip install BeautifulSoup4
!pip install selenium
```

**什么是词云，我们为什么需要它？**

Word Cloud 是 Python 中非常好的第三方 word cloud 可视化库。单词云是在文本中更频繁出现的关键词的可视化显示。WordCloud 会过滤掉大量低频低质量的文字信息，让受众一眼就能明白文字的主旨。

**什么是分词？**

分词是将连续的词序列按照一定的规范重新组合成词序列的过程。Jieba 是一个比较好的中文分词库，因为中文通常包含整个句子，所以我们需要使用 Jieba 来辅助分词的工作。

![](img/2218b1e03ecc10a326d8ae50b5cb7f1e.png)

在生成 word 云映像之前，必须做一些准备工作，在 Jupiter 笔记本中输入命令，它将安装以下库。

```
!pip install wordcloud
!pip install jieba
```

**#加载库**

```
#! /usr/bin/env python
# -*- encoding UTF-8 -*-import os
import sys
from importlib import reload
reload(sys)if sys.version[0] == '2':
    sys.setdefaultencoding("utf-8")

    import parse
    import urllib2
else:
    import urllib.parse
    from urllib.request import urlopenimport re
import jiebaimport pandas as pd
import numpy as np#from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptionsfrom wordcloud import WordCloud, STOPWORDS, ImageColorGeneratorfrom PIL import Imageimport matplotlib.pyplot as plt
%matplotlib inlinejieba.enable_paddle()
```

**#清理 CSS、JavaScript 和 HTML 标签**

```
def cleanHTML(html):
    for script in html(["script", "style"]): # remove all javascript and stylesheet code
        script.extract()
    # get text
    text = html.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text
```

**#查找当日主要焦点新闻链接**

```
def getNewsLink(news_date):
    try:
        options = ChromeOptions()
        options.add_argument('headless')
        driver = webdriver.Chrome(options=options) url = '[https://orientaldaily.on.cc/cnt/news/'](https://orientaldaily.on.cc/cnt/news/') + str(news_date) + '/mobile/index.html' driver.get(url)
        driver.implicitly_wait(30) html_source = (driver.page_source.encode('utf-8')) driver.quit() soup = BeautifulSoup(html_source, 'html.parser')
        news = soup.find('div', attrs={'id':'swipe'})
        main_focus = soup.find('div', attrs={'class':'main-focus-container'})
        main_focus_link = '[https://orientaldaily.on.cc'](https://orientaldaily.on.cc') + main_focus.find('a', href = re.compile(r'[/]([a-z]|[A-Z])\w+')).attrs['href'] return (main_focus_link)
    except:
        return 0
```

**#数据收集**

```
def getNews(news_url):
    options = ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome(options=options)

    driver.get(news_url)
    driver.implicitly_wait(30)

    html_source = (driver.page_source.encode('utf-8'))

    driver.quit()

    soup = BeautifulSoup(html_source, 'html.parser')
    paragraph = soup.find_all('div', attrs={'class':'paragraph'})
    paragraph_list = []

    for sub_paragraph in paragraph:
        clean_sub_paragraph = cleanHTML(sub_paragraph)
        paragraph_list.append(clean_sub_paragraph)

    full_paragraph_list = [e for e in paragraph_list if e]

    if (len(full_paragraph_list) > 0):
        f=open("news.txt", "a+")

        for i in range(len(full_paragraph_list)):

            for line in full_paragraph_list[i].splitlines():
                clean_paragraph = cleanText(line)

                f.write(clean_paragraph)

        f.write('\n\n')
        f.close()
```

**#画字云**

```
def draw_word_cloud(text, images_name, plt_title):
    images_path = images_name
    images = Image.open(images_path)

    #create a write mask 
    images_mask = Image.new("RGB", images.size, (255,255,255))
    images_mask.paste(images, images)
    images_mask = np.array(images_mask)

    color = ImageColorGenerator(images_mask)

    #Chinese need to use another font, download it from [https://www.freechinesefont.com/](https://www.freechinesefont.com/)
    font_path = 'HanyiSentyCandy.ttf'

    #create wordcloud ~ 
    wc = WordCloud(font_path=font_path, max_font_size=250, max_words=1000, mask=images_mask, \
                   margin=5, background_color="black").generate_from_text(text)
    wc.recolor(color_func = color, random_state = 7)

    #Save the image
    wc.to_file("news.png")

    plt.rcParams["figure.figsize"] = (16, 12)

    plt.title(plt_title)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
```

**#主逻辑:**

```
if __name__ == "__main__":
    start_date = input("Enter the start date (yyyymmdd): ")
    end_date = input("Enter the end date (yyyymmdd): ")

    if ((start_date != "") and (end_date != "")):
        daterange = pd.date_range(start_date, end_date)

        #Download the main focus news
        for news_date in daterange:
            print ('Downloading ' + str(news_date) + ' news...')
            single_news_date = dateConvert(news_date)
            getNews(getNewsLink(single_news_date))

        source_text = open('news.txt', 'r',encoding= 'UTF-8').read()
        tokens = ' '.join(jieba.cut_for_search(source_text))

        title = str(start_date) + ' - ' + str(end_date) + ' Main Focus News on on.cc'

        draw_word_cloud(tokens, 'hongkongpng.png', title)
```

**#未来改善**

1.  添加停用词的处理
2.  使用不同的分词库，如 thulac、FoolNLTK、HanLP、nlpir 和 ltp。

感谢阅读！如果你喜欢这篇文章，请通过鼓掌来感谢你的支持(👏🏼)按钮，或者通过共享这篇文章让其他人可以找到它。

最后希望你能学会刮痧的技巧。你也可以在 [GitHub](https://github.com/kindersham/100DaysDS/tree/master/NewsWordCloud) 库上找到完整的项目。

**参考文献**

[](/the-artificial-impostor/nlp-four-ways-to-tokenize-chinese-documents-f349eb6ba3c3) [## [NLP]中文文档的四种分词方法

### 有了一些支持子词分割技术的经验证据

medium.com](/the-artificial-impostor/nlp-four-ways-to-tokenize-chinese-documents-f349eb6ba3c3) [](https://amueller.github.io/word_cloud/index.html) [## WordCloud for Python 文档-word cloud 1 . 6 . 0 . post 54+GB 870 feb 文档

### 在这里你可以找到如何用我的 Python wordcloud 项目创建 wordcloud 的说明。与其他词云相比…

amueller.github.io](https://amueller.github.io/word_cloud/index.html)