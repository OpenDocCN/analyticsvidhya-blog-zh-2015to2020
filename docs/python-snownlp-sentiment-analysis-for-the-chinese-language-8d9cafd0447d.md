# Python 和 SnowNLP:中文情感分析

> 原文：<https://medium.com/analytics-vidhya/python-snownlp-sentiment-analysis-for-the-chinese-language-8d9cafd0447d?source=collection_archive---------8----------------------->

![](img/966f20a0cd0d2f2b23efcedbc716ef5b.png)

拍摄于印刷出版工艺学校

三亚大学要求我创建一个脚本，从一个特定的 URL 获取所有的新闻，然后获取关键词、摘要和总体情绪。

Python 是一种漂亮而强大的语言，非常适合这个任务。我将使用 Python 3.7.1 在卡特琳娜为这个项目。

# 各个击破

提供的网址是[http://people.com.cn/](http://people.com.cn/)。说到代码，我总是用“划分&征服”来处理任何问题。因此，我的第一个任务是，抓取并获得来自 http://people.com.cn/[的所有链接。为此，我使用了](http://people.com.cn/) [**请求库**](https://pypi.org/project/requests/) 和*[**bs4**](https://pypi.org/project/bs4/)***。*** 此处代码抓取所有网址*

```
*def get_all_website_links(url):

    urls = set()
    domain_name = urlparse(url).netloc
    soup = BeautifulSoup(requests.get(url).content, "html.parser") for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            continue href = urljoin(url, href)
        parsed_href = urlparse(href)
        # cleaning URL
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path if not is_valid(href):
            # not a valid URL
            continue
        if href in internal_urls:
            # already in the set
            continue
        if domain_name not in href:
            # external link
            if href not in external_urls:
                external_urls.add(href)
            continue urls.add(href)
        internal_urls.add(href)
    return urls*
```

*我将所有抓取的 URL 保存到一个. txt 文件中。使用循环，我使用[**newspaper 3k**](https://github.com/codelucas/newspaper)**从每篇文章中提取内容。***

```
*from newspaper import Article*
```

*导入后，您获得 URL，然后解析内容。*

```
*url = line
a = Article(url, language='zh') # Chinese
a.download()
a.parse()*
```

*然后，解析完所有内容后，就到了施展魔法的时候了*

# *SnowNLP*

*首先，什么是 NLP？*

*[**自然语言处理** ( **NLP** )](http://Natural_language_processing) 是语言学、计算机科学、信息工程和人工智能的一个分支，涉及计算机和人类(自然)语言之间的交互，特别是如何编写计算机程序来处理和分析大量自然语言数据(摘自维基百科)*

*Snow 是一个使用 NLP 的 Python 库，它与诸如中文之类的语言兼容。*

*首先，您必须通过 SnowNLP 类进行初始化，如下所示:*

```
*from snownlp import SnowNLP s = SnowNLP(u’我喜欢红包’)*
```

*建议以 **u** 为前缀，表示这是 **Unicode 字符串**。*

*汉语是一种复杂的语言，因为单词之间没有空格。这使得很难确定一个句子中的单词数，因为一个单词可以是一个或多个汉字的组合。因此，在进行任何自然语言处理时，都需要将整个文本拆分成单词。您可以很容易地使用以下命令来进行令牌化:*

```
*from snownlp import SnowNLP s = SnowNLP(u’我喜欢红包’)
s.words*
```

*这将会回来*

```
*['我', '喜欢', '红包']*
```

*如果你想得到这些单词的标签(我说的标签是指这个单词是名词、副词、动词、形容词等等)，你可以使用下面的函数*

```
*from snownlp import SnowNLP 
s = SnowNLP(u’我喜欢红包’)
list(s.tags)*
```

*这将会回来*

```
*[('我', 'r'), ('喜欢', 'v'), ('红包’', 'n')]*
```

*   ***r** :指代词。*
*   ***v** :指动词。*
*   ***名词**:指名词。*
*   ***w** :指标点符号。*

*而且像这样，还有很多其他有用的功能。*

# *更深入*

*回到任务上来，现在我们需要获取关键字，为此我将使用:*

```
*s.keywords(10)*
```

*其中数字 10 表示关键字的数量。然后我们继续总结:*

```
*s.summary(3)*
```

*其中 3 将是摘要的最大数量的句子。*

*最后，我们将整篇文章分成句子，并使用强大的功能:*

```
*s.sentiments*
```

*出于解释的目的，我们总结出了一个更简单的版本*

```
*s = SnowNLP(a.text[:1000])# keywords
print("The keywords are:")
print(*s.keywords(5), sep=", ")

# summary
print("The summary is:")
print(*s.summary(2), sep=", ")
print(" ")

#sentiment
sent = s.sentences
for sen in sent:
    s = SnowNLP(sen)
    print(s.sentences)
    print(s.sentiments)*
```

*变量 **a** 是使用报纸上的文章读取 URL 的全部内容的结果*

*这段代码将输出句子，然后输出一个 0 到 1 之间的数字。值输出范围从 0 到 1，0 表示负面情绪，1 表示正面情绪。您还可以使用自定义文本数据集来训练您的模型。*

*有了这个，我就能获取所需的大部分信息。*

# *感谢阅读*

*这篇不起眼的文章演示了如何使用 SnowNLP 模块对简体中文进行情感分析。自然语言处理有更多的选择，每种语言都有其优缺点。对于中文(简体中文)来说，SnowNLP 是最好的。*

*如有任何问题或意见，请访问[https://www.lastrescarlos.com/](https://www.lastrescarlos.com/)*