# 用 Python 从新冠肺炎新闻中提取关键词

> 原文：<https://medium.com/analytics-vidhya/extracting-keywords-from-covid-19-news-with-python-13249571d37b?source=collection_archive---------1----------------------->

## 使用 Python 和 NLP 技术，我们能够提取由冠状病毒(新冠肺炎)引起的新传染病的超过 100，000 篇文章和出版物的关键词。

![](img/b2410954f13d74060771334a11602d74.png)

使用 Python 从大量文本中提取关键词

新型冠状病毒——或现在称为新冠肺炎——作为一种新的传染病正在影响整个世界。从不同来源的几条新闻中，我们能够应用一些 NLP 技术和框架。

在本文中，我们将提取每篇文章的*关键词*，并创建一个数据集，以使用一个应用了词性标注概念的函数来识别*关键词*。

我们将使用 spaCy 和 News API，这是一个很好的数据源，可以从网络上搜索和检索实时文章。本文将有几个部分:

*   设置
*   编码
*   结论
*   未来作品

我们走吧！

![](img/e56fb883a5e3105e646671564fa9efc8.png)

来做点编码吧！

## 设置— Google Colab

首先让我们了解一下本文中将要用到的所有东西。我首先在 Google Colab 上创建了这个笔记本——但是这可以用另一个 IDE 或 Python 笔记本来实现。我们将使用 pip 安装 spaCy，只要我会说巴西葡萄牙语，我就会使用英语语言模型。

新闻 API 也有一个 Python 库，我们可以用 pip 安装。

```
!pip install spacy
!pip install newsapi-python
```

[](https://spacy.io/usage) [## 安装空间空间使用文档

### 使用 pip，spaCy 发行版可以作为源包和二进制轮获得(从 v2.0.13 开始)。将模型下载到…

空间. io](https://spacy.io/usage) 

之后，我们可以下载最大尺寸的 spaCy 英语语言模型。这个型号的文件大小约为 800MB，你可以下载其他版本的中小版本。我们将使用下面的代码:

```
!python -m spacy download en_core_web_lg
```

安装后，您需要导入空间库，我们将使用另一个库来帮助实现 NLP 分析。我们还必须导入空间模型，并通过一个我们称之为 *nlp_eng* 的变量来加载它。

```
nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient (api_key='PLACE_HERE_YOUR_API_KEY')
```

这完成了我们的设置，之后我们可以做一些编码。

## 编码

新闻 API 提供了一种简单的方法来研究大量不同来源的文章和出版物，我们可以免费获得 API_KEY。当我们发送一个 HTTP 请求时，API 最多返回 100 篇文章——您必须向 dev 帐户付费才能获得结果总数。

 [## 文档—新闻 API

### News API 是一个简单的 HTTP REST API，用于搜索和检索网络上的实时文章。它可以帮助你…

newsapi.org](https://newsapi.org/docs) 

这些文章分成几页，每页有 20 篇文章。您必须实现某种类型的分页，才能免费获得总共 100 篇文章。我们对过去的最大搜索日期也有限制，即 30 天。

```
temp = newsapi.get_everything(q='coronavirus', language='en', from_param='2020-02-03', to='2020-03-03', sort_by='relevancy', page=pagina)
```

当我在 Google Colab 上使用 Python 笔记本时，需要将数据集保存在我的 Google Drive 中，以便在发生任何糟糕的事情时维护数据。我用泡菜库保存了所有的文章。我们可以创建并保存。 *pckl* 文件使用这些命令:

```
filename = 'articlesCOVID.pckl'
pickle.dump(articles, open(filename, 'wb'))filename = 'articlesCOVID.pckl'
loaded_model = pickle.load(open(filename, 'rb'))filepath = '/content/path/to/file/articlesCOVID.pckl'
pickle.dump(loaded_model, open(filepath, 'wb'))
```

之后，我们将处理数据并转换 Pandas 数据框架中的文章字典。我们从新闻 API 获得的原始 JSON 非常丰富，我们可以对其进行清理，只使用*标题*、*日期*、*描述*和*内容*。

```
for i, article in enumerate(articles):
    for x in article['articles']:
        title = x['title']
        description = x['description']
        content = x['content']
        dados.append({'title':titles[0], 'date':dates[0], 'desc':descriptions[0], 'content':content})df = pd.DataFrame(dados)
df = df.dropna()
df.head()
```

在我们清理数据并创建一个 Pandas DataFrame 之后，我们将创建一个名为***get _ keywords _ eng***的函数，该函数将接收 ***文本*** ，并使用 spaCy 模型来识别与新闻的*关键字*相匹配的词性标注。

> 词性标注是基于在文本或语料库中标记单词的概念，因为它对应于特定的定义及其上下文。还考虑与短语、句子或段落中的其他相关单词的关系来识别正确的标签。

我们将提取出*动词*(动词)、一个*名词*(名词)和*专有名词* (PROPN)。

```
if (token.text in nlp_eng.Defaults.stop_words or token.text in punctuation):
  continue
if (token.pos_ in pos_tag):
  result.append(token.text)return result
```

创建好函数后，是时候将它应用到我们的数据帧中了。我们可以在 DataFrame 中添加一个名为 *keywords* 的新列来接收函数的结果。

我已经用文章的*标题*、*描述*和*内容*进行了测试。我得到的最好的结果是文章的*内容*，因为它的字数最多。

```
for content in df.content.values:
    results.append([('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)])df['keywords'] = results
```

之后，我们可以将这些新数据插入到我们的数据帧中，并再次保存到我们的。 *pckl* 文件。

## 结论

现在我们有了一个数据集，其中包含了每篇文章中最常见的 5 个关键词，并与另外几篇关于新冠肺炎的文章连接在一起，是时候选择展示我们结果的最佳方式了。我选择了文字云，这是一种根据文字出现的频率显示文字的图片。

```
text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

在我们的图片下方:

## 未来的工作

当然，这项工作可以不断发展，得到更多的应用，而不仅仅是创建一个单词云。

您可以应用 DataFrame 的其他部分，或者创建另一种方法来连接多个字段。也许从推特或脸书获取数据会很棒。

我的一项调查显示，我们有足够的数据集来训练新冠肺炎新闻的语言模型，并通过这些新闻获得一些问题的答案。我现在正在和[伯特](https://arxiv.org/abs/1810.04805)和[奥博奈·GPT-2](https://openai.com/blog/better-language-models/)一起尝试——很快会有更多的结果！

![](img/50be8e8a5801070810c45c825ceaabda.png)

感谢阅读，并随时评论和分享！

在 [LinkedIn](https://www.linkedin.com/in/gilvandroneto1991/) 上联系我保持联系。:)