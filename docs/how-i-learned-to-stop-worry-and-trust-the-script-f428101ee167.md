# 我是如何学会不再担心并相信剧本的。

> 原文：<https://medium.com/analytics-vidhya/how-i-learned-to-stop-worry-and-trust-the-script-f428101ee167?source=collection_archive---------12----------------------->

![](img/a0f716debaa4cd702cd5e5981a751106.png)

## 或者如何在 python 中自动创建特定的文本分类模型

互联网很棒，我们可以找到任何主题的信息。然后，困难的部分是组织它们，并根据您感兴趣的主题将这些数据分类到正确的桶中。一个经典的方法是浏览所有内容，阅读它，以某种方式将你提取的主题存储在某个地方(在你的脑海中，post-it，csv 文件…)，然后手动将所有这些文档分类到正确的桶中。如果你的语料库不太大，这还可以，但是一旦你有十几个长度变化很大的文档，这就成了一个很大的负担。此外，您可能只对文档中的一个主题感兴趣，但仍然需要阅读全部内容，并且您的分类可能不一致。肯定有更好的方法来利用我们的时间…

*牢记这些要点，我编写了一个脚本，用于文本数据语料库的自动提取和分类。除了创建这个解决方案，其他人的目标是获得一些关于数据废弃、API 和空间的实践经验。*

# **1-查找数据**

对于任何数据科学项目，最困难的部分通常是找到一个有意义的数据集，该数据集具有足够的内容和代表性数据，以实际上表现良好。幸运的是，至少对于 NLP 项目来说，很多人愿意用近 300 种语言发表大约 3000 万篇文章。在过去的 18 年里，人们在一个开放的在线平台上自由地发表文章并互相更正。我当然指的是在线百科全书维基百科。该平台提供了一个关于许多主题的记录良好、相当可靠且组织良好的百科全书，因此可以用作 NLP 项目的数据提取的参考。首先，这些数据必须从维基百科中删除。由于维基百科页面有这样的格式[https://en.wikipedia.org/wiki/Turing_machine,](https://en.wikipedia.org/wiki/Turing_machine,)其中‘en’代表英语，而‘Turing _ machine’是你想要探索的主题，你可以动态地创建一些你将用来获取数据的 URL(起初我想使用维基百科 API，但是在创建这个脚本时没有一个被支持)。这样，脚本将遍历 html 页面，提取 span 部分中的每个字幕，并在以后将其用作数据集的标签。

这是通过这个函数使用 beautifulsoup 完成的:

```
######################
##this function takes as input the 
##language and the subject you want to create a classifier for
## it will then get the ad hoc url in wkipedia
## then it will scrape each span to get its title as a category
#################### 
def get_category(language, subject):                                        wikipedia.set_lang(language)                           
    source = urllib.request.urlopen(wikipedia.page(subject).url).read()                             soup = bs.BeautifulSoup(source,'lxml')                                   soup_txt = str(soup.body)                           
    category = []                           
    for each_span in soup.find_all('span', {'class':'mw-headline'}):                                   
        soup = BeautifulSoup(str(each_span).replace(' ','_'), "html.parser").getText()                                              category.append(soup)                           
    return category
```

然后，一旦你有了标签，你就需要相应的数据。这是通过以下方式实现的:

```
#################### 
## this function takes as input the language and the subject 
## it gets the text for each span and cleans it 
## then it will output a list containing each clean sentence #################### 
def get_data(language, subject): 
    wikipedia.set_lang(language) 
    source = urllib.request.urlopen(wikipedia.page(subject).url).read() 
    soup = bs.BeautifulSoup(source,’lxml’) 
    soup_txt = str(soup.body) 
    div = [] 
    for each_span in soup.find_all(‘span’, {‘class’:’mw-headline’}):             str(each_span).replace(‘ ‘,’_’) 
        div.append(str(each_span)) 
    filter_tag = [] 
    i = 0 
    while i < len(div)-1: 
        start = div[i] 
        end = div[i+1] 
        text = soup_txt[soup_txt.find(start)+len(start):soup_txt.rfind(end)] 
        soup = str(BeautifulSoup(text, “html.parser”)) 
        soup = re.sub(“([\(\[]).*?([\)\]])”, “\g<1>\g<2>”, soup) 
        soup = BeautifulSoup(soup, “html.parser”) 
        soup = re.compile(r’<img.*?/>’).sub(‘’, str(soup.find_all(‘p’))) 
        soup = BeautifulSoup(soup, “html.parser”) 
        soup = (re.sub(“[^a-zA-Z,.;:!0–9]”,” “,soup.getText()).replace(‘[‘,’’).replace(‘]’,’’).lstrip().rstrip().lower())
        clean_text = re.sub(‘ +’, ‘ ‘,soup).replace(‘,’,’ ‘)              filter_tag.append(clean_text) 
        i += 1 
   return filter_tag
```

最后，所有东西都通过函数组合在一起:

```
#################### 
## this function takes as input the list of filter tags
## it will output a list containing the number of sentences in each category 
#################### 
def get_len_list(filter_tag): 
    filtered_text = [] 
    len_list = [] 
    i = 0 
    while i < len(filter_tag): 
        doc = nlp(filter_tag[i]) 
        text = [sent.string.strip() for sent in doc.sents] 
        filtered_text.append(text) 
        len_list.append(len(filtered_text[i])) 
        i += 1 
    return filtered_text,len_list
```

这将创建一个基于主题和语言的 had hoc pandas 数据框架。在每个区间中提取的每个句子被提取，并且它所链接的字幕被用作标签。

```
#################### 
##this function takes as input the len_list and the list of category ## it will output a dataframe containing the label (category) for each sentence 
#################### 
def generate_dataset(len_list, category): 
    i = 0 
    label_list = [] 
    while i < len(len_list): 
        j = 0 
        if len_list[i] != 0: 
            while j != len_list[i]: 
            label_list.append(category[i].lower()) 
            j += 1 
        i += 1 
    flat_list = [item for sublist in get_len_list(get_data(language, subject))[0] for item in sublist] 
    data = {‘text’: flat_list,’label’: label_list} 
    df = pd.DataFrame.from_dict(data) print(df.head()) 
    print(‘Repartition of labels:’, df[‘label’].iloc[0]) 
    print(‘Data Shape:’, df.shape) 
    return df
```

# 2-训练模型

一旦有了带有标注的特殊数据集，现在就该挑选模型并对其进行训练了。起初，我想用 LSTM 进行分类实验，但结果并不太好，我想这是因为我没有太多的数据。所以我选择了一种更经典的方法，使用一些众所周知的 NLP 操作。

首先，我去掉了给定语言的停用词(spacy 提出了英语、法语、德语、西班牙语、葡萄牙语、意大利语和荷兰语的内置列表)。

```
#################### 
## this function takes as input the language 
## it will output the list of stopwords for this language as a list #################### 
def get_stop_words(language): 
    if language == ‘en’: 
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS 
    if language == ‘fr’: 
        spacy_stopwords = spacy.lang.fr.stop_words.STOP_WORDS 
    if language == ‘de’: 
        spacy_stopwords = spacy.lang.de.stop_words.STOP_WORDS 
    if language == ‘es’: 
        spacy_stopwords = spacy.lang.es.stop_words.STOP_WORDS 
    if language == ‘pt’: 
        spacy_stopwords = spacy.lang.pt.stop_words.STOP_WORDS 
    if language == ‘it’: 
        spacy_stopwords = spacy.lang.it.stop_words.STOP_WORDS 
    if language == ‘nl’: 
        spacy_stopwords = spacy.lang.nl.stop_words.STOP_WORDS 
    return spacy_stopwords
```

然后，我应用了词汇化，只保留了单词的词根。我可以使用词干，但是它不太适合拉丁语。

```
#################### 
## this function takes as input the language on the generated dataset 
## it performs some last cleaning and data visualsation on the dataset 
#################### 
def clean_dataset(language, df): 
    srce_labels = df.label.values.tolist() 
    srce_text = df.text.values.tolist() 
    spacy_stopwords = get_stop_words(language) 
    clean_text = [] 
    text = [] 
    i = 0 
    while i < len(srce_text): 
        extract = [] 
        doc = nlp(srce_text[i]) 
        for token in doc: 
            extract.append(token.lemma_)                     clean_text.append(“,”.join(extract).replace(“,”,” “).replace(“ “,” “)) 
        i += 1 
    print(‘Number of stop words: %d’ % len(spacy_stopwords)) 
    i = 0 
    while i < len(clean_text): 
        doc = nlp(clean_text[i]) 
        tokens = [token.text for token in doc if not token.is_stop]             text.append(“,”.join(tokens).replace(“,”,” “).replace(“ “,” “).replace(“ “,” “).replace(“-PRON-”,” “).rstrip().lstrip()) 
        i += 1 
    data = {‘text’: text,’label’: srce_labels} 
    df = pd.DataFrame.from_dict(data) 
    df = df.dropna() 
    return df
```

我应用了更多的操作来得到一个更干净的数据集，适合 NLP 操作。

问题是模型理解数字，而不是单词，所以我决定使用单词包方法将这些单词转换成数字。

然后在一个随机森林中应用网格搜索，以获得这个给定主题在这个给定语言中可能的最佳模型。然后，模型被保存为 pickle 格式，并可以被调用进行分类。

# 3-结论

这个小项目让我有机会尝试许多不同的新概念和新技术，从使用最新库的经典 NLP 操作到 web 报废和公共 API 操作以及其他库，如 argparse 或 pickle。我想通过添加虚拟环境和 docker 容器来不断改进这一点，使它们更容易复制和共享，并在这个问题上与 LSTM 一起做好实验。但至少有了这个项目，我有了一个一致的、非常强大的方法来为 7 种语言中给定主题的文本语料库分类创建自动模型。这就是我如何学会不再担心并相信剧本的。

# 4 向前进

到目前为止，我认为当前模型的几个缺点很容易解决:

*   使用 TF-IDF 代替 countvectorizer 可能会得到更一致的矢量化结果
*   使用进化算法，通过例如 TPOT 库而不是网格搜索来创建更好的分类模型

*资源库可以在这里找到:*[](https://github.com/elBichon/blue_orchid)