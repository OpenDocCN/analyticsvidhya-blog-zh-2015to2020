# 命名实体的游戏？—第一部分

> 原文：<https://medium.com/analytics-vidhya/game-of-named-entities-part-1-1ee92bff4838?source=collection_archive---------19----------------------->

奥拉朋友们！我同意这是永恒的。但是我一直忙得不可开交，所以一直保持低调。是的，你猜对了，**文字游戏**是为了稍微改变齿轮带你通过这个平行宇宙所谓的自然**语言**处理。在你认为这是又一个伟大的机器学习博客之前，让我告诉你，我非常同意你的观点。这仅仅是一个关于我们如何利用语言结构来帮助我们的短期失忆症朋友多莉的故事——没有任何那些花哨的模型恶作剧。

迫于同龄人的压力，可怜的多莉决定阅读广受欢迎的电视节目《权力的游戏》。很自然地，她就是记不住这部剧有一百万个角色。我们能不能给她画一张所有主要人物的人物素描，让她的生活轻松一点？布兰，艾莉亚，珊莎，…好吧，我们先列出名单。

![](img/8a58245b47d3c757740a20545740e541.png)

在研究如何解决这个令人生畏的任务时，我偶然发现了这个叫做[的家伙，他的名字叫做实体](https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da)，瞧！基本上， [Python 的自然语言工具包](https://www.nltk.org/)已经有了一些奇妙的资源，不仅可以识别专有名词，还可以将它们分类为人、组织等。我想，这很简单，这就是我最后得到的。

```
with open('../dataset/GOT_book1.pdf', 'rb') as file_stream:
        file_reader = PyPDF2.PdfFileReader(file_stream)
        characters = set()
        for i in range(START_PAGE-1, file_reader.numPages):
            page = file_reader.getPage(i)
            page_content = page.extractText()
            page_content = page_content.replace('\n', ' ')
            page_content = page_content.encode('ascii', 'ignore').decode()
            tokens = nltk.word_tokenize(page_content)
            tagged = nltk.pos_tag(tokens)
            chunks = nltk.ne_chunk(tagged)
            trees = list(filter(lambda x: isinstance(x, nltk.Tree), chunks))
            for t in trees:
                name = ' '.join(c[0] for c in t.leaves())
                if t.label() == 'PERSON' and name:
                    characters.add(name)
        print(characters)
        print('number of characters', len(characters))
```

对于所有那些**幸运的**非 Python/NLTK 母语的人，我们基本上是阅读这本书，清理文本，提取单词，[标记](https://www.nltk.org/book/ch05.html)和[分块](https://www.nltk.org/api/nltk.chunk.html)它们，然后最终提取人。对于其他向我翻白眼的人，这个脚本的一个更干净、更有用的版本马上就要出来了。

现在，是时候做些回应了。

*   首先，剧本需要一个狗的年龄来运行！
*   我们找到了 1549 个字符，这实在是太多了。
*   我们还发现了许多假阳性——**饮料**、**食物**、**烈性**、**冬季**等等……这些大概就是在这样的背景下使用的。比如你单纯说`Winter is coming`，从纯语言角度来说，很难说**冬天**是真的人名还是季节或者*只是大家无缘无故害怕的东西*。
*   雪诺大人、琼恩·雪诺、琼恩、可怜的琼恩都是不同的角色。嗯，这可能说起来容易做起来难。

我们能做得更好吗？遵命船长。

*   与其翻遍所有的页面，也许我们可以跳过，比如说 10 页，然后执行我们的搜索。毕竟，如果你是一个重要人物，*多莉会找到你*，无论如何！
*   我们可以通过使用`nltk.corpus.words.words()`简单地检查它是否是字典中的单词来消除奇怪的字符。请注意，我们应该只检查整个名称，而不是每个单词。例如，我们冒着失去琼恩·雪诺 T4 的风险，仅仅因为雪是一个普通的名词。我们需要不惜一切代价让他苏醒！
*   最后，我们总是可以通过结合所有这些角色的特征来处理一个角色的多个名字(艾德、奈德、艾德大人等等)。在一天结束的时候，多莉只需要能够认出他/她。

所以，最后，我们得到了大约 389 个字符，这还不算太糟。对吗多莉？哦不，她已经忘记我了。在我向她重新介绍我自己的时候，请继续关注。同时，祝你在一个平行世界中快乐编码！😊