# 如何用 python 翻译文本

> 原文：<https://medium.com/analytics-vidhya/how-to-translate-text-with-python-9d203139dcf5?source=collection_archive---------1----------------------->

## 在 python 中使用不同的著名翻译器(例如 google translator 等等)。

![](img/a117e1b9ee3b33b0bb3ba86d7105903c.png)

## 介绍

在本教程中，我们将探索使用 python 翻译文本或单词的不同可能性。根据我的经验，如果你想自动翻译许多段落、句子或单词，这非常有帮助。

此外，您可以拥有一个后端工作器，它不断接收新数据，并且可以返回带有翻译的请求，或者将不同的翻译存储在数据库中(这在 [**NLP**](https://en.wikipedia.org/wiki/Natural_language_processing) 任务中非常有用)。

除了清晰的语法和丰富的库之外，选择 Python 的一个原因是，这个伟大的社区广泛致力于语言本身的开发或使用第三方模块扩展功能。

确切地说，使翻译文本变得简单的模块之一是 [**deep_translator**](https://pypi.org/project/deep-translator/) ，它为多个著名翻译家提供支持。

## 概观

![](img/ac53edb50eb09c33fe00e135d88b0154.png)

deep_translator 是一个灵活的 python 包，以简单的方式在不同语言之间进行翻译。基本上，这个包的目标是在一个广泛的工具中集成许多翻译器，包括[**Google Translator**](https://translate.google.com/)**、DeepL、**[**Pons**](http://pons.com)**、**[**linguie**](https://www.linguee.com/)**和**[**o**](https://mymemory.translated.net/)**thers**。我将在这里举例说明如何使用一些翻译器。然而，我不会面面俱到，因此我鼓励你事后查看官方文件。

## 装置

建议使用 [**pip**](https://en.wikipedia.org/wiki/Pip_(package_manager)) 安装包。简而言之，要安装稳定版，请在终端中运行以下命令:

```
pip install deep-translator
```

这是安装 deep_translator 的首选方法，因为它将始终安装最新的稳定版本。如果你没有安装 pip，这个 [Python 安装指南](http://docs.python-guide.org/en/latest/starting/installation/)可以指导你完成这个过程。

## 谷歌翻译

google translator 已经集成在 deep_translator 包中，导入即可直接使用。然后，创建一个实例，其中源语言和目标语言作为参数给出。之后可以使用 translate 方法返回翻译后的文本。

在下面的代码中，值 *auto* 用于让 google translator 检测哪种语言用作源语言，目标值作为缩写给出，在本例中代表*德语*。

```
from deep_translator import GoogleTranslatorto_translate = 'I want to translate this text'translated = GoogleTranslator(source='auto', target='de').translate(to_translate)# outpout -> Ich möchte diesen Text übersetzen
```

此外，您可以从文本文件进行翻译。这也很简单，可以通过更新前面的代码轻松实现。我还想在这里指出，源语言和目标语言可以通过名称而不是缩写来传递。

```
from deep_translator import GoogleTranslatortranslated = GoogleTranslator(source='english', target='german').translate_file('path/to/file')
```

现在，如果你有不同语言的句子，你想把它们都翻译成相同的目标语言。下面的代码演示了如何做到这一点

```
from deep_translator import GoogleTranslatortranslated = GoogleTranslator(source='auto', target='de').translate_sentences(your_list_of_sentences)
```

好极了。现在我们来探讨一下其他的翻译器。

## 桥

庞斯是德国领先的语言出版商之一。它主要以翻译单词或小句子而闻名。它有一个丰富的单词数据库，在翻译单词和获取同义词方面甚至可以超过谷歌翻译。

幸运的是，deep_translator 也支持 PONS。下面的代码演示了如何使用它。这个 API 看起来和前面的一样，只有很小的变化

```
from deep_translator import PonsTranslatorword = 'good'
translated_word = PonsTranslator(source='english', target='french').translate(word, return_all=False)# output: bien
```

此外，您可以获得 pons 返回所有同义词或建议

```
from deep_translator import PonsTranslatorword = 'good'
translated_word = PonsTranslator(source='english', target='french').translate(word, return_all=True)# output: list of all synonymes and suggestions
```

## 林吉语

Linguee translator 是一个在线双语词典，它为许多语言对提供了在线词典，包括许多双语句子对。与以前的翻译器一样，该功能集成在 deep_translator 包中。

```
from deep_translator import LingueeTranslatorword = 'good'
translated_word = LingueeTranslator(source='english', target='french').translate(word)
```

与 Pons translator 相同，通过设置 *return_all* 参数，您可以获得所有同义词和附加建议

```
from deep_translator import LingueeTranslatorword = 'good'
translated_word = LingueeTranslator(source='english', target='french').translate(word, return_all=True)
```

## 我的记忆

我的记忆翻译是世界上最大的翻译记忆，它是 100%免费使用。它的创建是为了收集来自欧盟、联合国的 TMs，并整理最好的特定领域多语言网站。

deep-translator 的最新版本支持我的内存翻译。下面的代码演示了如何使用它

```
from deep_translator import MyMemoryTranslatortranslated = MyMemoryTranslator(source="en", target="zh").translate(text='cute')# output -> 可爱的
```

## 结论

本教程演示了如何使用 python 翻译文本和自动化多种翻译。更准确地说是使用了**深度翻译器**包，它支持多个著名翻译器。请记住，我在这里演示了一些功能。事实上，深度翻译器也支持 **DeepL、Yandex 和其他翻译器**，这里不讨论。你可以在官方文件中查到这些。下面是 [**docs**](https://deep-translator.readthedocs.io/en/latest/) 和 github[**repo**](https://github.com/nidhaloff/deep_translator)**的链接。如果你对这个包有任何问题或者你想为这个项目做贡献，请随时给我写信。**