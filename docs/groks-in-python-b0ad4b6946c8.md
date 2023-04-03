# Python 中的 Groks

> 原文：<https://medium.com/analytics-vidhya/groks-in-python-b0ad4b6946c8?source=collection_archive---------8----------------------->

![](img/581109ecb947414ab98f8992171cc72a.png)

在我之前的[博客](https://techscouter.blogspot.com/2020/04/using-grok-for-information-extraction.html)中，我写了关于使用 GROKS 和 REGEX 提取信息的文章。

如果你还没有读过，我会鼓励你先浏览一下[这个](https://techscouter.blogspot.com/2020/04/using-grok-for-information-extraction.html)博客。

任何工具的一个重要方面是在不同环境中使用它并自动执行任务的能力。

在这篇文章中，我们将会看到使用 [pygrok](https://github.com/garyelephant/pygrok) 库在 python 中实现 GROKs。

到目前为止，我们知道 GROKs 是一种可读性更强的正则表达式。

# 装置

Pygrok 是 python 中 grok 模式的一个实现，可以通过 pip 发行版获得

```
pip install pygrok
```

# 使用

这个库对于使用预制的洞穴和我们自己定制的洞穴非常有用。

让我们从一个非常基本的例子开始:

## 解析文本

```
*#import the package* from pygrok import Grok*#text to be processed* text = 'gary is male, 25 years old and weighs 68.5 kilograms'*#pattern which you want to match*
pattern = '%{WORD:name} is %{WORD:gender}, %{NUMBER:age} years old and weighs %{NUMBER:weight} kilograms'#create a GROK object by giving the pattern
grok = Grok(pattern)#use match function to get all the parsed patterns
print grok.match(text)
```

注意:这也将返回部分匹配模式，即忽略字符串开头和结尾的不匹配模式。

所有可用的 GROK 模式列表可以在[这里](https://github.com/garyelephant/pygrok/tree/master/pygrok/patterns)看到

## 使用自定义 GROK 模式

```
*#import the package* from pygrok import Grok*#text to be processed* text = 'gary is male, 25 years old and weighs 68.5 kilograms'*#pattern which you want to match*
pattern = '%{WORD:name} is %{WORD:gender}, %{NUMBER:age} years old and weighs %{NUMBER:weight} kilograms'#create a GROK object by giving the pattern
pat={"S3_REQUEST_LINE": "(?:%{WORD:verb} %{NOTSPACE:request}(?: HTTP/%{NUMBER:httpversion})?|%{DATA:rawrequest})"}
grok = Grok(pattern,**custom_patterns_dir=pattern_dir_path,custom_patterns=pat**)#use match function to get all the parsed patterns
print grok.match(text)
```

我们可以使用 **custom_patterns_dir** 选项提供自定义模式目录，此处目录与此处可见的[目录相同。](https://github.com/garyelephant/pygrok/tree/master/pygrok/patterns)

如果您需要添加一些模式，那么您可以避免创建目录的开销，并将模式作为键值对在 **custom_patterns** 字段中传递。

我觉得有一些功能可以添加到 groks 中，比如给出文件路径而不是目录路径，解析完整的文本或不返回任何内容，等等，我将尝试为这个项目做出贡献。

我希望这篇博客能在解析过程中帮助你。

快乐学习。

更多博客请看我的博客