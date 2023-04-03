# iOS 开发中的 NLP

> 原文：<https://medium.com/analytics-vidhya/nlp-in-ios-development-3238c2cfd244?source=collection_archive---------36----------------------->

使用 NLP 让您的应用变得智能

![](img/78a57e85919cfe768276d2570ef2f760.png)

Jazmin Quaynor 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

NLP 是人工智能的一个领域，有助于更好地理解人类语言。自然语言处理主要用于文本分类和标注。

许多自然语言处理任务必须从标记化、词条化和嵌入开始。苹果公司提供“自然语言”框架来对用户数据执行 NLP 任务。

苹果还提供了用于问题回答的 BERT-SQuAD CoreML 模型。这个可以直接用在我们的 app 里，步骤很少。

但是理解自然语言框架是执行诸如实体识别和文本分类等 NLP 任务所必需的。

让我们看一些我们可以用自然语言框架执行的琐碎的 NLP 任务。

# 标记化

自然的人类语言是一串单词和句子。它将被标记或列举到段落、句子或单词中进行处理。

“自然语言”中的 NLTokenizer 类用于完成此任务。

```
let tokenizer = NLTokenizer(unit: .word)
```

这里的单元可以是单词或句子，这取决于使用情况。点击了解更多关于符号化[的信息。](https://developer.apple.com/documentation/naturallanguage/tokenizing_natural_language_text)

# 语言识别

在自然语言处理中，一个非常普通和基本的任务是识别给定文本的语言。

这可以通过“NLLanguageRecognizer”轻松完成。

```
let recognizer = NLLanguageRecognizer()
recognizer.processString("English Language")
```

您可以指定可能的语言列表，如下所示

```
recognizer.languageConstraints = [.french, .english, .german,                                  .italian, .spanish, .portuguese] 
```

点击了解更多关于语言识别的信息[。](https://developer.apple.com/documentation/naturallanguage/identifying_the_language_in_text)

# 词性识别

识别词类有时是必要的。考虑这样一种情况，当提供段落的概述时，您需要从给定的文本中识别名词或动词。

这可以通过“NLTagger”来完成

```
let tagger = NLTagger(tagSchemes: [.lexicalClass])
tagger.string = text
```

标记方案可以是词汇类、名称类型或脚本。只对你的用例使用必要的方案。

例如，当你对文本的词性感兴趣时，如果文本的语言有所不同，那么就包括语言。点击了解更多关于方案[的信息。](https://developer.apple.com/documentation/naturallanguage/nltagscheme)

根据用例，您可以进一步提供诸如 omitPunctuation 和 omitWhiteSpaces 之类的选项。

点击了解更多关于 POS 识别[的信息](https://developer.apple.com/documentation/naturallanguage/identifying_parts_of_speech)

## 人员、地点和组织标识

识别人物、地点或组织有时是快速浏览文本所必需的。

这类似于 POS 识别。将 nametype 指定为 tagScheme，并使用 tags 数组筛选可用的枚举标记。

```
let tagger = NLTagger(tagSchemes: [.nameType])tagger.string = text let options: NLTagger.Options = [.omitPunctuation, .omitWhitespace, .joinNames]
let tags: [NLTag] = [.personalName, .placeName, .organizationName]
```

点击了解关于此任务[的更多信息。](https://developer.apple.com/documentation/naturallanguage/identifying_people_places_and_organizations)

# 把...嵌入

文本嵌入是简单的 NLP 任务。文本嵌入用于查找相似的单词和句子。这也是建立推荐引擎的基本步骤。

自然语言框架的 NLEmbedding 类可以用来完成这项任务。

```
let embedding = NLEmbedding.wordEmbedding(for: .english)
```

通过嵌入，我们可以获得向量表示，使用距离知道单词之间的相似性，或者获得单词的邻居。

```
if let vector = embedding.vector(for: word) { 
print(vector) }
let specificDistance = embedding.distance(between: word, and: “motorcycle”) 
```

以上步骤也适用于句子。在这里了解关于这个任务[的更多信息。](https://developer.apple.com/documentation/naturallanguage/finding_similarities_between_pieces_of_text)

# 结论

我们已经看到了一些可以用“自然语言”框架执行的基本 NLP 任务。我们也可以使用 createML 创建自定义的单词标签或分类器。

## 参考

【https://developer.apple.com/documentation/naturallanguage 