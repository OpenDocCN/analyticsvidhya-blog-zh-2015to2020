# 正则表达式提取中的 OCR 校正器

> 原文：<https://medium.com/analytics-vidhya/ocr-corrector-in-regex-extraction-6d2af0d92dc?source=collection_archive---------9----------------------->

从文本图像中提取实体有不同的方法。一些方法是:

1.  正则表达式
2.  模板
3.  机器学习/深度学习

我们不会深入每种方法的细节。然而，我们将探索一种方法来改善正则表达式提取。

正则表达式实体提取-在图像的 OCR 文本上应用正则表达式来提取实体。

对于本教程，让我们以 PAN 文档(在印度使用)为例。这里可以使用任何类型的文本图像和任何类型的 OCR 引擎。

让我们假设下面的 OCR 文本是完美的/没有任何错误

```
good_ocr_text = '''आयकर विभाग
भारत सरकार
GOVT OF INDIA
INCOME TAX DEPARTMENT
स्थायी लेखा संख्या काई
Permanent Account Number Card
ABCXY1234Z
Name
ABCD XYZ
father's Name
PQRST UVW
Applcart Sigrahre
Date of Birth
01/01/1975
uSianature'''
```

现在我们将编写一些正则表达式来从 OCR 文本中提取实体

```
def getPanName(text):
   pattern='(?:Name)\s*([a-z]*\s*[a-z]*\s*[a-z]*)$'
   value = re.search(pattern, text, re.I|re.MULTILINE)
   if value:
      return value.group(1)def getPanNumber(text):
   pattern= '(?:Number Card)\s*([a-z]{5}\d{4}[a-z]{1})$'
   value = re.search(pattern, text, re.I|re.MULTILINE)
   if value:
      return value.group(1)print('Name:',getPanName(good_ocr_text))
Name: ABCD XYZprint('PAN No:',getPanNumber(good_ocr_text))
PAN No: ABCXY1234Z
```

我们使用简单的正则表达式成功地提取了想要的实体。

现在，让我们假设由于某种原因(图像质量改变，OCR 引擎改变等)。)OCR 文字不好

```
Number --> Numper
Name --> Nane bad_ocr_text = '''आयकर विभाग
भारत सरकार
GOVT OF INDIA
INCOME TAX DEPARTMENT
स्थायी लेखा संख्या काई
Permanent Account Numper Card
ABCXY1234Z
Nane
ABCD XYZ
father's Nane
PQRST UVW
Applcart Sigrahre
Date of Birth
01/01/1975
uSianature'''
```

运行相同的正则表达式不会产生预期的结果

```
print('Name:',getPanName(bad_ocr_text))
Name: Noneprint('PAN No:',getPanNumber(bad_ocr_text))
PAN No: None
```

因此，我们有一种可能的方法来达到预期的结果，那就是更新现有的正则表达式，使其包含拼写错误的单词。

```
pattern='(?:Name|Nane)\s*([a-z]*\s*[a-z]*\s*[a-z]*)$'
```

但是我们可以给正则表达式添加多少不正确的单词。OCR 引擎有无数种方法可以拼错一个单词。

那么，如果我们能够纠正这些不好的单词呢？

让我们创建一个正则表达式中使用的单词列表。我们称之为好词(稍后他们将与坏词斗争并取代它们:)

```
good_words = ['Name','Number','Card']
```

我们将编写一些代码来用好词替换坏词。

```
'''
If bad OCR word matches(75% match) with any of good word then that bad OCR word should be replaced with good word
'''
def ocr_corrector(text,good_words):
  corrected_text = ''
  for sent in text.split("\n"):
    #print(sent)
    new_sent = []
    for word in sent.split(" "):
      if not word.lower() in good_words:
        for gword in good_words:
          if Levenshtein.ratio(word,gword) >= 0.75:
            word = gword.strip()
      new_sent.append(word)
    corrected_text += (" ".join(new_sent)+"\n")
  return corrected_text
```

现在，我们将把错误的 OCR 传递给这个方法

```
bad_ocr_text_corrected = ocr_corrector(bad_ocr_text,good_words)'''
Bad OCR words are corrected.
Numper --> Number
Nane --> Name
'''
print(bad_ocr_text_corrected)आयकर विभाग
भारत सरकार
GOVT OF INDIA
INCOME TAX DEPARTMENT
स्थायी लेखा संख्या काई
Permanent Account Number Card
ABCXY1234Z
Name
ABCD XYZ
father's Name
PQRST UVW
Applcart Sigrahre
Date of Birth
01/01/1975
uSianature
```

这种校正的 OCR 将给出期望的结果

```
print('Name:',getPanName(bad_ocr_text_corrected))
Name: ABCD XYZ
print('PAN No:',getPanNumber(bad_ocr_text_corrected)) 
PAN No: ABCXY1234Z
```

因此，我们在不更新基本正则表达式的情况下获得了想要的结果。我们可以根据特定场景中预期的 OCR 质量，将 75%匹配阈值更改为任何其他阈值。

OCR 校正器不必仅应用于正则表达式提取。它可以应用于任何涉及 OCR 文本的用例。

完整的代码可在 [git-hub](https://github.com/sarang0909/OCR-Corrector) 获得

如果你喜欢这篇文章或有任何建议/意见，请在下面分享！

让我们在 [LinkedIn](https://www.linkedin.com/in/sarang-mete-6797065a/) 上联系讨论