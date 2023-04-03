# 理解词典释义

> 原文：<https://medium.com/analytics-vidhya/comprehending-dictionary-comprehensions-42a694d6fe2?source=collection_archive---------17----------------------->

*TLDR；词典理解可以组织和储存大量信息。它们只需要一行代码。*

这里有一个例子:

```
#This is just creating a list of numbers, called key_list. 
key_list = [0,1,2,3,4,5,6,7,8,9]#Just a simple function right here, returns input plus 100
def add_hundred(number):
 return input + 100#THIS IS THE DICTIONARY COMPREHENSION
dict_comp = {key:add_hundred(key) for key in key_list}
```

下面是 dict_comp 包含的内容。

```
dict_comp{0: 100
 1: 101,
 2: 102,
 3: 103,
 4: 104,
 5: 105,
 6: 106,
 7: 107,
 8: 108,
 9: 109}
```

那么到底是怎么回事呢？下面我会详细解释。

## 字典理解，一个爱情故事。

![](img/d52bddb2bd1f1b7b25bdbd0ba9386306.png)

你不是我真正的**型**，我不想**串**你混日子。

我想写这篇博客，因为学习编码是令人生畏的。我很早就被教导字典很重要，但不是如何和为什么。

我会快速解释什么是字典，为什么它们如此有用，并分解我上面写的字典理解。

> 注意:这篇文章假设你对 for 循环、列表和函数有所了解。如果你能写一份清单理解，那是非常酷和有帮助的，但不是必须的。列表理解和字典理解是相似的。

## 字典的关键在于理解它们的价值

**TLDR；字典是键值对的列表。**

列表是基于索引排序的，而字典是基于键排序的。键可以是任何不可变的数据类型，通常是 string 或 int。在我看来，理解一本字典最有帮助的方法是…想象一本字典。

![](img/b85a21f6b51819e4eec92c320d8e5913.png)

这太元了

把单词想象成键，把它们的定义想象成值。当你想查一个单词的意思时，你只需在字典里找到这个单词。

虽然键必须是不可变的，但值却不是。这些值可以是整数、字符串、列表、数据帧、函数、数组、元组，甚至其他字典！

这就是字典如此有用的原因。它们可以容纳许多不同类型的物品，并保持有序。

## 无聊地编字典。

词典其实并没有那么难制作。下面是它们的样子:

```
dict_ators = {'Admiral General Aladeen': 'Wadiya',
              'Joffrey Baratheon': 'Westeros',
              'President Snow': 'Panem', 
              'Dwight Schrute': 'Dunder-Mifflin Scranton'}
```

这个……有效，但是需要一段时间。如果您想向现有字典中添加新元素，您可以这样做

```
dict_ators['Dwight Schrute'] = 'Dunder-Mifflin Scranton'
```

![](img/c857303e4600efd089d22a390b90338a.png)

手动编写词典后的感受

这篇博文并不是要用强硬的方式做事。所以现在你知道了如何用枯燥的方法编写字典，你可以学习更酷更快的理解字典。

## 分解字典理解

只是把我之前发布的代码带回来。

```
#This is just creating a list of numbers, called key_list. 
key_list = [0,1,2,3,4,5,6,7,8,9]#Just a simple function right here, returns input plus 100
def add_hundred(number):
 return input + 100#THIS IS THE DICTIONARY COMPREHENSION
dict_comp = {key:add_hundred(key) for key in key_list}
```

所以让我们来分解实际的字典理解

## 第 1 部分:从结尾开始

从结尾开始实际上更有意义

```
for key in key_list
```

那是 for 循环吗？哦，哇，这是一个 for 循环！所以变量 key 是不断迭代的。在第一次迭代中，key = 0。下一次迭代 key = 1，依此类推。

## 第 2 部分:关键想要的功能，必须有功能

```
key:add_hundred(key)
```

现在，每个键-值对都将是键，并且将该特定键作为函数的输入。哇哦。什么？

更容易想到一个具体的例子。让我们把 0 作为关键。

我们把 0 放入函数 add _ Bailey

```
add_hundred(0)
 return 0 + 100
```

结果是 100。记得我们用 0 作为密钥。所以这一对将是

```
0:100
```

1 英镑一对

```
1:101
```

诸如此类…

## 第三部分:停止制造这样的支架

```
dict_comp = {key:add_hundred(key) for key in key_list}
```

最后，在开头和结尾加上花括号。然后，您需要为字典指定一个名称。在这种情况下，我称之为 dict_comp。

![](img/7e330eba13ff659bba84e876f396c4b5.png)

这就是理解和使用字典的理解。恭喜你坚持到了最后。如果你有任何建议或批评，请评论。感谢阅读和快乐理解。