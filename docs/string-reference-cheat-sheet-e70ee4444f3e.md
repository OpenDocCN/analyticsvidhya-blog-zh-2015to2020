# 字符串引用备忘单

> 原文：<https://medium.com/analytics-vidhya/string-reference-cheat-sheet-e70ee4444f3e?source=collection_archive---------14----------------------->

![](img/f9f0039b29dc3d28d36a9592e61e8198.png)

由 [Roman Synkevych](https://unsplash.com/@synkevych?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

在 Python 中，你可以用字符串做很多事情。在这个备忘单中，您将找到最常见的字符串操作和字符串方法。

# 字符串操作

*   len(string)返回字符串的长度
*   for character in string 迭代字符串中的每个字符
*   如果字符串中的子字符串检查子字符串是否是字符串的一部分
*   string[i]访问字符串索引 I 处的字符，从零开始
*   string[i:j]访问从索引 I 开始，到索引 j-1 结束的子字符串。如果省略 I，默认为 0。如果省略 j，则默认为 len(string)。

# 字符串方法

*   string.lower() / string.upper()返回包含所有小写/大写字符的字符串副本
*   string . lstrip()/string . rstrip()/string . strip()返回不带左/右/左或右空格的字符串副本
*   string.count(substring)返回子字符串在字符串中出现的次数
*   如果字符串中只有数字字符，string.isnumeric()返回 True。如果不是，则返回 False。
*   如果字符串中只有字母字符，string.isalpha()返回 True。如果不是，则返回 False。
*   string . split()/string . split(delimiter)返回由空格/分隔符分隔的子字符串列表
*   string.replace(old，new)返回一个新字符串，其中所有出现的 old 已被 new 替换。
*   delimiter.join(字符串列表)返回一个新字符串，其中所有字符串都由分隔符连接

# `tr.**startswith**` ( *前缀* [，*开始* [，*结束* ])

如果字符串以*前缀*开头，则返回`True`，否则返回`False`。*前缀*也可以是要查找的前缀元组。使用可选的*启动*，从该位置开始测试管柱。使用可选的*端*，在该位置停止比较管柱。

`str.**strip**` ([ *chars* ])

返回删除了前导和尾随字符的字符串的副本。 *chars* 参数是指定要删除的字符集的字符串。如果省略或`None`，那么 *chars* 参数默认移除空白。 *chars* 参数不是前缀或后缀；相反，其值的所有组合都被剥离:

>>>

```
**>>>** '   spacious   '.strip()
'spacious'
**>>>** 'www.example.com'.strip('cmowz.')
'example'
```

最外面的前导和尾随*字符*参数值从字符串中剥离。字符从前端移除，直到到达一个不包含在*字符*的字符集中的字符串字符。在尾端发生类似的动作。例如:

>>>

```
**>>>** comment_string = '#....... Section 3.2.1 Issue #32 .......'
**>>>** comment_string.strip('.#! ')
'Section 3.2.1 Issue #32'
```

`str.**swapcase**`()

返回字符串的副本，其中大写字符转换为小写字符，反之亦然。注意`s.swapcase().swapcase() == s`不一定是真的。

`str.**title**`()

返回字符串的大小写形式，其中单词以大写字符开始，其余字符为小写字符。

例如:

>>>

```
**>>>** 'Hello world'.title()
'Hello World'
```

该算法使用一个简单的与语言无关的单词定义，将单词定义为一组连续的字母。该定义在许多上下文中有效，但它意味着缩写和所有格中的撇号形成单词边界，这可能不是想要的结果:

>>>

```
**>>>** "they're bill's friends from the UK".title()
"They'Re Bill'S Friends From The Uk"
```

撇号的变通方法可以使用正则表达式来构造:

>>>

```
**>>> import** **re**
**>>> def** titlecase(s):
**... **    **return** re.sub(r"[A-Za-z]+('[A-Za-z]+)?",
**... **                  **lambda** mo: mo.group(0).capitalize(),
**... **                  s)
**...**
**>>>** titlecase("they're bill's friends.")
"They're Bill's Friends."
```

`str.**translate**` ( *表*)

返回字符串的副本，其中每个字符都通过给定的翻译表进行了映射。表格必须是通过`[__getitem__()](https://docs.python.org/3/reference/datamodel.html#object.__getitem__)`实现索引的对象，通常是一个[映射](https://docs.python.org/3/glossary.html#term-mapping)或[序列](https://docs.python.org/3/glossary.html#term-sequence)。当按 Unicode 序数(一个整数)进行索引时，table 对象可以执行下列任何操作:返回 Unicode 序数或字符串，以将字符映射到一个或多个其他字符；return `None`，删除返回字符串中的字符；或者引发一个`[LookupError](https://docs.python.org/3/library/exceptions.html#LookupError)`异常，将角色映射到自身。

您可以使用`[str.maketrans()](https://docs.python.org/3/library/stdtypes.html#str.maketrans)`从不同格式的字符到字符映射创建一个翻译映射。

另见`[codecs](https://docs.python.org/3/library/codecs.html#module-codecs)`模块，了解更灵活的自定义字符映射方法。

`str.**upper**`()

返回字符串的副本，其中所有大小写字符 [4](https://docs.python.org/3/library/stdtypes.html#id15) 都转换为大写。请注意，如果`s`包含未区分大小写的字符，或者如果结果字符的 Unicode 类别不是“Lu”(字母，大写)，而是“Lt”(字母，标题大小写)，则`s.upper().isupper()`可能是`False`。

Unicode 标准的第 3.13 节描述了所使用的大写算法。

`str.**zfill**` ( *宽度*)

返回用 ASCII 码`'0'`数字填充的字符串的副本，以形成长度*宽度*的字符串。前导符号前缀(`'+'` / `'-'`)通过在符号字符之后而不是之前插入填充符*来处理。如果*宽度*小于或等于`len(s)`，则返回原字符串。*

例如:

>>>

```
**>>>** "42".zfill(5)
'00042'
**>>>** "-42".zfill(5)
'-0042'
```

# `printf`-样式字符串格式

> 注意
> 
> 这里描述的格式化操作展示了导致许多常见错误的各种怪癖(比如无法正确显示元组和字典)。使用更新的[格式的字符串文字](https://docs.python.org/3/reference/lexical_analysis.html#f-strings)、`[str.format()](https://docs.python.org/3/library/stdtypes.html#str.format)`接口或者[模板字符串](https://docs.python.org/3/library/string.html#template-strings)可能有助于避免这些错误。这些备选方案中的每一个都提供了它们自己的简单性、灵活性和/或可扩展性的折衷和好处。

String 对象有一个独特的内置操作:`%`操作符(模)。这也被称为字符串*格式化*或*插值*操作符。给定`format % values`(其中*格式*为字符串)，*格式*中的`%`转换规范被替换为*值*的零个或多个元素。其效果类似于在 C 语言中使用`sprintf()`。

如果*格式*需要单个参数，则*值*可以是单个非元组对象。 [5](https://docs.python.org/3/library/stdtypes.html#id16) 否则， *values* 必须是一个元组，其项数正好由格式字符串指定，或者是一个映射对象(例如字典)。

转换说明符包含两个或更多字符，并具有以下组件，这些组件必须按此顺序出现:

1.  `'%'`字符，它标志着说明符的开始。
2.  映射键(可选)，由带括号的字符序列组成(例如，`(somename)`)。
3.  转换标志(可选)，它影响某些转换类型的结果。
4.  最小字段宽度(可选)。如果指定为`'*'`(星号)，则从*值*中元组的下一个元素读取实际宽度，要转换的对象在最小字段宽度和可选精度之后。
5.  精度(可选)，给定为`'.'`(点)，后跟精度。如果指定为`'*'`(星号)，实际精度从*值*中元组的下一个元素读取，要转换的值在精度之后
6.  长度修饰符(可选)。
7.  转换类型。

当右边的参数是一个字典(或其他映射类型)时，那么字符串*中的格式必须*包括一个带括号的映射键，该映射键插入到紧接在`'%'`字符之后的字典中。映射键从映射中选择要格式化的值。