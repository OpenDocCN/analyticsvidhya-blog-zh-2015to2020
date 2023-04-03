# Python 正则表达式

> 原文：<https://medium.com/analytics-vidhya/python-regex-5852186dcbad?source=collection_archive---------19----------------------->

## 找到匹配的两种方法

RegEx 或正则表达式是一种小型语言，使用字符串模式在字符串中搜索一个或多个子字符串。

![](img/a663f635ce8798875d424bf13e7977ba.png)

照片由[praewhida K](https://unsplash.com/@pkvoyage?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

导入 re 模块后，我们可以使用四种方法进行查询。

*   匹配()
*   搜索()
*   findall()
*   finditer()

所有这四个方法都可以通过两种方式调用，在模块级调用，或者从编译后的模式对象中调用:

*   模块级函数

```
import restring = "Lorem ipsum dolor sit amet, consectetur adipiscing elit"
pattern = r'\wo'
res = re.search(pattern, string)
```

*   汇编/编译

```
import restring = "Lorem ipsum dolor sit amet, consectetur adipiscing elit"
pattern = re.compile(r'\wo')
res = pattern.search(string)
```

这两者的区别在两个方面:

*   如何定义模式
*   如何调用 search()函数

# 模块级函数

![](img/b0178859d570d3ae071ae9e0fedee160.png)

照片由[贝恩德·克鲁奇](https://unsplash.com/@bk71?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

以 search()方法为例，我们可以直接从 re 模块调用它:re.search()

```
re.search(r'\wo', "Lorem ipsum dolor sit amet")
```

第一个参数是模式，可以是文字模式，也可以是编译模式。第二个参数是要搜索的字符串。当然，这两个变量可以用上面例子中的变量代替。

# 汇编

![](img/cfdeb06eca11b66c95a647a7bd6cf44a.png)

Marc Zimmer 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

编译正则表达式需要首先用 compile()方法创建一个模式。然后使用该模式调用 search()，例如。

```
import repattern = re.compile(r'\wo', re.IGNORECASE)
res = pattern.search("Lorem ipsum dolor sit amet, consectetur adipiscing elit")
```

使用 compile()方法创建模式的好处之一是可以提供一个或多个标志来优化搜索过程，这是第二个参数。

*   关于。答/答美国信息交换标准代码
*   关于。S / re。DOTALL
*   关于。我/ re。IGNORECASE
*   关于。L / re。现场
*   关于。M / re。多线
*   关于。X / re。冗长的

查看 [Python 文档](https://docs.python.org/3/howto/regex.html#use-string-methods)了解这些标志的更多定义

您可以使用 pipe: re 设置多个标志。我|re。x，忽略大小写并设置 verbose/comments。

就我个人而言，我更喜欢使用编译方法，因为当我使用 re.search()时，我总是忘记两个参数的位置，而且我喜欢先设置模式，这样更方便地使用模式来调用 search(0 方法。

# 正则表达式的其他一些注意事项

![](img/30e546502f80892fc7b752c0775289b5.png)

凯尔·格伦在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

## 关于 match()、search()、findall()、finditer()

*   match()和 search()将返回一个匹配，而 findall()和 finditer()将返回所有匹配。
*   match()和 search()将返回一个 re。Match 对象，而 findall()返回一个列表，finditer()返回 iterable。
*   match()将只查找字符串开头的匹配项，而 search 将搜索整个字符串。

## 使用原始字符串

正如开始提到的，regex 是 python 中的一种迷你语言，python 解释字符串的方式与 regex 略有不同，最好使用原始字符串来创建模式。

## 一个特殊字符“？”

*   不管有没有

```
p = re.compile(r'\d\d\d-?\d\d\d\d')   #with or without "-"
p = re.compile(r'(\d{2})?d')          # with or without (\d{2})
```

*   匹配子模式但不要捕捉它。((?:模式)模式)

```
>>> p1 = re.compile(r'(\we)+')
>>> p2 = re.compile(r'(?:\we)+')
>>> m1 = p1.search("references")
>>> m2 = p2.search("references")
>>> m1.groups()
('re',)
>>> m2.groups()
()
```

*   命名匹配子串。(?P <name>图案)</name>

```
>>> p = re.compile(r'name is (?P<name>\w+)(\.| )')
>>> m = p.search('Hi, my name is Jack.')
>>> m.group("name")
'Jack'>>> p = re.compile(r'am (?P<fname>\w+) (?P<lname>\w+)(\.| )')
>>> m = p.search('Hi, I am Jack London.')
>>> m.groupdict()
{'fname': 'Jack', 'lname': 'London'}
```

ps:那个？后面是大写的 P

*   前瞻断言。有还是没有

```
(?!...)  # not have it
(?=...)  # have itre.compile(r"""
    .*              # zero or more characters
    [.]             # with a "."
    (?!exe$)        # should not be end with exe after the ".", this is not pattern but a look ahead assertion
    [^.]*$          # zero or more not '.' characters after the "."
""", re.X)
```

## re 的方法。匹配对象

*   。group()，。第一组。组()，。groupdict()

。group() ==。组(0)

```
>>> p = re.compile(r"c(o)?(a)t")
>>> m = p.search('a cat in coat')
>>> m1 = p.search("a coat on cat")
>>> print(m.group(), m.groups())
>>> print(m1.group(), m1.groups())
>>> print(p.findall('a cat in coat'))cat (None, 'a')          #(o)? returns None ...
coat ('o', 'a')
[('', 'a'), ('o', 'a')]  #(o)? returns "" ...
```

*   。span()，。start()，。结束()

```
>>> p = re.compile(r"c(o)?(a)t")
>>> m = p.search('a cat in coat')
>>> print(m.span(), m.start(), m.end())(2, 5) 2 5
```

今天到此为止。