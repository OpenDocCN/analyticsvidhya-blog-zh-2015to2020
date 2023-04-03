# 正则表达式入门

> 原文：<https://medium.com/analytics-vidhya/getting-started-with-regular-expressions-1ccdb8a6ca98?source=collection_archive---------20----------------------->

正则表达式(或“regex”甚至“regexp”)是搜索和修改文本的强大工具。这个想法非常简单:通过一个正则表达式，你可以组装一些代表不同元素(字符、数字、标点、重复)的特定乐高积木来形成一个模式。然后对输入使用这种模式并检索匹配。

![](img/ec7dc405160a44a0478d8b7593e179e8.png)

[ **乐高**比克斯做重物。照片由[詹姆斯·庞德](https://unsplash.com/@jamesponddotco?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

在匹配的位置和方式上，您有多种选择:在行首、解析每个单词、在单词内部等等。您可以检索您匹配的片段——通过捕获组的方式，我们稍后会用到——并选择对您的匹配执行特定的操作，例如将所有匹配的“\”更改为“/”或相反。

在一篇提倡正则知识的[名篇](https://www.theguardian.com/technology/2012/dec/04/ict-teach-kids-regular-expressions)、**科里·多克托罗**中写道:

> 正则表达式是现代软件的基本组成部分。它们以某种形式存在于每一个现代操作系统中。文字处理器、搜索引擎、博客程序……尽管几十年前为普通人设计的软件就已经假定用户知道正则表达式，但它们仍然潜伏在我们使用的几乎每一个环境中。

我们将在 *Python* 中执行我们的正则表达式测试，并且我们将从学术论文中提取书目数据。如果你急于开始，跳过以下简短的历史部分。

# 正则表达式的显著特征

正则表达式是由克莱尼——没错，就是那个[克莱尼](https://en.wikipedia.org/wiki/Stephen_Cole_Kleene)——在 50 年代早期，当他描述[正则语言](https://en.wikipedia.org/wiki/Regular_language)时构思出来的。后来，它们被实现为 Unix 文本处理实用程序。所有高级的书呆子文本编辑器(Vim，Vi，Grep，sed，AWK 和 co)都有。

多克托罗邀请学校系统向孩子们教授 regex 的文章已经被提及。没有提到的是激发数字人文学科研究和培养技术素养的最佳段落之一:

> 你所接触的大部分世界，从提款机到你的银行网站，到你申请残疾津贴的网站，到早上叫醒你的闹钟，再到追踪你的位置、社交网络和个人想法的手机，都是由软件支撑的。至少，对软件工作的粗略了解将有助于你判断软件在你生活中的可靠性、质量和实用性。

# 通用匹配表达式

正则表达式本身是一种小型语言。你有代表其他东西的短符号。我们用这个正则表达式乐高来建立一个模式。然后，我们将模式与给定的文本进行匹配。(通常情况下，它们有不同的风格和不同的实现)。

正则表达式起初看起来很难，但是你所需要的[只是一点耐心，就像阿克塞尔说的](https://www.youtube.com/watch?v=ErvgV4P6Fzc)。大多数乐高积木都是以一个\开头，后面跟着一些东西。

我们大部分的乐高积木都是两个角色组合。以下是一些最常用的:

*   你想匹配任何一个*数字*吗？这意味着一个从 0 到 9 的数字。使用 **\d**
*   是否要定义一个要匹配的*自定义范围类*？使用方括号并将范围限制放入其中，用破折号(-)分隔。如果你想匹配从 1 到 7 的数字，选择**【1–7】**。如果您想要匹配任何数字，并且不喜欢我们上面使用的\d 语法，请选择[0–9]
*   如果你想匹配任何字符，使用点**。**
*   如果您想要匹配某个模式的一个或多个实例，请使用星号 *****
*   如果想让某个组件可选，加个**？**后构件。这是一个表示零或一的量词
*   如果要匹配空格，请使用 **\s**
*   括号' **(** ' *)是语言的符号*，实际上它们定义了*捕获群*——见后文。如果你想匹配一个括号，你必须使用正则表达式转义符，即\
*   如果你想匹配任何不是空白的字符，你可以使用 **\S** (这样你就可以捕捉大写和小写的字母、数字和标点符号)
*   如果要匹配一个单词字符，使用 **\w** (单词字符包括大写和不大写的字母*和*数字*和*下划线字符)

# 从人文学科的实践中学习:匹配书目模式

我们的任务是识别论文中的书目数据。我们假设论文是作为一些文本输入给出的。我们的任务是为不同类型的书目开发正则表达式，特别是:

*   作者(日期)；
*   (作者，日期)；
*   (作者日期)；
*   完整数据(大约。):作者分隔符标题分隔符出版分隔符日期。

我们希望能够匹配这些不同的风格，我们的目标是这些输入的标准化。对于这些，我们将提取作者和日期。为此我们需要捕捉组。

# 捕获组

我们希望将我们在不同风格中找到的引用标准化为更简单和通用的内容，比如作者日期。没有花哨的括号或类似的。这种标准化可以用来对大多数引用或有影响的论文或任何你想要的东西进行分析。

我们需要分离一些信息，这样我们就可以比较采用不同风格的不同期刊的参考书目。

为了实现标准化，我们可以使用**捕获组**。它们是允许我们检索匹配内容的一部分的组(这里是:作者和日期)。要将捕获组添加到正则表达式中，我们需要做的就是用括号来分隔捕获组。在 Python 中，组是作为列表中的项目来访问的。

假设我们有一个像 **regexbookchicago** 这样的正则表达式，它捕获芝加哥风格的书籍。如果我们从中挑选出两个捕获组，我们可以用 list[0]和 list[1]来访问它们。

```
**import** re
*#mock regex matching chicago style*
regexbookchicago = '(authorpart) title (date part)'
text = 'Some text with Chicago style references'
*#store a list of lists of all our matches*
match_pattern_to_text = re.findall(regexbookchicago, text)
**for** match **in** match_pattern_to_text:
  print('Match found: ', match_pattern_to_text[0] + ' ', match_pattern_to_text[1])
```

# 匹配作者(日期)

先说匹配作者(日期)。我们的基本任务是构建 regex-LEGO 砖块来匹配作者和日期。一个日期是非常简单的，它只是一个 4 位数的块(我们忽略引用的东西 3 位数，如 Giustiniano 529 或处理 2003a 或类似)。

约会无非是这样的:

```
**import** re
dateregex = '\d\d\d\d'
```

我们增加了一个以文本元素为特征的样本测试。我们将在这个测试中匹配日期，然后打印结果。

```
**import** re
dateregex = '\d\d\d\d'
sampletext = 'The biggest contribution to the field is due to Master (2001) and its impac cannot be denied'
match = re.findall(dateregex,sampletext)
print(match)
```

运行这个，看看我们是否能捕捉到“2001”。

现在，我们需要添加作者，只不过是一个姓，即一个大写字母后跟一些非大写字母。为了捕捉作者，我们基本上需要一个范围为[a-z]的大写字母，后跟范围为[A-Z]的任意数量的字母。

第一种实现是:

**author1 = '[A-Z][a-z]*'**

请注意，这与

**author2 = '[a-zA-Z]* '。**

事实上，author1 要求在匹配的开始有一个大写字母 in；作者 2 没有。它将匹配连续的大写和非大写字母的字符串，包括这里的这个。(请注意，author2 如果写成“[A-Za-z]*”，可能会导致某些在线编译器出错。)

您也可以将作者实现为:

**author3 = '\S*'** 解决资本化问题。请注意，在这里你也将得到所有的文字进入文本标点符号包括在内。

要排除标点符号，请:

**author4 = '\w* '。**

下面的代码模板允许您尝试并理解捕捉作者的各种方法。将上面不同的作者表达式代入 authorregex 变量，并尝试只匹配主**的**。

```
**import** reauthorregex = 'INSERT ONE OF THE AUTHOR ABOVE'
sampletext = 'The biggest contribution to the field is due to Master (2001). Its impact cannot be denied, on pain of miSbeHaving.'
match = re.findall(authorregex,sampletext)
print(match)
```

如你所见，我们正在超越我们的文本。根据所用的表达方式，我们将得到以大写字母开头的任何东西(“the”、“Its”)等等。

![](img/b906627a0166b055887814ae1035328b.png)

[挂在墙上的 42 条毛巾……照片由[丹尼·米勒](https://unsplash.com/@redaquamedia?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄]

**不要慌，我们在正确的轨道上**。我们需要将我们已经确定的两个要素联系起来。是什么以独特的方式将它们联系在一起？

第一，括号在日期左右，所以我们要把它们加到日期上(记得转义)。然后，是中间的空间。结果代码如下。我选择“[A-Z][a-z]*”来匹配作者，因为这样更容易看到我们要求匹配的内容。

```
**import** re
authordateregex = '[A-Z][a-z]*\s\(\d\d\d\d\)'
sampletext = 'The biggest contribution to the field is due to Master (2001). Its impact cannot be denied, on pain of miSbeHaving.'
match = re.findall(authordateregex,sampletext)
print(match)
```

运行这个，开心点，我们要拿到‘Master(2001)’。

现在是时候用捕获组隔离元素了。我们想要隔离作者和日期，不带括号。看看你是否能把这两组加上括号匹配起来。

这里是放置它们的位置:

```
**import** re
capturingauthordate = '([A-Z][a-z]*)\s\((\d\d\d\d)\)'
sampletext = 'The biggest contribution to the field is due to Master (2001). Its impact cannot be denied, on pain of miSbeHaving.'
capturingmatch = re.findall(capturingauthordate,sampletext)
print(capturingmatch[0] + ' ', capturingmatch[1])
```

如果你的文本变大了，那么你必须打开 capturingmatch 项，这将是一个所有匹配的列表。事实上，每个作者(日期)都呈现为['作者'，'日期']。

# 匹配(作者日期)和(作者，日期)

这里的情况开始变得很糟糕。首先，我们必须选择是否希望同一个正则表达式执行两个匹配。似乎第二个表达式(作者，日期)只不过是第一个表达式(作者日期)加了一个逗号。这是一个使用可选匹配的好机会(上面有零个或一个量词，例如'？').

然而，如果我们沿着这条路走下去，我们必须意识到，并不是我们上面看到的所有等价的作者选项都仍然有效。事实上，如果我们将作者与' \S '匹配，我们将在(作者，日期)中捕捉到作者后面的逗号作为作者姓名的一部分。所以它将在作者捕获组中。

这个表达式的好处在于，我们需要捕捉的东西都在括号之间，所以不需要匹配这两个部分，然后找到连接它们的方法。

下面是匹配表达式的代码，带有可选的括号。我们新的正则表达式叫做*capturing parents*，示例文本现在包括(作者日期)和(作者，日期)。

```
**import** re
capturingparenthesis = '\(([A-Z][a-z]*),?\s(\d\d\d\d)\)'
sampletext = 'The biggest contribution to the field is due to (Master 2001). Nonetheless, (Slave, 2002) is probably a more accessible version of these ideas.'
capturingmatch = re.findall(capturingparenthesis,sampletext)
print(capturingmatch[0] + ' ', capturingmatch[1])
```

# 完整数据(敌人)

完整数据是我们任务中最糟糕的部分。不仅不同项目(文章、期刊等)的风格不同。);通常你不会在最后得到一个参考书目来有一个简化的工具来检查准确性。

对我们来说不好的特征是，我们关心的信息彼此相距甚远。作者在开头的某个地方，但是日期在结尾，被一大堆乱七八糟的东西包围着，比如标题、卷的编辑、期刊的期次、期刊名称等等。每一个因素都增加了我们猜测和挑选方案的复杂性。

好好想想，想办法解决。这里有一个尝试:

```
fulldatatextsample = 'Murphy, "Was Hobbes a Legal Positivist?," Ethics (1995)'
regexfulldata= '([A-Z][a-z]*),\s"[A-Za-z\s?,"]*\((\d\d\d\d)\)'
```

在这里起重要作用的是下面这个词。，"] 。事实上，我们知道如何获取名称和年份，困难的部分是获取标题的 mre 模块部分。作品的标题将包括字母(大写和非大写)和空格，因为标题通常由更多的单词组成。我们还需要捕捉特殊的分隔符，比如双引号。

请注意，这也将捕获日志的名称。其实我们也不知道题目会有多少字。该脚本的其余实现与之前相同，因此您可以使用它

# 问题和限制

我们找不到包含括号的标题。我们不能包含数字。更好的尝试是将 fulldata 字符串解析为:

1.  作者
2.  标题
3.  日记/书/随便什么
4.  年份。

我们需要使用合适的分隔符来匹配日志的样式。

# 如何构建正则表达式

我发现一点一点地构建正则表达式很有用。所以，在这种情况下，我试着分离出作者和日期。这有助于了解正则表达式是如何失败的，比如匹配除作者姓氏之外的任何大写单词。如果需要更复杂的东西，比如为了更好地实现 fulldata，可以尝试反过来:首先匹配字符串的大部分，然后减少匹配的部分。

# 开放的问题

我们只是触及了表面，但仍然表明我们在这里有一个强大的工具。一些限制是:

*   我们没有在日期中捕获 a，b，c:我们如何处理呢？
*   我们不考虑像冯·芬特尔、冯·赖特、范·德·托雷、德·雷、德·塞这样的两部分姓氏。那么德拉斯·卡萨斯呢？(你可能会说这不是问题，因为我们将获得最后一个姓氏)
*   有时候我们会多次引用同一作者的话，比如《大师》(2001，2002，2003，2004)(不，那不是费德勒温网系列的样本)。我们该如何应对？
*   作者(日期)很酷，但也可能变坏。我们的实现无法获得 Master (2001，2002)以及 Master(2001:112–121)。有办法解决吗？
*   合著论文呢？我们将致力于“作者和作者”以及“作者等”。

# 还要做/轮到你了/下一步是什么/练习

*   开发日期也应与 2003a 或类似日期一致
*   编写程序来获取文本中的内部引用。这些引用可以是对一本书中的章节的交叉引用，或者是对法律语料库中的文章和法律的引用，甚至更多。

这有趣吗？你可以在 Linkedin 上随意联系，或者在 T2 的 Twitter 上参与更广泛的对话。

这项工作是作为中科院院士计划的一部分在中科院开展的。点击了解更多关于 T4 的信息。