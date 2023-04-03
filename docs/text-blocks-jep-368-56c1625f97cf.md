# 文本块(JEP 368)

> 原文：<https://medium.com/analytics-vidhya/text-blocks-jep-368-56c1625f97cf?source=collection_archive---------16----------------------->

![](img/123f8a20998bb966380b2674713b5170.png)

日航·马哈尔

在 Java 中编写多行字符串不是一件有趣的事情，你必须编写代码，要求把 HTML、JSON、XML 或 SQL 查询作为字符串放入你的代码中。

请看下面的例子，我们试图把 JSON 作为字符串

```
String sampleJson = "{\"name\":\"josh\",\"salary\":\"8000$\",\"age\":27,\"isMarried\":\"True\"}";
```

这里我们有一个非常简单的 JSON，有很多转义序列，很难读懂。

我们可以做得更好，使其可读，见下文

```
String sampleJsonReadableABit = "{" +
        "\"name\":\"josh\"," +
        "\"salary\":\"8000$\"," +
        "\"age\":27," +
        "\"isMarried\":\"True\""+
        "}";
```

它比早期的要好，但是仍然难以阅读(大量的连接和转义序列)并且容易出错。添加或删除另一个属性对任何开发人员来说都不容易。

引入文本块的主要目的是

*   易于编写多行字符串
*   删除转义序列并增强可读性
*   智能地处理空白和缩进

## 什么是文本块？

文本块作为预览功能在 2019 年 6 月与 JDK 13 一起发布，基于反馈，这将作为 JEP 368 与 JDK 14 再次预览。

这个特性被开发人员忽略了很长时间，特别是当有人不得不阅读嵌入了 JSON 文档或 SQL 语句的代码时。

正如定义" ***文本块是一个多行字符串文字，它避免了对大多数转义序列的需要，以可预测的方式自动格式化字符串，并在需要时让开发人员控制格式*** "。

文本块用三个双引号**“”**定义，作为开始和结束分隔符。

让我们看看如何通过文本块呈现上述 JSON

```
String jsonAsTextBlocks =  """
        {
        "name":"josh",
        "salary":"8000$",
        "age":27,
        "isMarried":"True"
        }
         """;
```

读书不是更好吗？没有转义序列:)

像 Kotlin、Scala 或 Groovy 这样的语言都有这个特性。

## 要点

*   在内容真正开始之前，开始分隔符后面必须跟一个行结束符，尝试编写如下所示的文本块将会出现编译时错误。

```
String textBlockSample = """Ashish //Compile time error
        """;String textBlockSampleTwo = """Ashish"""; //Compile time errorString textBlockSampleThree = """"""; //Compile time errorString textBlockSampleFour = """ """; //Compile time error
```

#在第一个示例中，内容放在没有行结束符的开始分隔符“”之后。这是不允许的，将会是一个编译时错误。

#在第二个示例中，内容也直接放置在开始分隔符""和结束分隔符"""下，没有行结束符(通常使用字符串我们喜欢类似于***String str = " a shish "****的样式，但是这种样式不允许用于文本块*)。这也是不允许的，将会是一个编译时错误。

编译器会忽略开始分隔符和行结束符之间的空格，因为内容只会在行结束符之后开始。

*   **结束分隔符**，下面提到的例子中没有这样的规则有效。

```
String textBlockSample = """ //valid syntax
        Ashish""";String textBlockSampleTwo = """ //valid syntax
        Ashish
        """;String textBlockSampleThree = """ //valid syntax (empty)
        """;
```

*   **附带空白，**文本块区分巧妙区分附带空白和必要空白。Java 编译器会自动去掉附带的空格。

```
String htmlAsTextBlocks = """
        <html>
           <body>
                <h1>
                Sample Heading
                </h1>
           </body>
        </html>
             """;
```

如果您运行上面的代码片段，输出将如下所示，标记之前的任何空格都将被视为附带空格，并被编译器简单地删除，但下面标有***……***的空格不是附带空格，此类空格将被视为预期/所需/必要的空格，并被编译器考虑。

```
<html>
...<body>
........<h1>
........Sample Heading
........</h1>
...</body>
</html>
```

如果您想在内容之前显式添加一些空格，可以使用结束分隔符 **"""** 来控制内容，请注意下面的代码片段，这里的结束分隔符已被显式向左移动。

```
String htmlAsTextBlocksWithSpace = """
          <html>
              <body>
                  <h1>
                  Sample Heading
                  </h1>
              </body>
          </html>
""";
```

它将有如下输出:

```
 <html>
              <body>
                  <h1>
                  Sample Heading
                  </h1>
              </body>
          </html>
```

在这里，前导空格由编译器考虑的结束分隔符标记。注意结束定界符有控制前导空白的作用，但是它不能对尾随空白产生任何影响，例如，如果上面的代码像下面这样改变

```
String htmlAsTextBlocksWithSpace = """
        <html>
            <body>
                <h1>
                Sample Heading
                </h1>
            </body>
        </html>
                                          """;
```

它会有如下输出

```
<html>
   <body>
       <h1>
       Sample Heading
       </h1>
   </body>
</html>
```

要特别添加一些尾随空白，您可以这样做:

```
String textBlockWithTrailingSpace = """
        Ashish    \040""";
```

如果你计算上面的文本块长度，它将是 11。

## 文本块和字符串之间的相似性

*   很大程度上，我们可以在任何字符串用法适用的地方使用文本块，例如，如果任何方法定义可以将字符串变量作为参数，我们可以在那里传递文本块。

```
private static void textBlocksAsString(String str){
    System.*out*.println(str);
} 
```

以上方法可以调用为"***textBlocksAsString(" "
我是 Fun
" ")；***

这是完全有效的语法。

*   传统的字符串值和文本块都被编译成相同的类型:字符串。字节码类文件不区分字符串值是从传统字符串还是文本块中派生出来的。这意味着文本块值像任何普通字符串一样存储在字符串池中。**如果==和 equals 引用完全相同的值，它们将返回 true** 。参见下面的代码片段

```
String str = "Ashish";
String textBlock = """
        Ashish""";
```

如果你运行一些东西

```
System.*out*.println(str==textBlock);
System.*out*.println(str.equals(textBlock));
```

这两条语句都将打印出 **true** ，但是如果您稍微修改一下上面的代码:

```
String str = "Ashish";
String textBlock = """
        Ashish
        """;
```

==和 equals 都将返回 false。

*   您可以像连接两个字符串一样连接字符串和文本块，如下所示

```
System.*out*.println( """
    Works
    is fun
    """
        +
        "Let's have some");
```

*   对于某些业务需求，我们必须编写代码，其中字符串需要根据某些正则表达式进行拆分并转换为列表。这种用例也通过文本块得到了简化

```
String fruits = "Apple,Orange,Mango";
List<String> fruitsList = new ArrayList<>();
Collections.*addAll*(fruitsList,fruits.split(","));//Such operation will be simplified like below
String fruitsTextBlocks = """
        Apple
        Orange
        Banana
        """;
List<String> lisOfFruits = fruitsTextBlocks.lines().collect(Collectors.*toList*());
```

## 编译步骤

编译器分三步处理文本块:

1.  行结束符— Windows 和 Linux 有不同的行结束符，Windows 使用回车和换行符(" \r\n ")，而 Linux 只使用换行符(" \n ")。为了避免源代码将一个操作系统转移到另一个操作系统时出现任何问题，文本块将行尾规范化为\u000a。规范化期间不解释转义序列\n (LF)、\f (FF)和\r (CR)。转义处理稍后发生。
2.  移除附带的空白-以上步骤进行标准化，在此步骤中，所有附带的空白将被移除，如上所述。算法不解释转义序列\b(退格)和\t (tab );转义处理稍后发生。
3.  解释转义序列—如果您的文本块有任何转义序列，它将被立即解释。文本块支持字符串中支持的所有转义序列，包括\n，\t，\ '，\ "，和\\。

## 换码顺序

*   转义序列**的处理**

**在文本块中有转义序列是完全有效的，见下文**

```
String textBlockWithES = """
    When your "work" speaks for yourself don't Interrupt
    """;
```

**但是，如果您试图将它放在靠近结束分隔符的位置，如下所示:**

```
String textBlockWithESI = """
    When your "work" speaks for yourself don't "Interrupt"""";
```

**这将是一个编译时错误，有两种方法可以解决:**

**->将结束分隔符带入下一行**

```
String textBlockWithESI = """
    When your "work" speaks for yourself don't "Interrupt"
    """;
```

**->使用转义**

```
String textBlockWithESI = """
    When your "work" speaks for yourself don't "Interrupt\"""";
```

**此转义字符\可以放在上一行的最后四个字符("""")中的任何一个之前。**

*   **文本块嵌入了另一个文本块**

**即使您尝试在文本块中除了开始和结束分隔符之外的任何位置放置三元组""，您也已经放置了转义符，否则将会出现编译时错误:**

```
String textBlockWithESII = """
    When your "work" \""" speaks for yourself don't "Interrupt\"""";
//Above is a valid Text Block
```

**在类似行中，如果你需要在一个文本块中嵌入另一个文本块，你必须使用转义符**

```
String textBlockWithAnotherEmbedded =
        """
        String text = \"""
            A text block inside a text block
        ""\";
        """;
```

****新逃脱序列****

1.  **一个很常见的实践/需求是，将一个长字符串分成多个子字符串，用“+”将它们连接起来，以保持代码的可读性，但输出将是单行。为了处理在 java 中添加的类似需求 n **ew 转义序列\** ,这只适用于 TextBlocks。**

```
String quoteAsStr = "Fearlessness is like a muscle. " +
        "I know from my own life that the more I exercise it, " +
        "the more natural it becomes to not let my fears run me.";

System.*out*.println(quoteAsStr);String quoteAsTextBlocks = """
        Fearlessness is like a muscle. \
        I know from my own life that the more I exercise it, \
        the more natural it becomes to not let my fears run me.""";System.*out*.println(quoteAsTextBlocks);
```

**如果您执行上述代码，两个打印输出将完全相同**

```
Fearlessness is like a muscle. I know from my own life that the more I exercise it, the more natural it becomes to not let my fears run me.Fearlessness is like a muscle. I know from my own life that the more I exercise it, the more natural it becomes to not let my fears run me.
```

**2.添加了这个 JEP 的另一个新的转义序列 **\s** ，就是简单地翻译成一个空格(\u0020)。转义序列直到事件空格剥离后才被转换，因此 **\s** 可以作为栅栏来防止尾部空格的剥离。这个转义序列适用于字符串和文本块。**

```
String colors = """
        red  \s""";
System.*out*.println(colors.length());
```

**在控制台输出将打印 6。**

## **新方法**

*   **string::striping dent()**

**如上所述，编译器会从文本块中删除附带空格，这种方法会添加到字符串中，以获得与普通字符串相同结果。**

```
String fruitsStr = "red\n   green\n   blue";System.*out*.println(fruitsStr);
//**1-** It will print 
//red
//   green
//   blueSystem.*out*.println(fruitsStr.replace(" ","."));
//**2**- It will print
//red
//...green
//...blueSystem.*out*.println(fruitsStr.stripIndent());
//**3**- It will print
//red
//   green
//   blue
```

**在上面的代码片段中，您可以观察到 stripIndent()方法没有影响，因为编译器没有发现附带的空白，但是如果您有如下所示的字符串:**

```
String fruitsStrWithLeadingSpace = "  red\n   green\n   blue";System.*out*.println(fruitsStrWithLeadingSpace);
//**1**- It will print
//  red
//   green
//   blueSystem.*out*.println(fruitsStrWithLeadingSpace.replace(" ","."));
//**2**- It will print
//..red
//...green
//...blueSystem.*out*.println(fruitsStrWithLeadingSpace.stripIndent());
//**3**- It will print
//red
// green
// blue
```

**在这里，您可以看到第二个和第三个控制台打印之间的前导空格已从字符串中删除。**

*   **String::translateEscapes()**

**顾名思义，这将翻译字符串中的转义序列，见下面的例子**

```
String fruitsStr = "red\n   green\\n   \"blue\"";
System.*out*.println(fruitsStr);
//**1-** It will print
//red
//   green\n   "blue"System.*out*.println(fruitsStr.translateEscapes());
//**2-** And after translateEscape, it will print
//red
//   green
//   "blue"
```

**您可以在 **2-** 控制台打印中观察到新行字符已被翻译。**

*   **String::格式化(Object… args)**

```
String jsonAsTextBlocks =  """
{
"name":"%s"
}
 """.formatted("Ashish");System.*out*.println(jsonAsTextBlocks);
//It will print
//{
//"name":"Ashish"
//}
```

**它工作方式非常类似于字符串。**