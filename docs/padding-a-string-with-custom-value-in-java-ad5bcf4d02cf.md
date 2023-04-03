# 在 Java 中用自定义值填充字符串

> 原文：<https://medium.com/analytics-vidhya/padding-a-string-with-custom-value-in-java-ad5bcf4d02cf?source=collection_archive---------11----------------------->

![](img/8d726fa489198a288acf735991f495b4.png)

Java 字符串— Edureka

在这篇短文中，我们将探索在 Java 中填充字符串的不同方法。我们将创建用默认值(空格)填充的方法，以及允许我们用自定义字符填充字符串的方法。Java 中的 String 类没有现成的填充方法，所以我们将创建不同的方法，并使用不同的库来填充 Java 中的字符串。

**用默认值填充:**我们将创建一个重载方法，用默认字符或自定义字符填充。第一个填充方法将包含初始文本和所需长度，同时将第二个填充方法的自定义填充字符设置为空格字符。

默认填充方法:

```
public static String pad(String text, int len){
    return *pad*(text,len,' ');
}
```

**用自定义字符填充:**这个方法包含一个额外的字符参数，允许我们指定用什么字符填充我们的字符串。

```
public static String pad(String text, int len, char value){
    StringBuilder sb = new StringBuilder();
    if(text.length()<len){
        sb.append(text);
        for(int i=text.length();i<len;i++){
            sb.append(value);
        }
        return sb.toString();
    }
    return text;
}
```

如果您从上面的代码片段中注意到，我们将新的字符集填充到初始字符串的左侧。让我们创建一个类似的从左到右填充的方法。

**从左边填充:**下面的代码片段类似于默认的填充方法，但是它从左边填充默认值。

```
public static String padLeft(String text, int len) {
    return *padLeft*(text,len,' ');
}
```

**从左侧填充自定义字符:**从左侧填充一组自定义字符。

```
public static String padLeft(String text, int len, char value){
    StringBuilder sb = new StringBuilder();
    if(text.length()<len){
        for(int i=text.length();i<len;i++){
            sb.append(value);
        }
        sb.append(text);
        return sb.toString();
    }
    return text;
}
```

# 使用字符串库填充

Apache Commons Lang: 这个库有几个 Java 实用程序类。其中一个类是 StringUtils 类，它有很多操作和格式化字符串的方法。在使用这个库之前，您需要在 pom.xml 文件中添加依赖项。

```
<dependency>
 <groupId>org.apache.commons</groupId>
 <artifactId>commons-lang3</artifactId>
 <version>3.8.1</version>
</dependency>
```

根据 apache-commons 文档，我们可以像上面的例子一样用空格或不同的字符进行左填充。以下是文档中的示例方法。

[**left pad**](http://commons.apache.org/proper/commons-lang/apidocs/org/apache/commons/lang3/StringUtils.html#leftPad-java.lang.String-int-)([**String**](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html?is-external=true)str，int size)用空格填充字符串。

```
StringUtils.leftPad(“bat”, 3) = “bat”
StringUtils.leftPad(“bat”, 5) = “ bat”
```

[**left pad**](http://commons.apache.org/proper/commons-lang/apidocs/org/apache/commons/lang3/StringUtils.html#leftPad-java.lang.String-int-char-)([**String**](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html?is-external=true)str，int size，char padChar)用指定字符填充字符串。

```
StringUtils.leftPad("bat", 3, 'z')  = "bat"
StringUtils.leftPad("bat", 5, 'z')  = "zzbat"
```

[**left pad**](http://commons.apache.org/proper/commons-lang/apidocs/org/apache/commons/lang3/StringUtils.html#leftPad-java.lang.String-int-java.lang.String-)([**String**](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html?is-external=true)str，int size，[**String**](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html?is-external=true)padStr)用指定的字符串填充字符串。

```
StringUtils.leftPad("bat", 5, "yz")  = "yzbat"
 StringUtils.leftPad("bat", 8, "yz")  = "yzyzybat"
```

[**r**](http://commons.apache.org/proper/commons-lang/apidocs/org/apache/commons/lang3/StringUtils.html#leftPad-java.lang.String-int-)**ightPad**([**String**](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html?is-external=true)str，int size)向右填充一个带空格的字符串。

```
StringUtils.rightPad(“bat”, 3) = “bat”
StringUtils.rightPad(“bat”, 5) = “bat  ”
```

[**r**](http://commons.apache.org/proper/commons-lang/apidocs/org/apache/commons/lang3/StringUtils.html#leftPad-java.lang.String-int-)**ightPad**([**String**](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html?is-external=true)str，int size，char padChar)用指定字符向右填充字符串。

```
StringUtils.rightPad("bat", 3, 'z')  = "bat"
StringUtils.rightPad("bat", 5, 'z')  = "batzz"
```

**右** [**填充**](http://commons.apache.org/proper/commons-lang/apidocs/org/apache/commons/lang3/StringUtils.html#leftPad-java.lang.String-int-java.lang.String-)([**String**](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html?is-external=true)str，int size，[**String**](https://docs.oracle.com/javase/7/docs/api/java/lang/String.html?is-external=true)padStr)用指定的字符串向右填充一个字符串。

```
StringUtils.rightPad("bat", 5, "yz")  = "batyz"
StringUtils.rightPad("bat", 8, "yz")  = "batyzyzy"
```

我相信，通过这篇文章，我们已经能够涵盖填充字符串的不同方法，通过自己实现不同的方法，并使用 Java 中最流行的库之一来填充字符串。