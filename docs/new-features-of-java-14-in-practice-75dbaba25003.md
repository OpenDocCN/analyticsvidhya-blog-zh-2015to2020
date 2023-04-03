# Java 14 在实践中的新特性

> 原文：<https://medium.com/analytics-vidhya/new-features-of-java-14-in-practice-75dbaba25003?source=collection_archive---------19----------------------->

![](img/f295de9bc75da8c43ea169efb221af94.png)

照片由[阿兹哈鲁尔·伊斯拉姆](https://unsplash.com/@azhar93?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/coding?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

最近发布的 JDK 14 总共带来了 16 项主要增强功能。在本文中，我将带您浏览最有趣的部分，大多与语言支持有关。

**1。**
记录(预览功能)可能是一个新版本中最令人印象深刻的部分。这是 Java 中一种新的类型声明。

```
public record User(long id, String name) {}
```

只用一行代码我们就得到了一个新的 final 类，它的所有字段都是 final。在编译时，`record`会自动生成样板文件`constructors`、`public get`、`equals()`、`hashCode()`、`toString()`。不会有任何设定者，因为所有字段都是最终的。

我们可以开始使用新用户`record`,就像我们习惯使用类一样:

```
public static void main(String[] args) {
    User user = new User(1L, "Mark");
    user.id();
    user.name();
}
```

记录作为数据载体，对于支持开箱即用的不变性有严格的限制。

它们不能是抽象的，扩展任何其他类，甚至它的隐式超类。这是因为`record’s` API 是由它维护的状态定义的，不允许它的组成类修改它。

出于同样的原因,`record`不能有本地方法声明，因为它不能依赖外部逻辑。

```
public native String getSystemTime(); //compilation error

static {
    System.*loadLibrary*("nativedatetimeutils");
}
```

一个`record`不能显式声明实例字段或者有 setter 方法。只有记录的标题定义了记录值的状态。这就是下面的代码无法编译的原因:

```
void setId(long id){
    this.id = id; //compilation error
}
```

一个`record`可以使用您最喜欢的 java 库之一，比如 Gson 或 Jackson，序列化成 JSON:

```
Gson gson = new Gson();

String userJson = gson.toJson(user);
System.*out*.println(userJson); //outputs {"id":1,"name":"Mark"}
```

并反序列化回来:

```
User newUser = gson.fromJson(userJson, User.class);
System.*out*.println(newUser); //outputs User[id=1, name=Mark]
```

一个更有趣的事实是每个`record`的超类是记录本身:

```
Class<?> superclass = user.getClass().getSuperclass();
System.*out*.println(superclass); //class java.lang.Record
```

不幸的是，`records`不能用作轻量级持久域对象，每个对象代表关系数据库中的一个表。这是因为 JPA 需要一个记录没有的无参数构造函数。JPA provider 必须使用反射来获取和设置实体的字段，这也是不可能的，因为记录的字段是最终的。所以下面的代码不会编译:

```
@Entity
public record User(long id, String name) { //compilation error

}
```

您将需要继续使用旧的已知龙目语:

```
@Getter
@NoArgsConstructor
@AllArgsConstructor
public class Product {
    @Id
    @GeneratedValue(strategy = GenerationType.*IDENTITY*)
    private Long id;

    @Size(max = 40)
    private String name;
}
```

**2。模式匹配为** `instanceof` **【预览】**

在 Java 14 之前，如果你想做`instanceof`检查，然后把对象转换成一个变量，你应该做以下事情:

```
if (obj instanceof User) {
    String s = (User) obj;
    // use s
}
```

在 Java 14 中，这可以简化为一行:

```
if (obj instanceof String s) {
    // can use s here
}
```

尽管在大多数情况下您可能会尝试一个新特性，但是您仍然应该避免在产品代码中使用`instanceof`。多态性的良好使用应该优先于条件的基本使用。

**3。文本块(第二次预览)**

这个特性给 Java 语言增加了`text blocks`。文本块是由多行组成的字符串文字。使用`text blocks`有助于避免大多数转义序列、字符串连接。总的来说，它简化了编写程序的过程，使得用几行源代码来表达字符串变得容易。

Java 14 之前:

```
String someHtml = "<html>\n" +
        "   <body>\n" +
        "      <p>Hello World</p>\n" +
        "   </body>\n" +
        "</html>\n";
```

使用 Java 14:

```
String java14 = """
                <html>
                    <body>
                        <p>Hello World</p>
                    </body>
                </html>
        """;
```

**4。开关表达式**

在 Java 12 和 Java 13 中，`Switch Expressions`是一个预览特性，从 Java 14 开始，它已经成为一个标准的语言特性。新开关可以用作使用箭头语法的表达式:

```
switch (article.state()) {
    case *DRAFT* -> System.*out*.println(1);
    case *PUBLISHED* -> System.*out*.println(2);
    case *UNKNOWN* -> System.*out*.println(3);
}
```

Java 14 之前:

```
switch (article.state()) {
    case *DRAFT*:
        System.*out*.println(1);
        break;
    case *PUBLISHED*:
        System.*out*.println(2);
        break;
    case *UNKNOWN*:
        System.*out*.println(3);
        break;
}
```

现在可以产生/返回值了

```
int result = switch (article.state()) {
    case *DRAFT* -> 6;
    case *PUBLISHED* -> 7;
    case *UNKNOWN* -> 8;
};
```

Java 14 之前:

```
int result;
switch (article.state()) {
    case *DRAFT*:
        result = 6;
        break;
    case *PUBLISHED*:
        result = 7;
        break;
    case *UNKNOWN*:
        result = 8;
        break;
    default:
        throw new UnsupportedOperationException();
}
```

**5。有用的 NullPointerExceptions**

这个特性将有助于更快地跟踪和解决 JVM 产生的 NullPointerExceptions。在 Java 14 之前，这些消息根本不提供信息:

```
public static void main(String[] args) {
    User user = new User(1L, null);

    System.*out*.println(*toUpperCase*(user));
}

private static String toUpperCase(User user) {
    return user.name().toUpperCase(); // produces NPE
}
```

该消息将类似于:

*演示时线程“main”Java . lang . nullpointerexception
出现异常。Main.main(Main.java:16)"*

通过新的增强功能，将“XX:+ShowCodeDetailsInExceptionMessages”添加到虚拟机选项将使消息看起来像:

*线程“main”中出现异常 java.lang.NullPointerException:无法调用“String.toUpperCase()”，因为“test。在演示时，User.name()"为空
。演示时的 main . toupper case(main . Java:20)
。Main.main(Main.java:16)*

## 摘要

Java 14 最新特性的概述到此结束。现在，您已经掌握了 Java 世界中最新特性的知识和实践经验。我希望你喜欢这篇文章，并发现它很有用。坚持练习，因为明天的战斗是在今天的练习中赢得的！