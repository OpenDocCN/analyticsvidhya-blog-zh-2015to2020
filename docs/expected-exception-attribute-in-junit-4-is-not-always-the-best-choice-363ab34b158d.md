# JUnit 4 中的预期异常属性并不总是最佳选择

> 原文：<https://medium.com/analytics-vidhya/expected-exception-attribute-in-junit-4-is-not-always-the-best-choice-363ab34b158d?source=collection_archive---------8----------------------->

![](img/81f746c957b136efbeaef425802f4f82.png)

Justin Veenema 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

我考虑过给它起一个 clickbait 标题，类似于“JUnit 4 中的预期异常属性很糟糕，千万不要使用它！”

然而，事实是，`@Test`注释的`expected`属性有时会很有用。但大多数时候，还有更好的选择。

假设您正在使用 JUnit 4.12 为一个 Java 程序开发一个测试类。假设你从来没有为一个在给定情况下抛出特定异常的单元编写过测试。

如果您正在使用 NetBeans 开始您的测试类样板文件，您可能非常了解 JUnit 的`Assert.fail()`。因为不管其他什么都无法通过测试，所以你可以把它放在程序执行的一个给定分支上，以便让它更有用。

对于这类事情，我最常用的例子是在`Fraction`类中被零除。例如，试图用 3/4 除以 7/2，结果应该是 3/14。但是任何被 0 除的分数都会导致某种运行时异常。

所以你可能会这样写:

```
 @Test
    public void testDivisionByZero() {
        Fraction dividend = new Fraction(3, 4);
        Fraction zero = new Fraction(0, 1);
        **try** {
            Fraction result = **dividend.divides(zero)**;
            **fail**(dividend.toString() + " divided by 0 gave result "
                                               + result.toString());
        } **catch (ArithmeticException ae)** {
            System.out.println("\"" + ae.getMessage() + "\"");
        } catch (Exception e) {
            **fail**("Expected ArithmeticException but got " +
                                   e.getClass().getName());
    }
```

所以如果`divides()`给出一个不正确的结果(任何数字`Fraction`都可以代表)，或者如果它抛出一个不相关的异常(比如`PrinterException`)，这个测试应该会失败。

但是您可能会担心这不是 JUnit 的惯用用法。所以你可能会求助于谷歌。如果你正在使用 IntelliJ，你可能会很快转向谷歌。

然后 Google 会告诉你关于`@Test`注释的`expected`属性，它将异常的类作为它的参数。被零除测试将被改写如下:

```
 **@Test(expected = ArithmeticException.class)**
    public void testDivisionByZero() {
        Fraction dividend = new Fraction(3, 4);
        Fraction zero = new Fraction(0, 1);
        Fraction result = dividend.divides(zero);
        System.out.println(dividend.toString() +
                 "divided by 0 is said to be " + result.toString());
    }
```

这看起来干净多了，而且它的工作方式与之前的版本非常相似:如果`divides()`给出不正确的结果或者抛出除`ArithmeticException`之外的任何异常，测试就会失败。

另请注意，它会将`divides()`可能给出的任何错误结果打印到控制台。如果您正在进行测试驱动的开发，您可能有一个`divides()`存根，它给出的结果是 0，而不管实际涉及的数字。

但是这个版本的测试没有打印出`ArithmeticException`消息。这是一个小缺陷，除非您有兴趣对消息做出任何断言(比如它不是空的`String`)。

如果你认为两个或三个不同的例外是有效的呢？例如，在被零除的情况下，我认为`IllegalArgumentException`会是更好的选择，但是`ArithmeticException`是可以接受的。

对我来说，`ArithmeticException`提出了一个可以用更多资源解决的问题。例如，如果`Math.addExact()`溢出了一个`int`，它抛出`ArithmeticException`，这表明也许你应该使用一个`long`或者一个`BigInteger`来代替一个`int`。

我认为，被零除通常是由程序员的愚蠢错误引起的。这种愚蠢的错误通常不需要切换到其他数据类型就可以解决。

TestNG 中的`@Test`注释有一个属性来指定应该发生两个或更多异常中的一个。它还有一个属性来指定异常消息应该是什么。

但是在 JUnit 中，如果想要捕获多个有效异常中的任何一个并测试消息，就需要 try-catch-fail 构造。

更罕见的是，您可能希望断言某个异常包装了其他异常。那么您的测试过程需要捕捉包装异常，然后使用`getCause()`来检查包装异常。

另一种选择是使用 JUnit 的`ExpectedException`类(带有`@Rule`注释),然后使用`expectCause()`过程。下面是来自 JUnit 文档的例子[:](https://junit.org/junit4/javadoc/4.12/org/junit/rules/ExpectedException.html)

```
 @Test
    public void throwsExceptionWhoseCauseCompliesWithMatcher() {
        NullPointerException **expectedCause** = new
                                             NullPointerException();
        **thrown.expectCause(is(expectedCause));**
        throw new IllegalArgumentException("What happened?", cause);
    }
```

您的测试类应该导入`org.junit.rules.ExpectedException`和`org.junit.Rule`注释。所以你还需要这个:

```
 @Rule
    public ExpectedException thrown = ExpectedException.none();
```

这是从 JUnit 文档中逐字复制的，除了增加了一个空格。事实证明，如果您还没有导入`org.hamcrest.core.Is`，您也需要导入。

下面是我的被零除测试使用`expectCause()`的样子:

```
 @Test
    public void testDivisionByZero() {
        ArithmeticException **expectedCause** = new
                                              ArithmeticException();
        **thrown.expectCause(Is.is(expectedCause));**
        Fraction dividend = new Fraction(3, 4);
        Fraction zero = new Fraction(0, 1);
        Fraction result = dividend.divides(zero);
        System.out.println(dividend.toString() +
                " divided by 0 is said to be " + result.toString());
    }
```

我不得不写“`Is.is`”，因为我没有静态导入`is()`。没什么大不了的。

然而，另一方面，让测试中的`ArithmeticException`与来自`Fraction.divides()`的`ArithmeticException`相匹配变得有点令人头疼。

不同之处在于，在 try-catch-fail 中，我只会在`getCause()`上写一个`instanceof`检查，而 Hamcrest 匹配器可能希望异常完全相同，具有相同的消息和相同的堆栈跟踪。

`ExpectedException`也有`expectMessage()`，可能比`expectCause()`对你有用得多。我认为在这种情况下包装异常是不值得的，我更关心异常消息。

因此，通过静态导入`org.hamcrest.core.StringStartsWith.startsWith`,我这样重写了测试:

```
 @Test
    public void testDivisionByZero() {
        **thrown.expectMessage(startsWith("Dividing 3/4 by 0"));**
        Fraction dividend = new Fraction(3, 4);
        Fraction zero = new Fraction(0, 1);
        Fraction result = dividend.divides(zero);
        System.out.println(dividend.toString() +
                " divided by 0 is said to be " + result.toString());
    }
```

首先让它失败，然后让它通过要容易得多。

因为您已经从 JUnit JAR 中导入了`@Test`注释和各种断言，并且还引入了 Hamcrest(您可以通过查看测试库来使用 NetBeans 项目窗格验证这一点)，所以您也可以使用`ExpectedException`。

但是，与经典的“尝试-捕捉-失败”相比，额外的努力值得吗？这是一个你必须自己回答的问题。

我想你会同意我的观点，无论是经典的 try-catch-fail 还是`ExpectedException`类，在大多数情况下都比`@Test`注释的`expected`属性有用得多。

最后，我不建议同时使用`expected`属性和`ExpectedException`类。由于相互冲突的期望，这很可能导致误导性的测试失败。