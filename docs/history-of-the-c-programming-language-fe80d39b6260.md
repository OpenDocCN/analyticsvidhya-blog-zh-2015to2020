# c 编程语言的历史

> 原文：<https://medium.com/analytics-vidhya/history-of-the-c-programming-language-fe80d39b6260?source=collection_archive---------20----------------------->

# C 语言的创造

编程语言是丹尼斯·里奇在 1971 年创造的。它源于由肯·汤普森在 1969 年创造的`B`编程语言。B 编程语言源于马丁·理查兹在 1966 年创造的`BCPL`编程语言。

所有这些语言都受到了这样或那样的影响，1960 年诞生的`Algol 60`编程语言。

C 编程语言与 BCPL 和 B 有相似之处，但主要区别在于它是一种类型化语言。

从 1971 年到 1974 年，C 语言的参考书籍有:

*   [C 参考手册](https://www.bell-labs.com/usr/dmr/www/cman.pdf)，1974 年，丹尼斯·m·里奇。
*   [C 语言编程教程](https://www.math.utah.edu/computing/compilers/c/Kernighan-CTutorial.pdf)，1974 年，布莱恩·w·柯尼根。

# 经典，或传统，或 k&R，C

**后来**，对编程语言做了一些改动。例如，添加了新的类型，如:`short`和`union`，一个用于造型的符号:`(type-name) expression`，以及使用:`typedef`操作符为新类型创建名称的能力。这个版本的 C，被称为经典或传统或`k&R` C。

在 1976 - 1983 年间`c`编程语言的**参考文献**是:

*   [《C 程序设计语言》第一版](https://archive.org/details/TheCProgrammingLanguageFirstEdition)，1978 年，布莱恩·克尼根和丹尼斯·里奇。

一些**句法**的变化，存在于`C` 1974 和经典`C`之间，比如在初始化一个变量时。

```
int y 3;
/* Initialize a variable in C 1974 */int y = 3;
/* Initialize a variable in C 1978 */
```

还有一些**语义**的变化，存在于`C` 1974 和经典`C`之间，例如:

```
y=-2; 
/* Decrement y by 2 , in C 1974 */y=-2;
/* Assign -2 to y , in C 1978 */
```

参考书中还描述了一个用于输入和输出的**便携式库**。

# ANSI C 或 C89 或 C90

之后，对语言做了进一步的修改，比如添加了`enum`类型和`const`限定词，以及一些语义上的修改。因此，有必要对 C 编程语言进行标准化。

1983 年，美国国家标准机构成立了`X3J11`委员会。该委员会的目标是标准化 C 编程语言。

1989 年，该委员会发布了`ANSI C 1989`标准，被国际标准化组织 ISO 采用为 ISO/IEC 9899:1990。

Ansi C 引入了一种新的方法来声明函数，旧的方法仍然有一些限制。

例如，在经典 C 语言中，函数可以声明如下:

```
name (argument list, if any)
argument declarations, if any
{
declarations and statements, if any
}
```

而在 ANSI C 中，函数被声明为:

```
return-type function-name(argument declarations) {
    declarations and statements
}
```

ANSI C 还改变了十六进制和八进制整数文字的**语义**，它们现在首先被认为是 int 类型，而不是`unsigned int`类型。

ANSI C，也引入了**标准 C 库**，其中包含了一组可移植的头文件。头声明了一些函数，例如输入和输出函数的声明，并定义了一些宏和类型。

ISO/IEC 9899:1990 `C`标准的**草案**，可以在这里找到[。](https://port70.net/~nsz/c/c89/c89-draft.html)

# 后来的修订

ISO/IEC 9899:1990 `C`标准发布后，**标准化工作**继续进行。这些标准化努力，只是对`C`标准的修订。

这些修订版包含对标准 C 库的一些添加或删除，例如，在`c11`修订版中删除了`gets`函数。

或者它们只是对 C 语言的一些句法或语义上的改变，比如在`c11`标准中增加了一些关键字，比如`_Atomic`关键字。

修订内容如下:

*   C95 : ISO/IEC 9899:1990/AMD1:1995 修订版。
*   ISO/IEC 9899:1999 草案。
*   C11 : [ISO/IEC 9899:2011 草案](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)。
*   C17 : [ISO/IEC 9899:2018 草案](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2310.pdf)。
*   C2x : [当前工作草案](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2573.pdf)。

所有版本的 C，或多或少都是向前兼容的。

# C 语言的目标

一个 C 程序，被编译成机器语言。它是可移植的，只要你坚持使用 C 标准和 C 标准库。

所以 C 编程语言是可移植的，你可以直接控制机器，在你的程序和计算机之间没有虚拟机或解释器，因此`C`语言用于**系统编程**。

当 C 语言被创造出来时，它是 Unix 操作系统的编程语言。

# 其他参考文献

*   b 编程语言[参考](https://www.bell-labs.com/usr/dmr/www/bintro.html)。
*   BCPL 编程语言[参考](https://www.bell-labs.com/usr/dmr/www/bcpl.html)。

*原载于 2020 年 11 月 7 日 https://twiserandom.com*[](https://twiserandom.com/c/history-of-the-c-programming-language/)**。**