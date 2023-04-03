# Bash 脚本简介— 4

> 原文：<https://medium.com/analytics-vidhya/introduction-to-bash-script-4-e1466a917272?source=collection_archive---------31----------------------->

介绍通配符和调试工具。

![](img/ac892e7fffb32ca68504eddada5a3cf6.png)

# 内容

1.  **通配符**
2.  **调试工具**

# **通配符**

通配符是用于**模式匹配的字符或字符串。**假设有 10 个你想删除的 txt 文件，但你不想一个一个删除，那么通配符可以轻松处理！

*   *****

***** 可以匹配零个或多个字符。例如，我们使用`a.*`来查找那些名为**和带有任意扩展名的**的文件。下面是有多个文件的目录，名字用 a 和 b。

```
A.c    A.csv  A.js   A.txt  b.cpp  b.java b.py
A.cpp  A.java A.py   b.c    b.csv  b.js   b.txt
```

如果我们只想查找 **A 文件，**那么我们可以键入:

```
ls A.*
```

输出将是:

```
A.c A.cpp A.csv A.java A.js A.py A.txt
```

让我们试试另一个:

```
ls A.j*
```

输出:

```
A.java A.js
```

*   ?

？仅匹配**一个**字符。让我们看一些例子:

```
ls ?.txt
```

输出:

```
A.txt b.txt
```

并且:

```
ls ??.txt
```

输出:

```
no matches found: ??.txt
```

这是因为我们没有用两个字符命名的文件！

*   []

[]是一个字符类，它匹配括号中的任何字符。

```
ls [Ab].txt
```

输出:

```
A.txt b.txt
```

另一个:

```
ls A.[js][js]
```

输出:

```
A.[js][js]
```

让我们看看它是如何工作的:

1.  首先**【js】**匹配字符 **j**
2.  第二个**【js】**匹配字符 **s**

我们也可以添加！紧接在[之后，以排除括号内的字符。

```
ls [!c].txt
```

输出:

```
A.txt b.txt
```

这里背后的概念是我们想找到那些**不是**以 c 开头的文件。

此外，我们可以使用连字符来表示一系列字符或数字:

```
ls [a-c].txt
```

输出:

```
b.txt
```

从给定的结果中，我们可以知道 **[]** 是区分大小写的。

让我们创建一个名为 1 的新文件。

```
ls [1–8].txt
```

输出:

```
1.txt
```

*   **人物类**

1.  [[:alpha:]] = >匹配字母
2.  [[:digit:]] = >匹配数字
3.  [[:album:]] = >匹配字母和数字
4.  [[:lower:]] = >匹配小写
5.  [[:upper:]] = >匹配大写
6.  [[:blank:]] = >匹配空格和制表符

到目前为止，我们已经讨论了如何使用通配符，我们也可以将通配符添加到 bash 脚本中。回想一下，我们使用 ***** 来匹配 case 语句中的默认参数。

# **调试工具**

调试真的是一件很烦的工作！所以我们引入一些内置的调试工具和其他资源来帮助我们。

*   **-x**

x 将在命令执行之前打印它们。有两种实现方式。

1.  **开始使用**

我们可以在 shebang 后面加上-x，让整个程序进入-x 模式:

```
#!/bin/bash -xTEST=1
echo "${TEST}"
```

输出:

```
./test.sh+ TEST=1
+ echo 1
1
```

开头的“+”表示将要执行的命令。我们期望首先声明并打印出**测试**变量。1 是测试的输出。

2.**使用设置**

我们还可以使用`set -x`来启动-x 模式，使用`set +x`来停止，这对于用户来说更加灵活。

```
#!/bin/bashTEST=1
echo "${TEST}"set -x
echo "${TEST}"
set +xecho "${TEST}"
```

输出:

```
1
+ echo 1
1
+ set +x
1
```

在`set -x`之前，我们只能看到输出。之后，我们可以知道将要执行哪些命令。使用`set +x`停止模式。请注意，`set +x`也会显示，因为`set +x`在完成之前不会被执行！

*   **-e**

当错误发生时，我们可以使用-e 来停止程序。让我们看看传统版本:

```
#!/bin/bash$(lj)echo "Hello"
```

输出:

```
./test.sh./test.sh: line 3: lj: command not found
Hello
```

这里，`echo`即使发生了一些错误，仍然会被执行，让我们用`e`来改变它:

```
#!/bin/bash -e$(lj)echo "Hello"
```

输出:

```
./test.sh./test.sh: line 3: lj: command not found
```

它会如我们所料停在 3 号线！

*   **-v**

v 将打印外壳读取的内容！

```
#!/bin/bash -vecho "Hello"
```

输出:

```
#!/bin/bash -vecho "Hello"
Hello
```

它打印出结果并自己发出命令！

*   **组合**

我们可以在 shell 中组合多种模式，例如-ex 或-ev。

```
#!/bin/bash -execho "Hello"$(lf)echo "Hello"
```

输出:

```
./test.sh
+ echo Hello
Hello
++ lf
./test.sh: line 5: lf: command not found
```

它将显示将要执行的命令，并在错误发生时停止。

*   **专用工具**

有一个可以帮助我们处理语法错误的工具叫做[**shellcheck**](https://github.com/koalaman/shellcheck)**。**

# **总结**

到目前为止，我们已经学习了 bash 脚本的基本知识，这是关于 Bash 脚本简介的最后一篇文章。感谢阅读，如有问题请写评论！

# **参考**

1.  [Shell 脚本:探索如何自动化命令行任务](https://www.udemy.com/course/shell-scripting-linux/)