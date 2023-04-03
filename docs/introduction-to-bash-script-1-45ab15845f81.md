# Bash 脚本简介— 1

> 原文：<https://medium.com/analytics-vidhya/introduction-to-bash-script-1-45ab15845f81?source=collection_archive---------35----------------------->

![](img/24c521ff56c5b1d49c14d00b61f521b2.png)

# **内容**

1.  bash shell 是什么？
2.  可变的
3.  如果语句

# 什么是 Bash 脚本？

**Bash (Bourne Again Shell)** 是一种处理命令的解释器。然后，脚本可以帮助我们将**许多命令**放到一个文件中并执行它。在本文中，我们将展示 bash 脚本的一些基本概念。

在开始之前，我们先进入一些我们经常使用的 shell 命令。粗略地说，这些命令可以写成:

```
# option is for some special function
# param is the arguments that user type in
commands [-option name]... param1 param2
```

A.`echo`

```
echo "Hello"
> Hello
```

B.`cd`

`cd`代表**变更目录**。有两种方法可以使用该命令。

*   **绝对路径**

```
# Change to /usr/test/hello/bash/practice
cd /usr/test/hello/bash/practice
```

*   **相对路径**

如果目标在某个相对路径中，我们可以只指定那个路径。

```
# Switch to AVL dir
cd AVL
```

C.`rm`

`rm`代表**删除**，它可以删除文件。

```
# remove the test.sh
rm test.sh
```

D.`ls`

`ls`表示列表，可以列出目录里面的东西。

```
# List thing inside the dir
ls
```

在 bash 脚本中， **#用于注释。**

# 可变的

有时我们需要**存储用户提示的结果**，因此，我们需要学习如何声明变量。通常，变量的名称应该全部是大写的**。这里是我们声明的方式:**

```
# There is no space between the equal sign
[variable name]=[result]
```

**现在，如果我们想把结果显示在屏幕上，我们该怎么做呢？嗯，**呼应**现在派上用场了！但是我们要注意，我们要回显**变量**，因此，我们要在**名称**前加上 **$** 。**

```
# This is how we did
TOTAL_SUM=5# This is how we show the result
echo $TOTAL_SUM
​
# Or more specific
echo ${TOTAL_SUM}
​
# Even more specific
echo "${TOTAL_SUM}"
```

**在我们结束这个主题之前，我们称之为尝试将所有这些内容写入脚本！**

```
#!/bin/bash
​
TOTAL_SUM=5
echo "${TOTAL_SUM}"
```

**太棒了，我们全都做了！**

**等等，我们如何执行这个文件，第一行是什么？**

*   ****射棒****

**第一行不是为人类写的，是为解释者写的。解释器必须计算出**加载哪个程序**来执行文件。在这里，我们使用 **/bin/bash** 意味着我们要使用 **/bin** 中的 bash 程序来执行文件。还有，如果想用 python，可以指定 **/bin/python** 。**

**另外， **#** 是**尖**和！就是**砰**。的确，这就是 shebang 名字的由来。注意， **bash** 的路径可能与计算机不同，您可以通过键入`which bash`来检查路径。**

*   ****权限****

**如果我们想运行文件，我们可以使用使用`./[filename]`**

```
# Execute test.sh
./test.sh
​
> permission denied: ./test.sh
```

**还有一个错误。该错误告诉我们无法执行该文件，因此我们必须通过`chmod`更改权限:**

```
# Change the permission
chmod u+x test.sh
```

**我不会进入**许可**和 **chmod** 。到目前为止，我们可以成功地执行它！**

# **如果语句**

**我们通常需要在某种情况下决定下一步做什么。因此，我们需要 **if** 语句来帮助我们！以下是它的实现方式:**

```
if [[ statement ]] # White space before and after the statement
then
  commands...
fi
```

**这句话有很多特殊的用法:**

```
# For compare# "=" used to check string type, also >, <, >=, <= for string type
if [[ ${TEST} = "ABC"]] # -eq used to check numeric type
​​if [[ ${TEST} -eq 2]] # -nq -> not equal for numeric type
​if [[ ${TEST} -ne 2]] # -le -> large or equal for numeric type
​if [[ ${TEST} -le 2]] 
​
# For file management
​
PATH="./Hello"
if [[ -d ${PATH} ]] # PATH is exist and is directory
​
FILE="./test.sh"
if [[ -e ${FILE} ]] # FILE is exist and is file
​
if [[ -r ${FILE} ]] # FILE exist and read permission is granted
​
if [[ -w ${FILE} ]] # FIEL exist and write permission is granted
```

**在这里，我只是提到了其中的一部分。如果你想要一些特殊的功能，你可以在谷歌上搜索并使用模板来达到你的目标！**

**现在，我们将了解如何使用 **if-else** 和 **if-elif-else*** 语句:**

```
# if-else
if [[ statement ]]
then
  commands...
else
  commands...
fi
​
​
# if-elif-else
if [[ statement ]]
then
  commands...
elif [[ statement ]]
then
  commands...
else
  commands...
fi
```

**对于 **if-elif-else** 语句，允许实现多个 **elif** 。**

# **摘要**

**今天，我们讨论 bash 脚本的一些基本概念。我们将在下一部作品中探讨更多主题。如果有什么不对或者让你困惑的地方，请告诉我！**