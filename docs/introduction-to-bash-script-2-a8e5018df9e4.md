# Bash 脚本简介— 2

> 原文：<https://medium.com/analytics-vidhya/introduction-to-bash-script-2-a8e5018df9e4?source=collection_archive---------30----------------------->

![](img/6a85cb44990f4a2cd0433d256fb5e496.png)

# 内容

1.  **案例陈述**
2.  **While 循环**
3.  **为循环**

# 选择语句

Case 语句是检查条件的另一种方式，下面是要做的模板:

```
case [expression] in
  pattern1)
    commands...
    ;;
  pattern2)
    commands...
    ;;
  ...
esac
```

这里有一个例子:

```
#!/bin/bashTEST=3 # Assign 3 to TESTcase ${TEST} in
  1)
    echo "TEST is 1."
    ;;
  2)
    echo "TEST is 2."
    ;;
  3)
    echo "TEST is 3."
    ;;
esac
```

正如我们所料，结果是:

```
> TEST is 3
```

到目前为止，我们已经学会了如何使用 case。但是，对于那些了解 C 语言的人来说，类 C 语言有一个**默认的**选项来捕捉异常。事实上，在 **bash 脚本**中有一个对应的脚本，但是使用了**通配符**。

```
case [expression] in
  pattern1)
    commands...
    ;;
  pattern2)
    commands...
    ;;
  ...
  * ) # using wildcards to catch the option not be specified
    commands
    ;;
esac
```

让我们看一个例子:

```
#!/bin/bash

TEST=5 # Assign 3 to TEST

case ${TEST} in
  1)
    echo "TEST is 1."
    ;;
  2)
    echo "TEST is 2."
    ;;
  3)
    echo "TEST is 3."
    ;;
  *)
    echo "Get the exception."
    ;;
esac
```

结果是:

```
> Get the exception
```

* can 代表任何长度的字符。我们将在以后的工作中更多地讨论**通配符**。

# While 循环

现在如果我们说我们有任务想要反复做，我们可以用循环来实现。一般来说，有两种实现循环的方法——对于的**和对于**的**。**

现在，我们停下来一会儿，然后再进入下一会儿！知道如何使用操作符很重要，因为我们可能想在循环中使用 counter。我们的语法是这样的:

```
$((param1 + param2)) # *, /, - are all the same
```

请注意，我们应该使用(( ))来阻止操作，之后，我们使它成为一个变量！这里有一些提示，请记住如何使用，即使它可能不是事实。首先(( ))是操作的语法，但是对于程序加载器，它将把这种形式视为字符串，因此添加$可以帮助它更具体。还有其他人来做操作，比如 [expr](https://vitux.com/how-to-do-basic-math-in-linux-command-line/) ， [let](http://www.geekpills.com/operating-system/linux/bash-shell-built-in-let-command) 。

回到我们的话题，下面是做 **while 循环**的模板:

```
while [[ condition ]] # Have space before and after condition
do
  commands...
done
```

例如，我们想重复“你好”10000 次，我们可以这样做:

```
#!/bin/bash# Count the times
COUNTER=0while [[ ${COUNTER} -ne 10000 ]] # Remember -ne is used for numeric
do
  echo "Hello"
  COUNTER=${{ $COUNTER + 1)) # Doing operation
done
```

# For 循环

我们首先讨论如何在 bash 脚本中使用该命令。在正常情况下，我们不能在 bash 脚本中直接使用命令，我们应该使用一个括号，并把它作为一个变量使用。它看起来是这样的:

```
$(commands)
```

假设我们想在 bash 脚本中使用`ls`并将其打印出来，我们应该使用:

```
echo $(ls)
```

接下来我们继续我们的话题。**为命令**可用于取出序列中的元素或检查条件。我们先说前者。

*   **序列**

```
for [element] in [sequence]
do
  commands...
done
```

假设我们想要打印出文件夹中的文件，我们可以使用 for 来实现它。首先我们必须使用`ls`来获取文件夹下的文件。：

```
#!/bin/bashFILES=$(ls) # Save the result to FILESfor file in $FILES   # file is the element inside the FILES
do
  echo "$file"  # Notice that file is still a variable
done
```

*   **表情**

我们可以通过使用不同的语法将的**视为**而**。以下是模板:**

```
for (( init; condition; Do after ))
do
  commands...
done
```

对于熟悉 C-like 的人来说，更容易记住。首先，我们需要一个变量作为计数器，这是 **init** 部分。在执行命令之前，循环的**要检查条件是否满足。如果不是，则结束，否则，您必须修改计数器，这是**部分之后的**部分。让我们使用 **for 循环打印出 hello 10000！****

```
#!/bin/bash#There're some different inside for
# 1\. Can use "<" directly in the condition part
# 2\. "COUNTER++" means "COUNTER = COUNTER + 1"
for (( COUNTER=0; COUNTER < 10000; COUNTER++ ))
do
  echo "Hello"
done
```

# 摘要

今天我们讨论如何使用**格**、**而**和**格用于**。这些都是我们在进入下一个话题之前要知道的基本的东西。我们还介绍了在 shell scipt 中执行操作和使用命令的一些方法。

如果有任何问题，请让我知道。谢谢你的阅读！