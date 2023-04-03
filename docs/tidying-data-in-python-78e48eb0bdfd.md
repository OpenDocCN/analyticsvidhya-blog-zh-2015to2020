# 使用 Python 整理数据

> 原文：<https://medium.com/analytics-vidhya/tidying-data-in-python-78e48eb0bdfd?source=collection_archive---------3----------------------->

![](img/18b59649f98639310b72647a7ca72066.png)

米卡·鲍梅斯特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

这篇文章是我之前写的[凌乱数据博客](/@kimrodrikwa/untidy-data-a90b6e3ebe4c)的延续。要了解什么是不整洁的数据，请看看帖子。概括地说，如果数据具有以下属性，则称其为整洁的:

1.  每个变量形成一列。
2.  *每个观察值形成一行。*
3.  *每种类型的观测单元形成一张表。*

这里我要整理一下我在那篇帖子里介绍的一个杂乱无章的数据。我将使用 Python 来做这件事，更具体地说是[熊猫库](https://pandas.pydata.org/)。然后，我将评估最终的“整洁数据”，以检查它是否符合这些要求。

**原始凌乱数据**

杂乱的数据。例如，its 既有学生观察单元，也有表现观察单元

关于完整的解决方案，请看这个 [github 库](https://github.com/Kimmirikwa/tidy-data)。

我首先打算进口熊猫

导入 python 作为其通用别名

将数据读入 [pandas dataframe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) 并获取数据的列。下面的代码实现了这一点。

将 csv 文件读入数据帧并检查列

现在我们有了数据框架，我们可以使用强大的 pandas 技术来整理数据。如前所述，该数据存在不整洁的数据问题。我现在将讨论这些问题，并讨论它们的修复方法。以下是前一篇文章中解释的数据问题:

1.  *有些列标题是值，而不是变量名*
2.  *多个变量存储在一列*
3.  *变量存储在行和列中*
4.  *多种类型的观测单位存储在同一个表中*

为了讨论修复，我不会按照上述问题的顺序。我将解决所有这些问题，首先将数据分成不同的表(针对学生和成绩)并整理每个数据集。

*   *多种类型的观测单位存储在同一个表中*

该数据包含了**学生**和**表现**的数据。我要做的第一件事是把这些不同的观察单位分开，让每个单位都有自己的表格。列 *id* 、*姓名*、*电话*和*性别和年龄*将出现在**学生**表中。*测试号*、*第一项*、*第二项*和*第三项*将出现在性能表中。 *id* 列也将被添加到性能表中，以标识性能的学生。

我们初始化两个数据帧 *student_df* 和 *performance_df* ，它们具有来自原始数据帧的相关列。

将原始数据集拆分为学生数据集和成绩数据集。这解决了上面的问题 4。

学生 df 现在看起来如下所示

学生表中的数据可能是在收集学生的详细信息时记录的

*performance_df* 如下图

成绩表有可能在考试后记录的数据

可以清楚地看到，数据帧现在包含不同的观察单位。几乎可以肯定，学生表中的数据是在注册学生时记录的。对于性能表，很明显，这些数据是在检查后记录的，尽管分散在一段时间内。这并不意味着不可能收集例如学生的性别和分数作为一个观察。例如，在进入一所学校之前做一次考试之后，这是可能的。然而，对于一个学生来说，在三个学期的多次考试中出现这种情况的几率非常低。

*   *多个变量存储在一列*

*student_df* 中的**性别和年龄**列包含性别和年龄数据。提取每个学生的**性别**和**年龄**并添加到相关列。原来的**性别和年龄**列是从数据帧中删除的。这是通过使用数据框的[应用](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html)和[下降](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)功能实现的。

提取性别和年龄，然后将它们放在不同的列中。然后删除原始的组合列。

学生表现在看起来如下所示。

整洁的学生桌子

从上表中可以看出，

1.  id、姓名、电话、性别和年龄都是构成表格列的变量。
2.  *每个学生数据都是一个观察值，形成一行。*
3.  *该表只有一种观察单位，即针对学生的观察单位*

现在我们有了一个整洁的学生数据，注意力将转移到成绩数据上。

*   *有些列标题是值，不是变量名*

**第 1 项**、**第 2 项**和**第 3 项**是数值，但用作列标题。这是通过使用熊猫[融化](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html)修复的。

熔化性能数据框架

在 melted dataset**term 1**、 **term 2** 和 **term 3** 列现在在 **term** 列中，它们以前的值在**标记**列中。

熔融性能数据

术语编号是从术语值中提取的。

提取术语编号

*   *多个变量存储在一列*

**测试 1** 和**测试 2** 是不同的变量，但存储在**测试号**栏中。**测试 1** 和**测试 2** 列添加了各自的标记作为数值。

添加测试 1 分数和测试 2 分数

我们快到了！我们现在删除**标记**和**测试编号**列，因为它们的数据在测试标记列中。然后，我们将重复的成绩条目放入表中，并按学生 id 对数据进行排序。

最终的性能数据如下所示。

整洁的性能数据

从上表中可以看出，

1.  *id、term、test 1 标记和 test 2 标记都是构成表格列的变量。*
2.  *每个学生在测试中的表现数据是一个观察值，形成一行。*
3.  *该表只有一种观察单位，即学生在考试中的表现。*

**结论**

我们现在已经将学生和成绩的数据分别放在不同的表中。对于学生数据集，最初保存**性别**和**年龄**数据的**性别和年龄**列被拆分为单独的列。对于性能数据集，我们最初用作列标题的**术语 1** 、**术语 2** 和**术语 3** 值被更改为**术语**列的值。最后**测试 1** 和**测试 2** 变量从数值变为列标题。

这标志着第二部分数据整洁博客的结束。

**参考文献**

1.  [我的第一篇关于凌乱数据的帖子](/@kimrodrikwa/untidy-data-a90b6e3ebe4c)。
2.  [整理资料论文](https://vita.had.co.nz/papers/tidy-data.pdf)由哈德利·韦翰撰写
3.  [熊猫图书馆](https://pandas.pydata.org/)
4.  [熊猫数据帧](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
5.  熊猫数据框[应用](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html)
6.  熊猫数据帧[下降](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)
7.  熊猫[融化](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html)。