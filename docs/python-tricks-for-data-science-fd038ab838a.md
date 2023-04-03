# 数据科学的 Python 技巧

> 原文：<https://medium.com/analytics-vidhya/python-tricks-for-data-science-fd038ab838a?source=collection_archive---------8----------------------->

![](img/72460d73c9c04ae23992f1f4c7de2d20.png)

这张图片让我想起了 df.explode()，这可能是我最喜欢的熊猫方法。致谢:[罗伯特·祖尼科夫](https://unsplash.com/@rzunikoff)

你是否经常发现自己抓耳挠腮，试图记起两天前在堆栈溢出帖子上发现的那段非常整洁的熊猫代码？这就是本文的目的——将所有的语法放在一个地方。

这篇文章充满了我经常使用的代码；这是一种将我最喜欢的片段保存在一个地方并与他人分享的方式。我会回来更新这篇文章，如果你觉得它有用，请随时发送其他有用的建议。

## 如何在 pandas 数据帧中找到空值的确切位置并看到整行？

```
# Syntaxdf[df.isna().any(axis=1)]# Exampleimport pandas as pd
titanic = pd.read_csv(“train.csv”)
titanic[titanic.isna().any(axis=1)]
```

如果您试图在上下文中查看同一行中的 nan 值和其他值，这很方便。

## 如何检查每列中空值的百分比？

这是一种非常快速的方法，可以看出整个数据帧中每一列的损坏程度。

## 如何检查一列字符串中是否包含任何已编码为字符串的数字？

```
# Syntaxdf[df.column.str.contains("\d+", na=False)]# Exampletitanic[titanic.Ticket.str.contains("\d+", na=False)]
```

这对于数据清理或特征工程来说是一个很好的方法——假设您试图从一列中取出数字或替换它们。

## 如何可靠地将 python 包安装到 Jupyter 笔记本中？

这是一个非常基本的，但我觉得值得记住！启动您的笔记本，然后在一个地方安装您需要的任何新的依赖项，这要方便得多。

```
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install your-desired-python-package
```

这将把您想要的包安装到您在笔记本中使用的同一个 python 环境中。

## 如何让一个相关矩阵可读性强很多？

我发现 seaborn 的相关矩阵热图表示的默认输出读起来非常混乱。

一半的输出是不必要的，配色方案第一眼看上去不容易读懂；这是我的首选解决方案。

舍入值和指定字体也很好！

## 如何获得一个 dataframe 的列的所有可能的组合？

如果您正在测试模型的准确性，并试图找出不同的特征如何影响模型，那么这个工具就很方便了。

这样做相当耗费处理器资源，而且通常过于自动化，但是如果您对一些特定的功能组合感兴趣，这可能会很有用。

## 如何读入大量 CSV 并将它们连接在一起以创建一个数据帧？

```
import os
import glob# Creates a list containing absolute file paths for all csvs in      # current working directorysheets = glob.glob(os.path.join(os.getcwd(), "*.csv"))# Use a generator to read in all the csvs as dataframes
# Then join them all together
# This only works if the columns are the same in each csvbig_dataframe = pd.concat(pd.read_csv((s)for s in sheets))
```

这是我在抓取网站或进行速率受限的 API 调用后经常使用的一种方法，这意味着我会在每次 API 调用时将数据写入文件，并在最后将所有数据连接到一个数据帧中。

## 如何处理一列列表？

explode 方法为列表中的每一项创建一个包含重复信息的新行，这非常漂亮。

虽然这可能会创建一些您不想要的额外的新数据，但它允许您找到一些新的数据点。在上面的例子中，您现在可以找出哪些演员通常获得最高的 star_rating。