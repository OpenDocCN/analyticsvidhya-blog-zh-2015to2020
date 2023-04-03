# 熊猫教程-高级索引

> 原文：<https://medium.com/analytics-vidhya/pandas-tutorials-advanced-indexing-b35909b26bcd?source=collection_archive---------10----------------------->

![](img/3a45906c3d010e9e263c94503a316a04.png)

要了解更多关于高级索引的知识，我们需要记住什么是索引，什么不是索引。

索引是一系列标签。Pandas 系列是一维标记的 NumPy 数组，DataFrame 是列是系列的二维数据结构。

在 Pandas 中，索引像字典键一样是不可变的。它们还假设数据类型是同质的，如 NumPy 数组。

在我们深入了解所有这些之前，让我们快速提醒自己如何创建系列、索引，并尝试修改索引名称，然后进入我们今天的课程。

## 创建系列

让我们创建一个股票和价格序列

```
In [1]: import pandas as pd
In [2]: prices = [10.70, 10.80, 10.50, 10.90]
In [3]: shares = pd.Series(prices)
In [4]: print(shares)
Out[4]:
       0       10.70
       1       10.80
       2       10.50
       3       10.90
dtype: float64
```

## 创建索引

```
In [5]: days = ['Mon','Tue','Wed','Thu']
In [6]: shares = pd.Series(prices, Index=days)
Out[6]
     Mon      10.70
     Tue      10.80
     Wed      10.50
     Thu      10.90
```

在上面代码的这一步中，我们将先前的默认指数更改为每日股价。我们通过在`pd.Series()`方法中指示 Index = days 来做到这一点。

我们现在可以通过调用 index 属性检查`Shares`来验证索引不再是默认的，而是字符串对象

```
In [7]: print(shares.index)
Out[7]: 
     Index(['Mon','Tue','Wed','Thu']),dtype='object')In[8]: print(shares.index[1])
Out[8]: 
     TueIn [9]: print(shares.index[ :2]
Out[9]:
    Index(['Mon','Tue'], dtype='object'In[10]: print(shares.index[ -2:]
Out[10]:
    Index(['Wed','Thu'], dytpe='object')In [11]: print(shares.index.name)
Out[11]:
     None
```

从第 11 行代码可以看出，索引“列”没有名称。我们通过调用`.name`属性(`shares.index.name`)发现了这一点。我们可以通过给`shares.index.name`起一个名字来纠正或修改这一点。这使得输出更清晰，更容易被第三方用户阅读和理解。

```
In [12]: shares.index.name = 'Weekdays'
In [13]: print(shares)
Out[13]:
       Weekdays
        Mon      10.70
        Tue      10.80
        Wed      10.50
        Thu      10.90
        dtype:float64
```

但是，请注意，不能在数据帧中修改索引。你不能简单地通过给一个特定的条目指定一个新的名字来改变一个索引。这将引发一个`TypeError`

```
In[14]: shares.index[2] = 'Wednesday'
TypeError*: Index does not support mutable operations*In[15]: shares.index[ :4] = ['Monday', 'Tuesday']
TypeError: *Index does not support mutable operations*
```

TypeError : Index 不支持可变操作，这意味着我们不能改变任何索引的形状或格式。熊猫不支持任何此类操作

让我们用另一个例子来更全面地了解我想表达的观点。我将“导入”存储在本地驱动器上的失业数据集。

```
In [1]: import pandas as pd
In [2]: Unemployment = pd.read_csv('unemployment.csv')
In [3]: Unemployment.head()
Out[3]:
      Zip     Unemployment   Participants
0    1001       0.06           13801
1    1002       0.09           24551
2    1003       0.17           11477
3    1004       0.10            4086
4    1005       0.05           11362
```

索引列没有标签。我们不希望出现这种情况。相反，我们希望分析这些失业数据，确切了解失业率是从哪里记录的。如果是这种情况，我们需要将 Zip 列作为新的参考点。我们将通过将 Zip 列指定为新索引来实现这一点

```
 In [2]: Unemployment.index = unemployment['Zip']
In [3]: unemployment.head()
Out[3]:
         Zip     Unemployment   Participants
Zip
1001     1001       0.06           13801
1002     1002       0.09           24551
1003     1003       0.17           11477
1004     1004       0.10            4086
1005     1005       0.05           11362
```

现在我们已经成功地使 Zip 列成为新的索引，但是还有一个小问题。我们似乎有一个 Zip 列的副本。我们可能不需要 Zip 列在我们的数据框架中占据空间，因为它的主要目的是作为我们 ie 索引的参考点。我们可以使用`del`函数删除该列。

```
In [4]: del Unemployment ['Zip']
```

此外，值得注意的是，我们可以在导入数据集时，从一开始就将 Zip 列指定为索引列，从而避免这种麻烦。通过在`'pd.read_csv()'`方法中指出`index_col='Zip'`,我们将节省一些编码时间并完成工作。

```
In [1]: Unemplyment=pd.Read_csv('unmeployment.csv',index_col='Zip')
In [2]: Unemployment.head()
Out[2]:
        Unemployment   Participants
Zip
1001      0.06           13801
1002      0.09           24551
1003      0.17           11477
1004      0.10            4086
1005      0.05           11362
```

## 更改数据帧的索引

正如我前面指出的，索引是不可变的。这意味着如果我们想要改变或修改数据帧中的索引，那么我们需要改变整个序列。我们可以通过使用`list comprehensions`来做到这一点

列表理解是在一行中生成列表的一种简单易读的方式。列表理解将几行代码压缩成一行可爱的代码，足够简短以保持读者的兴趣，也足够详细以完成预期的工作。

例如，代码

```
In[]:squares = [x**2 for x in range(20)]
```

生成包含从 0 到 19 的所有数字的平方的列表。
替代代码应该是

```
In[]:Squares = [ ]
     for x in range(20):
      Squares.append(x**2)
```

在修改数据帧的索引时，我们可以使用这种相当简洁的方式来处理大量的代码行。例如，如果我们想给数据帧中的所有邮政编码加 2 来帮助我们进行分析，我们可以使用列表理解

```
In []: new_zip = [i+2 for i in Unemployment.Zip]
In []: Unemployment.Zip = new_zip
In []: unemployment.head()
Out[]:
         Unemployment   Participants
new_zip
1003      0.06           13801
1004      0.09           24551
1005      0.17           11477
1006      0.10            4086
1007      0.05           11362
```

现在大概就是这样了。我希望你学到了新东西。在下一个教程中，我将进入层次索引。

我希望你在阅读和阅读这篇文章的时候和我写这篇文章的时候一样开心。

继续编码，直到我再次来到你身边！