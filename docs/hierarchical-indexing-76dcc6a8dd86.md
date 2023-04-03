# 分级索引

> 原文：<https://medium.com/analytics-vidhya/hierarchical-indexing-76dcc6a8dd86?source=collection_archive---------15----------------------->

![](img/88eca14a069f8bf678f937f862b651ef.png)

在上一个教程中，我们了解了高级索引。我们学习了如何使用默认索引将索引列重新分配给数据帧。我们还看到了如何通过用`List Comprehensions`改变整个索引列来改变或修改索引。你可以在这里刷新你的记忆，或者如果你还没有这样做的话，你可以阅读全部内容。

说完这些，我们将直接进入今天的主题:层次索引。

当我们的数据帧有两个以上的维度时，就出现了多索引或层次索引。正如我们已经知道的，Series 是一个一维的带标签的 NumPy 数组，DataFrame 通常是一个列是 Series 的二维表。在某些情况下，为了执行一些复杂的数据分析和操作，我们的数据以更高维度呈现。

多索引至少为数据增加了一个维度。顾名思义，分级索引是根据项目的排名对多个项目进行排序。

让我们用一些数据来看看它是如何工作的。

我将创建一个数据框架，从 Fifa19 数据集的一些球员的球员评级。我们将通过比较这些参与者以及分级指数如何帮助我们的事业来阐述我们的观点。

```
In [1]: import pandas as pd
In [2]: data = {'Position': ['GK','GK','GK', 'DF','DF','DF',
            'MF','MF','MF','CF','CF','CF'],   
            'Name': ['De Gea', 'Coutois','Allison','Van Dijk',
            'Ramos','Godin','Hazard','Kante','De Bruyne', 'Ronaldo'
            'Messi','Neymar'],
            'Overall': ['91','88','89','89','91','90','91',
            '90','92','94','93','92'],
            'Rank': ['1st','3rd','2nd','3rd','1st','2nd',
            '2nd','3rd','1st','1st','2nd','3rd']}
In [3]: fifa19 = pd.DataFrame(data, columns=['Position','Name',
                'Overall','Rank'])
In [4]: fifa19
Out[4]: 
             Position     Name        Overall     Rank
     0        GK         De Gea          91        1st
     1        GK         Coutios         88        3rd
     2        GK         Allison         89        2nd
     3        DF         Van Dijk        89        3rd
     4        DF         Ramos           91        1st
     5        DF         Godin           90        2nd
     6        MF         Hazard          91        2nd
     7        MF         Kante           90        3rd
     8        MF         De Bruyne       92        1st
     9        CF         Ronaldo         94        1st
     10       CF         Messi           93        2nd
     11       CF         Neymar          92        3rd
```

从我们的数据框架中，我们注意到该索引是默认的 Pandas 索引；列“位置”和“等级”都有重复的值或对象。当我们想要分析数据时，这有时会给我们带来问题。我们想要做的是使用有意义的索引来惟一地标识每一行，并使我们更容易理解我们正在处理的数据。

这就是多索引或层次索引的用武之地。我们将把我们的索引从默认索引重新分配到一个更细微的索引，以捕捉我们分析的本质。

我们通过使用`set_index()`方法来做到这一点。对于层次索引，我们传递一个列表来表示我们希望如何惟一地标识行。

```
In [5]: fif19.set_index(['Position','Rank'], drop=False)
In [6]: fifa19
Out[6]:            Position     Name        Overall     Rank
 Position  Rank
    GK     1st        GK         De Gea          91        1st
    GK     3rd        GK         Coutios         88        3rd
    GK     2nd        GK         Allison         89        2nd
    DF     3rd        DF         Van Dijk        89        3rd
    DF     1st        DF         Ramos           91        1st
    DF     2nd        DF         Godin           90        2nd
    MF     2nd        MF         Hazard          91        2nd
    MF     3rd        MF         Kante           90        3rd
    MF     1st        MF         De Bruyne       92        1st
    CF     1st        CF         Ronaldo         94        1st
    CF     2nd        CF         Messi           93        2nd
    CF     3rd        CF         Neymar          92        3rd
```

从上面的代码中我们可以看到，我们已经将新的索引设置为“Position”和“Rank ”,但是这些列有一个副本。这是因为我们通过了`drop=False`，它将列保持在原来的位置。然而，默认的方法是`drop=True`，所以如果没有指定`drop=False`，这两列将被设置为索引，并且自动删除这两列。

让我们看看

```
In [7]: fifa19.set_index(['Position','Rank'])
Out[7]:              Name        Overall    
 Position  Rank
    GK     1st       De Gea          91       
    GK     3rd       Coutios         88        
    GK     2nd       Allison         89       
    DF     3rd       Van Dijk        89        
    DF     1st       Ramos           91      
    DF     2nd       Godin           90     
    MF     2nd       Hazard          91     
    MF     3rd       Kante           90      
    MF     1st       De Bruyne       92       
    CF     1st       Ronaldo         94      
    CF     2nd       Messi           93    
    CF     3rd       Neymar          92 
```

我们使用带有列标签有序列表的`set_index()`来创建新的索引。

为了验证我们确实已经将数据帧设置为分层索引，我们调用了`.index`属性

```
In [8]: fifa19=fifa19.set_index(['Position','Rank'])
In [9]: fifa19.index
Out[9]: MultiIndex(levels=[['CF', 'DF', 'GK', 'MF'],
                          ['1st', '2nd',   '3rd']],
                   codes= [[2, 2, 2, 1, 1, 1, 3, 3, 3, 0, 0, 0],
                          [0, 2, 1, 2,0,1, 1, 2, 0, 0, 1, 2]],
                   names= ['Position', 'Rank'])
```

多指数将位置作为第一级，这意味着我们的数据被分为中锋、后卫、守门员和中场。然后是排名。这意味着，对于每一个指定的位置，球员被排在第一，第二或第三。

这看起来很容易，并且稍微减轻了分析的任务。以这种方式保存索引可以减少大量的工作，但是有一个小问题。排名不是以有序的方式进行的。更有价值的是，将球员从最好到最差排列在他们的位置上。

我们可以用`sort_index()`方法来解决这个问题。

```
In [10]: fifa19.sort_index()
Out[10]:              Name     Overall
    Position  Rank  
     CF       1st    Ronaldo     94
              2nd    Messi       93
              3rd    Neymar      92
     DF       1st    Ramos       91
              2nd    Godin       90
              3rd    Van Dijk    89
     GK       1st    De Gea      91
              2nd    Coutois     89
              3rd    Allison     88
     MF       1st    De Bruyne   92
              2nd    Hazard      91
              3rd    Kante       90
```

这看起来很棒。我们以分层的方式对数据进行了分类。根据 FIFA19 的数据，我们可以知道 c 罗是最佳前锋，而拉莫斯、德基和德布鲁因等人占据了其他位置。当真正的分析开始时，让我们的数据框架保持这种格式可以让我们更容易着陆。

我希望你发现这个教程信息丰富。接下来，我们将看看如何从多索引数据帧中切片和提取数据。

在那之前，编码快乐！