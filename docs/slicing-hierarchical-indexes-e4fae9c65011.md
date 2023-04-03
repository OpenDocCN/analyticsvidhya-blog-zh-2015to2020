# 分层索引切片

> 原文：<https://medium.com/analytics-vidhya/slicing-hierarchical-indexes-e4fae9c65011?source=collection_archive---------19----------------------->

![](img/dd3c2c36694342c3fa75ba3fa2ee635a.png)

在上一个教程中，我们学习了一些有用的工具，如何对一些数据集进行更细致的数据分析，特别是当它们以更高维度呈现时。这就是我们所说的层次索引或多重索引。当一个数据帧有一个以上的索引列或更高维度时，称之为分层索引。你可以在这里[刷新你的记忆](/analytics-vidhya/hierarchical-indexing-76dcc6a8dd86)，这样你就能感觉到我们在哪里，我将在本教程中谈论什么。

我将进入一些有用的分析，这些分析可以在多索引数据框架上进行。我将使用我在上一篇文章中使用的相同的`Fifa19`数据框架。

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

这个数据框架来自于从`EA Sport`的 FIFA19 数据集中选出的几个球员的球员评级。数据帧目前没有以我们喜欢的格式呈现，所以我将快速跳过几个步骤，对其进行多索引和排序。如果漏了哪一步，可以参考之前的帖子[这里](/analytics-vidhya/hierarchical-indexing-76dcc6a8dd86)。

```
In [5]: fifa19.set_index(['Position','Rank']).sort_index()
Out[5]:              Name     Overall
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

我们的数据现在已经以分层的方式分类，并准备好进行分析。

对分层索引的数据帧进行排序是有用的切片。例如，如果我们想根据 FIFA19 数据集找出这些人中谁是最好的中锋，我们将调用`.loc`访问器并传递一个元组`('CF' ,'1st')`来提取数据帧中排名最高的前锋。

```
In [6]: fifa19.loc[('CF', '1st')]
Out[6]:
      Name    Ronaldo
      Overall 94
      Name: (CF, 1st), dtype: object
```

但是，如果我们在 tuple 后面加上一个特定的列名，即“name”或“total ”,我们将提取表中的一个元素。

```
In [7]: fifa19.loc[('CF', '1st'), 'Name']
Out[7]: 
       'Ronaldo'
```

接下来，如果我们碰巧用一个单独的字符串调用`.loc`访问器，比如只调用‘GK’，Pandas 将对外部索引进行切片，并返回所有与切片字符串对应的行，即‘GK’。

```
In [8]: fifa19.loc['GK']
Out[8]:       Name      Overall
    Rank
     1st     De Gea     91
     2nd     Alisson    89
     3rd     Coutois    88
```

我们可以做的另一件事是使用索引列表提取切片。我们通过在调用`.loc`访问器后传递两个或更多索引字符串的列表来实现这一点。Pandas 然后返回这些索引的所有行。

```
In [9]: fifa19.loc[['GK', 'MF']]
Out[9]:              Name       Overall
 Position    Rank
       GK    1st     De Gea      91
             3rd     Coutois     88
             2nd     Alisson     89
       MF    2nd     Hazard      91
             3rd     Kante       90
             1st     De Bruyne   92
```

有些情况下，我们需要找出，例如，在一些特定的位置，只有排名最高的球员。Pandas 使用由一个列表和内部选择索引组成的元组实现了这一点。我要说的是，如果我们需要切片，比方说，守门员(' GK ')和中场(' MF ')中的第三名球员，我们将在调用`.loc`访问器后传递一个元组，在元组中，传递一个只包含' GK '和' MF '的列表，然后在逗号后传递'第三名'。然后，我们将在`.loc`调用的最后添加一个冒号，表示我们希望返回这些索引的所有行。

让我们看看

```
In [10]: fifa19.loc[(['GK' ,'MF'], '3rd'), :]
Out[10]:          Name     Overall
  Postion  Rank  
      GK    3rd   Coutois  88
      MF    3rd   Kante    90
```

在这种情况下，我们只对特定的列感兴趣，而不是对所选位置的整行感兴趣，我们称该特定列为冒号`:`

```
In [10]: fifa19.loc[(['GK' ,'MF'], '3rd'), 'Name']
Out[10]:  
       Postion  Rank  
       GK       3rd   Coutois  
       MF       3rd   Kante    
       Name: Name, dtype: object
```

这种奇特的索引切片方式也适用于内部级别。基于内部索引查找数据有时会很棘手。如果你注意细节，你可能会遇到问题。语法与外部索引非常相似。也就是说，如果我们想在数据帧中看到排名最好和最差的中锋，我们可以通过调用“CF ”,后跟“1st”和“3rd”列表，然后是冒号`:`来返回这些行的所有列。

```
In [11]: fifa19.loc[('CF', ['1st' , '3rd']),:]
Out[11]:         Name     Overall
  Postion  Rank  
      CF   1st   Ronaldo   94
           3rd   Neymar    92
```

说到分层索引，我觉得我必须向您介绍最后一个技巧。就是利用 python 的内置函数`Slice()`

用于索引的元组本身不识别带冒号的切片。为了做到这一点，我们显式地使用 Python 的`Slice()`。

一个`Slice()`创建一个 slice 类型的实例。我们已经知道，一个切片可以用来索引`Start`、`End`、`Step`中的序列。

假设我们想在每个位置上选择得分最高的球员，我们可以用

```
In [12]: fifa19.loc[(['CF','DF','GK' ,'MF', ], '1st'), :]
out[12]:           Name     Overall
  Postion  Rank  
      CF   1st     Ronaldo     94
      DF   1st     Ramos       91
      GK   1st     De Gea      91
      MF   1st     De Bruyne   92
```

这很完美，但是我们可以有一个“更简单”的选择，那就是使用`Slice()`。

```
In [13]: fifa19.loc[(slice(None), '1st'), :]
out[13]:           Name     Overall
  Postion  Rank  
      CF   1st     Ronaldo     94
      DF   1st     Ramos       91
      GK   1st     De Gea      91
      MF   1st     De Bruyne   92
```

我们使用“None”来获取所有 3 个切片参数的默认值。

`Slice(None)`创建一个 slice 实例作为 Slice(none，none，none ),从而返回所有具有“1st”索引的行。

这似乎是一个难题。此时你应该有足够多的东西可以咀嚼了。我会在这里结束它。你可以抽出时间自己练习这些。这些概念相对容易理解。

在下一个教程中，我将处理旋转，数据透视表，堆栈和融化。

在那之前，继续编码吧！