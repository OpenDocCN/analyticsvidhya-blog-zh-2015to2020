# 熊猫教程:融化

> 原文：<https://medium.com/analytics-vidhya/pandas-tutorial-melting-39a924a2a3ba?source=collection_archive---------23----------------------->

![](img/4507e48a726578e3d91d1d8585312135.png)

统计学家 Hadley Wickham 在他 2014 年的论文“整洁论文”中解释说，尽管数据有各种形状和格式，但总是需要有一种正式、明确的方式来描述数据的形状。

他认为，除了其他事情之外，以这种格式保存数据可以让数据科学家更容易地清理和分析他/她的数据。保持数据“整洁”可以剥离几层分析，帮助我们更好地理解数据的形状如何适应数据分析的各个组成部分。

为了更好地理解他的意思，我将向您介绍一个我使用`English Premiership Golden Boot Race 2018/2019`创建的简单数据集。这些值可能不完全准确，但请遵循我正在开发的论点和我试图提出的观点。

数据集 I

```
 Name      Goals     Assists
0   Mo Salah     25       8
1   Aguero       22       12
2   Aubameyang   22       11
```

数据集 II

```
 0          1          2
Name     Mo Salah   Aguero   Aubameyang
Goals     25         22         22 
Assists   8          12         11
```

这两个数据集类似于 Hadley Wickham 论文中给出的例子。这两个数据集代表相同的东西，但它们以不同的方式呈现。这两种风格都给负责从数据中获取洞察力的数据分析师或科学家带来了独特的困难。一些数据格式更适合于报告，而另一些则更便于分析。因此，为了我们的目的，我们必须重新组织这些数据集，使它们看起来整洁，有助于我们的分析。

**整理数据**有三个原则。这些是经验法则，通常被视为行业最佳实践。

*   列代表单独的变量
*   行代表单独的观察
*   观察单位形成表格

也就是说，在整齐的格式中，每个观察现在将包含球员的名字、进球和助攻。通过这种方式使数据变得整洁，可以更容易地解决常见问题。

我们希望用上面提供的数据集解决的问题是，将所有列转换为代表单独的变量，并使行代表单独的观察值。

我们将使用一种叫做**融化的熊猫方法来做这件事。**

融合数据是将数据列转换为数据行的过程。然而，如果列已经在正确的位置，熔化它将使您的数据混乱，并且您将在分析中遇到新的问题。

这里，顾名思义，Melting 方法将“进球”和“助攻”列“熔”成一列，我们方便地称之为“统计”，然后创建一个新列“数字”，其中将存储每个足球统计数据的数字。

假设`Golden Boot race`数据集存储为‘df ’,我们将首先指定我们想要使用的数据帧以及我们想要保持不变的列。

对于我们的数据集，我们希望保持“名称”不变，并将目标和辅助“融合”到一个单独的列中。我们通过在`pd.melt()`方法中指定`id_vars = 'Name'`参数来做到这一点。`pd.melt()`的另一个默认参数是`value_vars`。我们将想要融合的列的列表传递给该参数。`value_vars = ['Goals' , 'Assists']`

```
In [1]: pd.melt(frame=df, id_vars ='Name', 
                value_vars=['Goals,    'Assists'])
Out[1]:     Name         variable        Value
      0    Mo Salah       Goals          25
      1    Aguero         Goals          22 
      2    Aubameyang     Goals          22
      3    Mo Salah       Assists         8
      4    Aguero         Assists        12
      5    Aubameyang     Assists        11
```

请注意，如果我们没有像以前那样指定列，`pd.melt`将使用所有没有在`id_vars`参数中指定的列。

还要注意，默认的**变量**列名并不总是我们想要的。当融合数据帧时，让列名比**变量**和**值更有意义。**默认名称在某些情况下可能有效，但最好总是有不言自明的数据。我们可以用 **var_name** 和 **value_name** 来重命名新融合的列

```
In [2]: pd.melt(frame=df, id_vars='Name',
                value_vars=['Goals', 'Assists']
                var_name ='Stats', Value_name ='Numbers'
Out[2]:     Name          Stats        Numbers
      0    Mo Salah       Goals          25
      1    Aguero         Goals          22 
      2    Aubameyang     Goals          22
      3    Mo Salah       Assists         8
      4    Aguero         Assists        12
      5    Aubameyang     Assists        11
```

这种方法似乎非常方便，但情况并非总是如此。正如您所想象的，未来您将使用的真实世界的数据通常非常混乱，一点也不像本例中使用的经过修整的数据集。真实世界的数据可能会非常麻烦，所以掌握这种简单的技术将对你有很大的好处，当你着手处理个人项目的时候。

我强烈建议您获取一些原始数据，并使用它们。下一个教程将是这个方法的逆过程:旋转。

我希望你在阅读和阅读本教程时感到愉快。下次见，编码快乐！