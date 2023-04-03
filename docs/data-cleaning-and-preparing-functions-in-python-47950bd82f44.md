# Python 中的数据清理和准备函数

> 原文：<https://medium.com/analytics-vidhya/data-cleaning-and-preparing-functions-in-python-47950bd82f44?source=collection_archive---------33----------------------->

因为每个有抱负的数据科学家都知道数据清理和准备的重要性，所以让我们使用 python 中的 pandas 和 numpy 来深入研究一些可以用于数据清理的方法。

首先你需要导入两个库

```
import pandas as pd
import numpy as np
```

出于演示目的，我们将在本文中创建临时数据帧和序列。

1.  处理丢失的数据

```
string_data = pd.Series(['affd','fafd','dgbddb',np.nan])
string_data.isnull()#output
0    False
1    False
2    False
3     True
dtype: bool
```

正在申请。dataframe 上的 isnull()函数返回一系列 null 值为 true 的 bool 项。

2.过滤掉丢失的数据

```
from numpy import nan as NA
data = pd.Series([1,NA,3.5,NA,7])data_na = data.dropna()
data[data.notnull()]# output
0    1.0
2    3.5
4    7.0
dtype: float64
```

你可以用。dropna()函数或者你可以使用。notull()。结果是一个包含所有非空值的序列。

默认情况下，For dataframe dropna()删除所有包含 na 值的行。

```
data = pd.DataFrame([[1,2,3,4],[4,6,4,NA],
                    [2,6,3,1],[8,9,NA,NA]])
data
data.dropna()#output
**data**
Out[12]: 
   0  1    2    3
0  1  2  3.0  4.0
1  4  6  4.0  NaN
2  2  6  3.0  1.0
3  8  9  NaN  NaN**data.dropna()**
Out[13]: 
   0  1    2    3
0  1  2  3.0  4.0
2  2  6  3.0  1.0
```

传递 how="all "只会删除包含所有 NAs 的行。

```
data = pd.DataFrame([[1,2,3,4],[4,6,4,NA],
                    [2,6,3,1],[8,9,NA,NA],[NA,NA,NA,NA]])**data**
Out[17]: 
     0    1    2    3
0  1.0  2.0  3.0  4.0
1  4.0  6.0  4.0  NaN
2  2.0  6.0  3.0  1.0
3  8.0  9.0  NaN  NaN
4  NaN  NaN  NaN  NaN**data.dropna(how = "all")**
Out[18]: 
     0    1    2    3
0  1.0  2.0  3.0  4.0
1  4.0  6.0  4.0  NaN
2  2.0  6.0  3.0  1.0
3  8.0  9.0  NaN  NaN
```

以同样的方式删除列，传递 axis = 1

```
data = pd.DataFrame([[1,2,3,4],[4,6,4,NA],
                    [2,6,3,1],[8,9,NA,NA]])**data**
Out[22]: 
   0  1    2    3
0  1  2  3.0  4.0
1  4  6  4.0  NaN
2  2  6  3.0  1.0
3  8  9  NaN  NaN**data.dropna(axis = 1)**
Out[23]: 
   0  1
0  1  2
1  4  6
2  2  6
3  8  9**data.dropna(axis = 1, how = "all")**
Out[24]: 
   0  1    2    3
0  1  2  3.0  4.0
1  4  6  4.0  NaN
2  2  6  3.0  1.0
3  8  9  NaN  NaN
```

假设您希望只保留包含一定数量观察值的行。

```
df = pd.DataFrame(np.random.randn(7,3))
df.iloc[:4,1] = NA
df.iloc[:2,2] = NA**df**
Out[28]: 
          0         1         2
0  0.926414       NaN       NaN
1  0.159192       NaN       NaN
2 -0.724809       NaN -1.417761
3  0.282927       NaN -1.463186
4  0.170544 -0.490909 -0.974798
5 -0.770864 -0.087966 -0.125960
6 -1.874506  1.419397 -0.758035**df.dropna()**
Out[29]: 
          0         1         2
4  0.170544 -0.490909 -0.974798
5 -0.770864 -0.087966 -0.125960
6 -1.874506  1.419397 -0.758035

**df.dropna(thresh = 2)**
Out[30]: 
          0         1         2
2 -0.724809       NaN -1.417761
3  0.282927       NaN -1.463186
4  0.170544 -0.490909 -0.974798
5 -0.770864 -0.087966 -0.125960
6 -1.874506  1.419397 -0.758035
```

3.填充缺失值

用常数调用 fillna 会用该值替换丢失的值

```
**df.fillna(0)**
Out[31]: 
          0         1         2
0  0.926414  0.000000  0.000000
1  0.159192  0.000000  0.000000
2 -0.724809  0.000000 -1.417761
3  0.282927  0.000000 -1.463186
4  0.170544 -0.490909 -0.974798
5 -0.770864 -0.087966 -0.125960
6 -1.874506  1.419397 -0.758035
```

用字典调用 fillna()，你可以为每一列使用不同的填充值

```
**df.fillna({1:0.5,2:0})**
Out[32]: 
          0         1         2
0  0.926414  0.500000  0.000000
1  0.159192  0.500000  0.000000
2 -0.724809  0.500000 -1.417761
3  0.282927  0.500000 -1.463186
4  0.170544 -0.490909 -0.974798
5 -0.770864 -0.087966 -0.125960
6 -1.874506  1.419397 -0.758035
```

4.删除重复项

```
data = pd.DataFrame({'k1':['one','two'] *3 + ['two'],
                     'k2' : [1,1,2,3,3,4,4]})**data**
Out[40]: 
    k1  k2
0  one   1
1  two   1
2  one   2
3  two   3
4  one   3
5  two   4
6  two   4**data.duplicated()** ## returns a boolean series
Out[41]: 
0    False
1    False
2    False
3    False
4    False
5    False
6     True
dtype: bool**data.drop_duplicates()** data.duplicated() ## returns a boolean series
Out[41]: 
0    False
1    False
2    False
3    False
4    False
5    False
6     True
dtype: bool**data.drop_duplicates()** Out[42]: 
    k1  k2
0  one   1
1  two   1
2  one   2
3  two   3
4  one   3
5  two   4
```

您还可以指定一个子集来检测重复项

```
**data['v1'] = range(7)
data**
Out[44]: 
    k1  k2  v1
0  one   1   0
1  two   1   1
2  one   2   2
3  two   3   3
4  one   3   4
5  two   4   5
6  two   4   6**data.drop_duplicates(['k1'])**
Out[45]: 
    k1  k2  v1
0  one   1   0
1  two   1   1
```

在我们继续做进一步分析之前，这些是我们可以用来清理和准备数据的一些函数。将在接下来的部分中涉及更多内容，比如使用地图和 lambda 函数转换数据。

任何改进的建议都是受欢迎的，如果这篇文章在某些方面对你有所帮助，我会非常喜欢它。