# Pandas 中用于数据科学的基本命令

> 原文：<https://medium.com/analytics-vidhya/basic-commands-in-pandas-for-data-science-192c42595b99?source=collection_archive---------21----------------------->

`import pandas as pd
import numpy as np`

为了练习，我们将使用巴西出租房屋的数据集
，其链接如下。它对公众开放。

[https://www.kaggle.com/rubenssjr/brasilian-houses-to-rent](https://www.kaggle.com/rubenssjr/brasilian-houses-to-rent)

#将文件保存到变量 df
df = PD . read _ CSV(' C:\ \ Users \ JAY \ Desktop \ python \ datasets . CSV ')

`print(df)`

```
Unnamed: 0  city  area  rooms  bathroom  parking spaces floor  \
0              0     1   240      3         3               4     -   
1              1     0    64      2         1               1    10   
2              2     1   443      5         5               4     3   
3              3     1    73      2         2               1    12   
4              4     1    19      1         1               0     -   
...          ...   ...   ...    ...       ...             ...   ...   
6075        6075     1    50      2         1               1     2   
6076        6076     1    84      2         2               1    16   
6077        6077     0    48      1         1               0    13   
6078        6078     1   160      3         2               2     -   
6079        6079     1    60      2         1               1     4   

         animal      furniture      hoa rent amount property tax  \
0         acept      furnished      R$0     R$8,000      R$1,000   
1         acept  not furnished    R$540       R$820        R$122   
2         acept      furnished  R$4,172     R$7,000      R$1,417   
3         acept  not furnished    R$700     R$1,250        R$150   
4     not acept  not furnished      R$0     R$1,200         R$41   
...         ...            ...      ...         ...          ...   
6075      acept  not furnished    R$420     R$1,150          R$0   
6076  not acept      furnished    R$768     R$2,900         R$63   
6077      acept  not furnished    R$250       R$950         R$42   
6078  not acept  not furnished      R$0     R$3,500        R$250   
6079      acept      furnished    R$489     R$1,900          R$0   

     fire insurance     total  
0             R$121   R$9,121  
1              R$11   R$1,493  
2              R$89  R$12,680  
3              R$16   R$2,116  
4              R$16   R$1,257  
...             ...       ...  
6075           R$15   R$1,585  
6076           R$37   R$3,768  
6077           R$13   R$1,255  
6078           R$53   R$3,803  
6079           R$25   R$2,414  

[6080 rows x 14 columns]
```

#获取数据中列的标题
`print(df.columns)`

```
Index(['Unnamed: 0', 'city', 'area', 'rooms', 'bathroom', 'parking spaces',
       'floor', 'animal', 'furniture', 'hoa', 'rent amount', 'property tax',
       'fire insurance', 'total'],
      dtype='object')
```

#读取特定列
和`print(df[‘area’])`

```
0       240
1        64
2       443
3        73
4        19
       ... 
6075     50
6076     84
6077     48
6078    160
6079     60
Name: area, Length: 6080, dtype: int64
```

#你可以指定你想要多少行(切片)
`print(df[‘area’][0:5])`

```
0    240
1     64
2    443
3     73
4     19
Name: area, dtype: int64
```

#获取多列(标签列表)
`print(df[[‘area’,’animal’]])`

```
area     animal
0      240      acept
1       64      acept
2      443      acept
3       73      acept
4       19  not acept
...    ...        ...
6075    50      acept
6076    84  not acept
6077    48      acept
6078   160  not acept
6079    60      acept
```

#打印行数
#打印(data.head)
#打印第 1 4 行
`print(df.head(4))`

```
Unnamed: 0  city  area  rooms  bathroom  parking spaces floor animal  \
0           0     1   240      3         3               4     -  acept   
1           1     0    64      2         1               1    10  acept   
2           2     1   443      5         5               4     3  acept   
3           3     1    73      2         2               1    12  acept   

       furniture      hoa rent amount property tax fire insurance     total  
0      furnished      R$0     R$8,000      R$1,000          R$121   R$9,121  
1  not furnished    R$540       R$820        R$122           R$11   R$1,493  
2      furnished  R$4,172     R$7,000      R$1,417           R$89  R$12,680  
3  not furnished    R$700     R$1,250        R$150           R$16   R$2,116
```

#访问每个元素
#假设第 4 行
的房屋税`print(df.iloc[4,11])`

```
R$41
```

#打印 df.iterrows():
`print(index , row)`中索引的每一行
的数据

#获取数学统计数据

**以下命令的输出不能写入介质。你可以用笔记本查看输出**

`df.describe()`

#按值排序

`df.sort_values(‘city’)`

#您可以添加新列作为其他列的总和
`df[‘Total’] = df[‘hoa’] + df[‘rent amount’] +[‘property tax’] + [“fire insurance”]`

#另一种做法相同
`df[‘Total’] = df.iloc[:, 9:12].sum(axis=1)`

#删除列
`df = df.drop(columns=[‘Total’])`

#移动一列
#首先获取列为列表
`cols = list(df.columns.values)`

#然后根据需要重置 df(我们将交换浴室和房间列)
`df = df[cols[:3]+[cols[4]]+[cols[3]]+cols[5:]]`

# **保存文件**

成 CSV`df.to_csv(‘modified_data_version_1.csv’)`

导入 EXCEL

`df.to_excel(‘modified_data_version_1.xlsx’)`

转换成 TXT

`df.to_csv(‘modified_data_version_1.txt’, sep='\t')`

你必须通过分隔符以 txt 格式正确保存和可视化。