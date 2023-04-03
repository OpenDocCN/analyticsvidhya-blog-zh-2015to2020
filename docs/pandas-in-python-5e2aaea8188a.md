# 蟒蛇皮熊猫

> 原文：<https://medium.com/analytics-vidhya/pandas-in-python-5e2aaea8188a?source=collection_archive---------20----------------------->

Pandas 用于数据操作、分析和清理。

**什么是数据帧和序列？**

**Dataframe** 是一个二维的、大小可变的、潜在异构的表格数据。

它包含行和列，算术运算可以应用于行和列。

**系列**是一个一维标签数组，能够保存任何类型的数据。它可以是整数、浮点、字符串、python 对象等。熊猫系列不过是 excel 表格中的一列。

## 如何创建 dataframe 和 series？

s = pd。系列([1，2，3，4，56，np.nan，7，8，90])
打印

![](img/af1f15ae837e3955aded9b5ca1f8e0fa.png)

**如何通过传递 numpy 数组来创建 dataframe？**

1.  d= pd.date_range('20200809 '，periods=15)
    print(d)
2.  df = pd。DataFrame(np.random.randn(15，4)，index= d，columns = ['A '，' B '，' C '，' D'])
    print(df)

![](img/7545f2791d5b6b35c073b9dd89a2780f.png)

**如何通过传递对象字典创建数据框？**

df1 = pd。DataFrame({'A':[1，2，3，4]，
'B': pd。时间戳(' 20200809 ')，
'C': pd。Series(1，index= list(range(4))，dtype='float32 ')，
'D':np.array([5]*4，dtype= 'int32 ')，
' E ':" Lolitha " })
print(df)

![](img/1c8b40eb8feb9aa39d074621c79dbece.png)

**如何找到数据帧的数据类型？**

![](img/2a8d4cead12bb0ea7d995a8c652509e9.png)

**如何找到数据帧中的前五个和后五个值？**

df = pd。DataFrame(np.random.randn(15，4)，index= d，columns = ['A '，' B '，' C '，' D'])
print(df)

使用 df.head()和 df.tail()

![](img/9d4385f07d89118dcf172b7e8bec1f46.png)

**查找索引和列**

![](img/dc06f8ac8301a0d814d60b3b0700adb3.png)

**数据帧通过排序索引**

![](img/253733a3647d0bf9e4541e4bfdf5f036.png)

**按数值排序数据。**

df.sort_values(by='D ')

![](img/307fafabac82e991dbdb045c6abd96a0.png)

**如何在数据框中选择单列？**

![](img/307fafabac82e991dbdb045c6abd96a0.png)

**如何选择数据框中的单列？**

![](img/797d76346036e91595c11ee23b0df02f.png)

**如何使用标签选择数据？**

![](img/e64d93bf5f3bf8f2878dc24b0595aa3c.png)

**如何使用标签选择多路访问？**

![](img/6045b150521708635eb8918bc74f04be.png)

**如何对行进行切片？**

![](img/15225d04f2d893fde182808d5d57068f.png)

```
How to get particular values in a data frame?
df.loc[‘20200821’,[‘D’,’C’]]
D   -0.008524
C    0.479541
Name: 2020-08-21 00:00:00, dtype: float64How to get scalar Value?
df.loc[d[0],['D','A']]D    0.861121
A   -0.063109
Name: 2020-08-09 00:00:00, dtype: float64
```

Github 知识库链接-python 中的熊猫. ipynb-[https://github.com/lolithasherley7/lolitha.git](https://github.com/lolithasherley7/lolitha.git)

希望这给处理熊猫的基本想法。一定要试一试。