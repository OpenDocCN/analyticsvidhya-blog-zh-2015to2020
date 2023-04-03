# 数据科学中的 Python 正态性检验

> 原文：<https://medium.com/analytics-vidhya/normality-test-with-python-in-data-science-5abbefc81fd0?source=collection_archive---------3----------------------->

夏皮罗-维尔克检验、安德森-达令检验、达戈斯蒂诺的 K 平方检验

![](img/2d14b4edc7ce52b4ad54ef66ccdfc49c.png)

数据科学

# 目录:-

1.  夏皮罗-威尔克试验
2.  安德森-达林试验
3.  达戈斯蒂诺的 K 平方检验

# 1.夏皮罗-威尔克试验

夏皮罗-维尔克测试(T1)是频率主义者(T4)统计(T5)中的常态(T3)测试(T2)。该书于 1965 年由塞缪尔·桑福德·夏皮罗和马丁·维尔克出版。

夏皮罗-维尔克检验用于计算 W 统计量，检验随机样本 x1，x2，…，xn 是否(具体地)来自正态分布。W 的小值是 W 统计量偏离正态性和百分点的证据，通过蒙特卡罗模拟获得。

在与其他拟合优度测试的比较研究中，该测试表现非常出色。

**假设:**

*   每个样本中的观察值都是独立同分布的(iid)。

**假设:**

H0:数据服从正态分布。

H1:数据不符合正态分布。

```
*#Python code
#Example of Shapiro Wilk Test*
from scipy.stats import shapiro
data = [1,1.2,0.2,0.3,-1,-0.2,-0.6,-0.8,0.8,0.1]
stat, p = shapiro(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print("Data follows Normal Distribution")
else:
    print("Data does not follow Normal Distribution")***OUTPUT:*** *Data follows Normal Distribution*
```

# 2.安德森-达林试验

**安德森-达林检验**是一种[统计检验](https://en.wikipedia.org/wiki/Hypothesis_testing)，检验给定的数据样本是否来自给定的[概率分布](https://en.wikipedia.org/wiki/Probability_distribution)。在其基本形式中，测试假设在被测试的分布中没有要被估计的参数，在这种情况下，测试和它的一组[临界值](https://en.wikipedia.org/wiki/Critical_values)是无分布的。

它可以用来检查一个数据样本是否正常。该测试是一种更复杂的非参数拟合优度统计测试的修改版本，称为 [Kolmogorov-Smirnov 测试](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)。

**假设:**

*   每个样本中的观察值都是独立同分布的(iid)。

**假设:**

H0:数据服从正态分布。

H1:数据不符合正态分布。

```
*#Python code
#Example of Anderson-Darling Test*
from scipy.stats import anderson
data = [1,1.2,0.2,0.3,-1,-0.2,-0.6,-0.8,0.8,0.1]
result = anderson(data)
***OUTPUT***:
*AndersonResult(statistic=0.19788206806788722, critical_values=array([0.501, 0.57 , 0.684, 0.798, 0.95 ]), significance_level=array([15\. , 10\. ,  5\. ,  2.5,  1\. ]))*
```

检验统计量为 **0.1979** 。我们可以将该值与对应于每个显著性水平的每个临界值进行比较，以查看测试结果是否显著。

```
*#Python code* print('stat=%.3f' % (result.statistic))
for i in range(len(result.critical_values)):
 sl, cv = result.significance_level[i], result.critical_values[i]
 if result.statistic < cv:
  print('Data follows Normal at the %.1f%% level' % (sl))
 else:
  print('Data does not follows Normal at the %.1f%% level' % (sl))
***OUTPUT:*** *Data follows Normal at the 15.0% level
Data follows Normal at the 10.0% level
Data follows Normal at the 5.0% level
Data follows Normal at the 2.5% level
Data follows Normal at the 1.0% level*
```

# 3.达戈斯蒂诺的 K 平方检验

[达戈斯蒂诺的 K 检验](https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test)从数据中计算汇总统计数据，即峰度和偏度，以确定数据分布是否偏离了以拉尔夫·达戈斯蒂诺命名的正态分布。

*   **偏斜**是分布被向左或向右推的程度的量化，是分布不对称性的度量。
*   **峰度**量化了尾部分布的多少。这是一个简单且常用的正态性统计检验。

**假设:**

*   每个样本中的观察值都是独立同分布的(iid)。

**假设:**

H0:数据服从正态分布。

H1:数据不符合正态分布。

```
*#Python code
#Example ofD’Agostino’s K-squared Test* from scipy.stats import normaltest
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
stat, p = normaltest(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
 print('Data follows normal')
else:
 print('Data does not follow normal')
***OUTPUT:*** *stat=3.392, p=0.183
Data follows normal*
```

# 通过以下方式联系我:-

*LinkedIn:-*[https://www.linkedin.com/in/shivam-mishra17/](https://www.linkedin.com/in/shivam-mishra17/)

*电子邮件:——shivammishra2186@yahoo.com*

*推特:-*[【https://twitter.com/ishivammishra17】T21](https://twitter.com/ishivammishra17)