# numpy.random —用 Python 生成随机数

> 原文：<https://medium.com/analytics-vidhya/numpy-random-generating-random-numbers-in-python-fc5fff3a1594?source=collection_archive---------14----------------------->

![](img/ba58f4b22f0c751e2b75e55ef4495e9f.png)

随机数有很多应用，如密码学、游戏和神经网络。在这里，我们将讨论和识别在 python 的 NumPy 模块中生成随机数的不同方法。

```
 **import numpy as np** 
```

***numpy . random . random(size)***—返回半开区间内的随机浮点数[0.0，1.0]，即 **1.0 > x≥0.0**

```
>>>np.random.random((2,2))array([[ 0.66032591,  0.91397527],
       [ 0.63366556,  0.36594058]])
```

***numpy . random . randint(low，high，size，dtype=int)*** —从“**离散均匀分布**”中返回从*低*到*高*的随机整数。

```
>>> np.random.randint([1, 3, 5, 7], [[10], [20]], dtype=np.uint8)array([[ 8,  6,  9,  7], 
       [ 1, 16,  9, 12]], dtype=uint8)
```

***numpy.random.randn(d1，d2，..dn)*** —从“**标准正态**”分布中返回给定维度的样本。

```
>>> np.random.randn(2,2)array([[ 0.6762164 , -1.37066901],
       [ 0.23856319,  0.61407709]])
```

***numpy . rand . rand(S1，s2，..sn)***——给定维度的随机值，取值范围在(0，1)之间。

```
>>> np.random.rand(3,2)
array([[ 0.14022471,  0.96360618],  
       [ 0.37601032,  0.25528411],  
       [ 0.49313049,  0.94909878]]) 
```

random.rand 和 random.randn 在功能和语法上都是相似的函数，很容易混淆。
***random . rand***—根据标准正态分布返回样本。 ***random . randn***—返回介于(0，1)之间的数字

***【numpy . random . normal(loc = 0.0，scale=1.0，size = None)****——*从正态(高斯)分布中抽取随机样本。

```
loc : Mean (“centre”) of the distribution
scale : spread or “width” of the distribution>>> np.random.normal(3, 2.5, size=(2, 4))
array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   
       [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]]) 
```

这些是产生随机数最常用的方法。更多详情可以在 [*文档*](https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html) 中搜索。要获得更多关于 numpy 的知识，请查看我的 [*文章*](/@omrastogi/master-numpy-in-45-minutes-74b2460ecb00) 。

> 如果这篇文章是有帮助的，检查我的类似主题的其他文章。