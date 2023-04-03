# Numpy 迭代数组

> 原文：<https://medium.com/analytics-vidhya/numpy-iterating-array-2ed382bb4a03?source=collection_archive---------20----------------------->

![](img/214d09c96b6c3be5d0aeb07c25f2b483.png)

迭代意味着逐个遍历元素。

由于我们在 python (numpy)中使用多维数组，所以我们可以使用*进行*循环。

## 迭代一维:

对于一维数组，迭代一个接一个地进行。

```
import numpy as np
arr=np.array([1,2,3])
print("Original array:",arr)print("After iteration:")
for x in arr:
    print(x)
```