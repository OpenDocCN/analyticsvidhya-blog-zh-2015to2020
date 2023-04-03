# 通过游戏学习数字——概念、索引、矢量化等等

> 原文：<https://medium.com/analytics-vidhya/learn-numpy-by-playing-concepts-indexing-vectorization-and-more-efea45485f19?source=collection_archive---------8----------------------->

![](img/bffb5743db810acbdd5c33aebe66ca50.png)

# 介绍

这个博客旨在成为初学者通过尝试真实代码来学习 numpy 的游乐场。我尽可能少地使用文本内容，尽可能多地使用代码示例。

这也是你已经了解的 numpy 特性的**快速未来参考指南**。每个单元的输出都有描述结果的细节。

## 先决条件

*   基本编程知识
*   熟悉 python(循环、数组等)。)

## 我们将涵盖哪些内容

基础

*   创建数组
*   了解 Numpy 数组的结构(维度、形状和步距)
*   数据类型和转换
*   索引方法
*   数组运算

预先的

*   广播
*   …向量化…
*   Ufunc 和 Numba

## 什么是 Numpy？

Numpy 是 python 的基础计算库。它支持 N 维数组，并提供简单高效的数组操作。

NumPy 是用 C 语言编写的算法库，它将数据存储在连续的内存块中，独立于其他内置的 python 对象，并且可以在没有任何类型检查或其他 Python 开销的情况下对该内存进行操作。NumPy 数组使用的内存也比内置 Python 序列少得多。

## 为什么要用 Numpy？

Python 最初不是为数值计算而设计的。由于 python 是解释型语言，它天生就比 c 等编译型语言慢。因此 numpy 填补了这一空白，以下是使用 numpy 的一些优势

*   它在内存和计算方面提供了高效的多维数组操作
*   它提供了对整个数组的快速数学运算，而不需要使用循环
*   它还提供与线性代数、统计学、傅立叶变换等相关的科学运算
*   它为 c 和 c++的互操作性提供了工具

## Numpy 怎么玩？

我将推荐两种使用 Numpy 的方法

*   [kaggle](https://www.kaggle.com/notebooks) 或 [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#) :你可以直接进入编码，不需要任何设置
*   [Jupyter Notebook](https://jupyter.org/install) :您需要安装 Jupyter Notebook，然后[使用 pip 安装 numpy](https://pypi.org/project/numpy/) 库(如果您有 anaconda 或 miniconda，numpy 可能已经安装)

## 遵循本教程

你可以在 kaggle 或 google colab 上分别通过分叉或复制以下笔记本来尝试这个教程

> **叉本** [***kaggle 笔记本***](https://www.kaggle.com/devkhadka/numpy-guide-book-for-beginners)
> 
> 或者
> 
> **复制本** [***colab 笔记本***](https://colab.research.google.com/drive/1RFpHPJ5ZzK2VHjgJP-o6WSOtM_SXnjbq)

## 关于 Jupyter 笔记本的一些提示

*   要查看自动完成建议**,请按** `**tab**`
*   要查看功能**的参数，在输入功能名称和'('后按** `**shift + tab**`。键入`np.asarray(`，然后按下`shift + tab`
*   要查看文档字符串，请使用“？”喜欢`**np.asarray?**`然后按`**shift + enter**`

**看够了！！** ***让我们把手弄脏***

# 创建数组

## 来自 python 列表

```
**import** **numpy** **as** **np**
print(np.array([1,2,3,4]))print('**\n**', 'array of 16 bit integers')
print(np.array([1,2,3,4], dtype=np.int16))print('**\n**', '2 dimensional array')
print(np.array([[1,2,3], [4,5,6]]))
```

*输出*

```
[1 2 3 4]array of 16 bit integers
[1 2 3 4]2 dimensional array
[[1 2 3]
 [4 5 6]]
```

## 努皮方法

```
print('Numpy array from range')
print(np.arange(3,8))print('**\n**', '2D 3X3 array of zeros')
print(np.zeros((3,3)))print('**\n**', '2D 2X3 array of ones')
print(np.ones((2,3)))print('**\n**', 'Triangular array with ones at and below diagonal')
print(np.tri(3, 4))print('**\n**', 'Index matrix with ones at diagonal')
print(np.eye(3))print('**\n**', '20 equally spaced values between 1 and 5')
print(np.linspace(1, 5, 20))
```

*输出*

```
Numpy array from range
[3 4 5 6 7]2D 3X3 array of zeros
[[0\. 0\. 0.]
 [0\. 0\. 0.]
 [0\. 0\. 0.]]2D 2X3 array of ones
[[1\. 1\. 1.]
 [1\. 1\. 1.]]Triangular array with ones at and below diagonal
[[1\. 0\. 0\. 0.]
 [1\. 1\. 0\. 0.]
 [1\. 1\. 1\. 0.]]Index matrix with ones at diagonal
[[1\. 0\. 0.]
 [0\. 1\. 0.]
 [0\. 0\. 1.]]20 equally spaced values between 1 and 5
[1\.         1.21052632 1.42105263 1.63157895 1.84210526 2.05263158
 2.26315789 2.47368421 2.68421053 2.89473684 3.10526316 3.31578947
 3.52631579 3.73684211 3.94736842 4.15789474 4.36842105 4.57894737
 4.78947368 5\.        ]
```

## 使用`np.random`

```
print('3X2 array of uniformly distributed number between 0 and 1')
print(np.random.rand(3,2))print('**\n**', 'Normally distributed random numbers with mean=0 and std=1')
print(np.random.randn(3,3))print('**\n**', 'Randomly choose integers from a range (>=5, <11)')
print(np.random.randint(5, 11, size=(2,2)))print('**\n**', "Randomly selects a permutation from array")
print(np.random.permutation([2,3,4,5,6]))print('**\n**', "This is equivalent to rolling dice 10 times and counting **\**
occurance of getting each side")
print(np.random.multinomial(10, [1/6]*6))
```

*输出*

```
3X2 array of uniformly distributed number between 0 and 1
[[0.99718301 0.46455866]
 [0.12057951 0.95932211]
 [0.22538176 0.99273413]]Normally distributed random numbers with mean=0 and std=1
[[-0.53815353  1.58638922  0.81410291]
 [ 0.01157038 -0.03269712 -1.16455499]
 [-0.3351507  -0.05698716  0.10403848]]Randomly choose integers from a range (>=5, <11)
[[ 7  9]
 [ 6 10]]Randomly selects a permutation from array
[3 4 2 5 6]This is equivalent to rolling dice 10 times and counting occurance of getting each side
[3 4 1 2 0 0]
```

## 了解 Numpy 数组的结构(维度、形状和步距)

```
import numpy as np
arr = np.array([[1,2,3], [2,3,1], [3,3,3]])print('Number of array dimensions')
print(arr.ndim)print('\nShape of array is tuple giving size of each dimension')
print(arr.shape)print('\nstrides gives byte steps to be moved in memory to get to next \
index in each dimension')
print(arr.strides)print('\nByte size of each item')
print(arr.itemsize)
```

*输出*

```
Number of array dimensions
2Shape of array is tuple giving size of each dimension
(3, 3)strides gives byte steps to be moved in memory to get to next index in each dimension
(24, 8)Byte size of each item
8
```

## 更多关于跨步

```
print('Slice indexing is done by changing strides, as in examples below')print('Strides of original array')
print(arr.strides)print('\n', 'Slice with step of 2 is done by multiplying stride(byte step size) by 2 in that dimension')
print(arr[::2].strides)print('\n', 'Reverse index will negate the stride')
print(arr[::-1].strides)print('\n', 'Transpose will swap the stride of the dimensions')
print(arr.T.strides)
```

*输出*

```
Slice indexing is done by changing strides, as in examples below
Strides of original array
(24, 8)Slice with step of 2 is done by multiplying stride(byte step size) by 2 in that dimension
(48, 8)Reverse index will negate the stride
(-24, 8)Transpose will swap the stride of the dimensions
(8, 24)
```

## 一些步幅技巧:改变步幅的内积

您可能很少想要使用这些技巧，但是它有助于我们理解 numpy 中的索引是如何工作的

`as_strided`函数返回一个具有不同步幅和形状的数组视图

```
from numpy.lib.stride_tricks import as_stridedarr1 = np.arange(5)
print('arr1: ', arr1)arr2 = np.arange(3)
print('arr2: ', arr2)print('\n', 'Adding a dimension with stride 0 allows us to repeat array in that dimension without making copy')print('\n', 'Making stride 0 for rows repeats rows.')
print('As step size is zero to move to next row it will give same row repeatedly')
r_arr1 = as_strided(arr1, strides=(0,arr1.itemsize), shape=(len(arr2),len(arr1)))
print(r_arr1)print('\n', 'Making stride 0 for columns repeats columns.')
r_arr2 = as_strided(arr1, strides=(arr2.itemsize, 0), shape=(len(arr2),len(arr1)))
print(r_arr2, '\n')print('Inner product: product of every value of arr1 to every value of arr2')
print(r_arr1 * r_arr2)
```

*输出*

```
arr1:  [0 1 2 3 4]
arr2:  [0 1 2]Adding a dimension with stride 0 allows us to repeat array in that dimension without making copyMaking stride 0 for rows repeats rows.
As step size is zero to move to next row it will give same row repeatedly
[[0 1 2 3 4]
 [0 1 2 3 4]
 [0 1 2 3 4]]Making stride 0 for columns repeats columns.
[[0 0 0 0 0]
 [1 1 1 1 1]
 [2 2 2 2 2]]Inner product: product of every value of arr1 to every value of arr2
[[0 0 0 0 0]
 [0 1 2 3 4]
 [0 2 4 6 8]]
```

**利用广播**

```
print('Above example is equivalent to using broadcast to do inner product')
print(arr1[np.newaxis, :] * arr2[:, np.newaxis])print('arr1[np.newaxis, :].strides => ', arr1[np.newaxis, :].strides)
print('arr2[:, np.newaxis].strides => ', arr2[:, np.newaxis].strides)
```

*输出*

```
Above example is equivalent to using broadcast to do inner product
[[0 0 0 0 0]
 [0 1 2 3 4]
 [0 2 4 6 8]]
arr1[np.newaxis, :].strides =>  (0, 8)
arr2[:, np.newaxis].strides =>  (8, 0)
```

# 数据类型和转换

**注释**

*   Numpy 数组只能存储一种数据类型的项
*   `np_array.dtype`属性将给出数组的 dtype
*   下表显示了一些常见的数据类型及其字符串名称

```
Numpy Attribute                                 | String Name                | Description
------------------------------------------------------------------------------------------------------
np.int8, np.int16, np.int32, np.int64           | '<i1', '<i2', '<i4', '<i8' | signed int
np.uint8, np.uint16, np.uint32, np.uint64       | '<u1', '<u2', '<u4', '<u8' | unsigned int
np.float16, np.float32, np.float64, np.float128 | '<f2', '<f4', '<f8', '<f16'| floats
np.string_                                      |'S1', 'S10', 'S255'         | string of bytes (ascii)
np.str                                          |'U1', 'U255'                | string of unicode characters
np.datetime64                                   |'M8'                        | date time
np.Object                                       |'O'                         | python object
np.bool                                         |'?'                         | boolean
```

*   **分解字符串名'< u8':** 这里'<'表示小端字节顺序，' u '表示无符号整数，' 8 '表示 8 个字节。字节顺序的其他选项有'>'大端'和' = '系统默认值
*   上面讨论的所有数组初始化函数都采用“dtype”参数来设置数组的数据类型，例如:`np.random.randint(5, 11, size=(2,2), dtype=np.int8)`

## 铸造

```
import numpy as np
arr = np.arange(5, dtype='<f4')
print('arr: ', arr)print('\n', 'Cast to integer using astype function which will make copy of the array')
display(arr.astype(np.int8))print('\n', 'By default casting is unsafe which will ignore the overflow. e.g. `2e10` is converted to 0')
arr[3] = 2e10
print(arr.astype('<i1'))print('\n', 'Casting from string to float')
sarr = np.array("1 2 3 4 5.0".split())
print(sarr)
print(sarr.astype('<f4'))print('\n', 'Use casting="safe" for doing safe casting, which will raise error if overflow')
*# print(arr.astype('<i1', casting='safe'))*
```

*输出*

```
arr:  [0\. 1\. 2\. 3\. 4.]Cast to integer using astype function which will make copy of the arrayarray([0, 1, 2, 3, 4], dtype=int8)By default casting is unsafe which will ignore the overflow. e.g. `2e10` is converted to 0
[0 1 2 0 4]Casting from string to float
['1' '2' '3' '4' '5.0']
[1\. 2\. 3\. 4\. 5.]Use casting="safe" for doing safe casting, which will raise error if overflow---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-8-3f587ee2f6f0> in <module>()
     16 
     17 print('\n', 'Use casting="safe" for doing safe casting, which will raise error if overflow')
---> 18 print(arr.astype('<i1', casting='safe'))TypeError: Cannot cast array from dtype('float32') to dtype('int8') according to the rule 'safe'
```

## 重塑

*   只要元素总数不变，数组就可以改变成任何形状

在[9]中:

```
arr = np.arange(20)
print('arr: ', arr)print('**\n**', 'reshape 1D arr of length 20 to shape (4,5)')
print(arr.reshape(4,5))print('**\n**', 'One item of shape tuple can be -1 in which case the item will be calculated by numpy')
print('For total size to be 20 missing value must be 5')
print(arr.reshape(2,2,-1))arr:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19] reshape 1D arr of length 20 to shape (4,5)
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]] One item of shape tuple can be -1 in which case the item will be calculated by numpy
For total size to be 20 missing value must be 5
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]] [[10 11 12 13 14]
  [15 16 17 18 19]]]
```

## 具有不同数据类型的数组视图

*   `arr.view()`方法用新的数据类型给出相同数据的新视图。用不同的数据类型创建视图与强制转换不同。例如，如果我们有 NP . float 32(<F4’)的 ndarray，用 dtype byte(<i8’)创建视图将读取 4 字节浮点数据作为单个字节

```
arr = np.arange(5, dtype='<i2')
print('arr: ', arr)print('\n', 'View with dtype "<i1" for array of dtype "<i2" will breakdown items to bytes')
print(arr.view('<i1'))print('\n', 'Changing little-endian to big-endian will change value as they use different byte order')
print(arr.view('>i2'))print('\n', 'Following will give individual bytes in memory of each items')
arr = np.arange(5, dtype='<f2')
print(arr)
print(arr.view('<i1'))
```

*输出*

```
arr:  [0 1 2 3 4]View with dtype "<i1" for array of dtype "<i2" will breakdown items to bytes
[0 0 1 0 2 0 3 0 4 0]Changing little-endian to big-endian will change value as they use different byte order
[   0  256  512  768 1024]Following will give individual bytes in memory of each items
[0\. 1\. 2\. 3\. 4.]
[ 0  0  0 60  0 64  0 66  0 68]
```

# 索引方法

## 整数和切片索引

*   这种索引方法类似于 python list 中使用的索引
*   切片总是为数组 ie 创建视图。不复制数组

```
import numpy as np
arr = np.arange(20)
print("arr: ", arr)print('\n', 'Get item at index 4(5th item) of the array')
print(arr[4])print('\n', 'Assign 0 to index 4 of array')
arr[4] = 0
print(arr)print('\n', 'Get items in the index range 4 to 10 not including 10')
print(arr[4:10])print('\n', 'Set 1 to alternate items starting at index 4 to 10 ')
arr[4:10:2] = 1
print(arr)
```

*输出*

```
arr:  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]Get item at index 4(5th item) of the array
4Assign 0 to index 4 of array
[ 0  1  2  3  0  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]Get items in the index range 4 to 10 not including 10
[0 5 6 7 8 9]Set 1 to alternate items starting at index 4 to 10 
[ 0  1  2  3  1  5  1  7  1  9 10 11 12 13 14 15 16 17 18 19]
```

## 2D 数组中的切片索引

*   对于多维数组，切片索引可以用逗号分隔

```
arr = np.arange(20).reshape(4,5)print('arr:\n', arr)print('\n', 'Set 0 to first 3 rows and and last 2 columns')
arr[:3, -2:] = 1
print(arr)
```

*输出*

```
arr:
 [[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]Set 0 to first 3 rows and and last 2 columns
[[ 0  1  2  1  1]
 [ 5  6  7  1  1]
 [10 11 12  1  1]
 [15 16 17 18 19]]
```

## 布尔索引

*   与原始数组形状相同(或可扩展为该形状)的布尔数组可用作索引。这将选择索引值为真的项目
*   布尔数组也可以用来过滤具有特定条件的数组
*   布尔索引**将向数组返回副本**而不是视图

```
arr = np.arange(6).reshape(2,3)
print('arr:\n', arr)print('\n', 'Following index will gives last two items of 1st row and 1st element of 2nd row')
indx = np.array([[False, True, True], [True, False,False]])
arr[indx]print('\n', 'Boolean index to filter values greater than 3 from arr')
filter_indx = arr>3
print('Filter Index:\n', filter_indx)print('\n', 'Set 3 to values greater than 3 in arr')
arr[filter_indx] = 3
print(arr)
```

*输出*

```
arr:
 [[0 1 2]
 [3 4 5]]Following index will gives last two items of 1st row and 1st element of 2nd rowBoolean index to filter values greater than 3 from arr
Filter Index:
 [[False False False]
 [False  True  True]]Set 3 to values greater than 3 in arr
[[0 1 2]
 [3 3 3]]
```

## 花式索引

*   花式索引是指使用索引数组(整数)作为索引，以获得所有项目一次
*   花式索引**也将返回副本**而不是视图到数组

```
import numpy as nparr = np.arange(10)
print('arr:\n', arr)print('\n', 'Get items at indexes 3,5 and 7 at once')
print(arr[[3,5,7]])print('\n', 'Sorting arr based on another array "values"')
np.random.seed(5)
values = np.random.rand(10)
print('values:\n', values)
print('\n', 'np.argsort instead of returning sorted values will return array of indexes which will sort the array')
indexes = np.argsort(values) 
print('indexes:\n', indexes)
print('Sorted array:\n', arr[indexes])print('\n', 'You can also use fancy indexing to get same item multiple times')
print(arr[[0,1,1,2,2,2,3,3,3,3]])
```

*输出*

```
arr:
 [0 1 2 3 4 5 6 7 8 9]Get items at indexes 3,5 and 7 at once
[3 5 7]Sorting arr based on another array "values"
values:
 [0.22199317 0.87073231 0.20671916 0.91861091 0.48841119 0.61174386
 0.76590786 0.51841799 0.2968005  0.18772123]np.argsort instead of returning sorted values will return array of indexes which will sort the array
indexes:
 [9 2 0 8 4 7 5 6 1 3]
Sorted array:
 [9 2 0 8 4 7 5 6 1 3]You can also use fancy indexing to get same item multiple times
[0 1 1 2 2 2 3 3 3 3]
```

## 元组索引

*   可以使用等长的整数数组元组来索引多维数组，其中元组中的每个数组将索引相应的维
*   如果元组中索引数组的数量小于被索引数组的维度，则它们将被用于索引更低的维度(即从 0 到元组长度的维度)

```
arr2 = np.arange(15).reshape(5,3)
print('arr2:\n', arr2)print('\n', 'Will give items at index (4,0) and (1,2)')
indx = ([4,1],[0,2])
print(arr2[indx])print('\n', 'Tuple of length one will return rows')
indx = ([4,1],)
print(arr2[indx])
```

*输出*

```
arr2:
 [[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]
 [12 13 14]]Will give items at index (4,0) and (1,2)
[12  5]Tuple of length one will return rows
[[12 13 14]
 [ 3  4  5]]
```

## 带有高级索引的赋值

*   高级索引(即布尔、花式和元组)将副本而不是视图返回到索引数组。但是使用那些索引直接赋值会改变原来的数组，这个特性是为了方便。但是，如果我们链接索引，它可能会以一种似乎有些出乎意料的方式运行

```
import numpy as nparr = np.arange(10)
print('arr: ', arr)print('\n', 'Direct assignment will change the original array')
arr[[3,5,7]] = -1
print(arr)print('\n', 'When we chain the indexing it will not work')
arr[[3,5,7]][0] = -2
print(arr)print('\n', 'But chaining index will work with slicing indexing')
arr[3:8:2][0] = -2
print(arr)
```

*输出*

```
arr:  [0 1 2 3 4 5 6 7 8 9]Direct assignment will change the original array
[ 0  1  2 -1  4 -1  6 -1  8  9]When we chain the indexing it will not work
[ 0  1  2 -1  4 -1  6 -1  8  9]But chaining index will work with slicing indexing
[ 0  1  2 -2  4 -1  6 -1  8  9]
```

## 混合索引

*   在多维数组中，我们可以同时对每个维度使用不同的索引方法(切片、布尔和花式)
*   为了混合使用布尔和花式索引，布尔索引中 True 的数量必须等于花式索引的长度

```
arr = np.arange(64).reshape(4,4,4)
print('arr: ', arr)
print('\n', 'Following mixed indexing will select 1st and 3rd item in 0th dimension')
print('and item at index 0 and 2 at 1st dimension and item at index >=2')
print(arr[[True, False, True, False], [0,2], 2:])
```

输出

```
arr:  [[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]
  [12 13 14 15]][[16 17 18 19]
  [20 21 22 23]
  [24 25 26 27]
  [28 29 30 31]][[32 33 34 35]
  [36 37 38 39]
  [40 41 42 43]
  [44 45 46 47]][[48 49 50 51]
  [52 53 54 55]
  [56 57 58 59]
  [60 61 62 63]]]Following mixed indexing will select 1st and 3rd item in 0th dimension
and item at index 0 and 2 at 1st dimension and item at index >=2
[[ 2  3]
 [42 43]]
```

# 数组运算

## 简单的数组运算

*   Numpy 提供了简单的语法来在兼容形状的数组之间执行数学和逻辑运算。这里兼容的形状意味着，一个数组的形状可以使用广播规则扩展以匹配另一个数组的形状，我们将在下面讨论
*   在本节中，我们将只看到两种情况
*   数组具有相同的形状，在这种情况下，操作将是元素方式的
*   其中一个操作数是标量，在这种情况下，操作将在标量和数组的每个元素之间进行
*   数组之间的这些操作称为向量化，比使用循环的相同操作要快得多。
*   矢量化速度更快，因为它是用 C 实现的，没有类型检查等开销。

```
import numpy as npprint('Evaluate expression (x1*x2 - 3*x1 + 30) for x1 and x2 in range 0 to 10')
x1 = np.linspace(0,10,20)
x2 = np.linspace(0, 10, 20)
z = x1*x2 - 3*x1 + 30
print(z)print('\n', 'Spatial distance between corresponding points in two array')
p1 = np.random.rand(20,2)
p2 = np.random.rand(20,2)*'''np.sum will add values along given axis (dimension). If shape of array is (3,4,5)')*
*then axis 0,1 and 2 corresponds to dimension with length 3, 4 and 5 respectively'''*
d = np.sum((p1-p2)**2, axis=1)
print(d)print('\n', 'Element wise comparison, ">=" will give boolean array with True where element')
print('of p2 is greater than or equal to p1')
r = p2>=p1
print(r)print('\n', 'Element wise logical operation, "&" will give True where point of p2 is ahead')
print('in both x and y direction from corresponding point in p1')
print(r[:,0] & r[:,1])
```

*输出*

```
Evaluate expression (x1*x2 - 3*x1 + 30) for x1 and x2 in range 0 to 10
[ 30\.          28.69806094  27.9501385   27.75623269  28.11634349
  29.03047091  30.49861496  32.52077562  35.09695291  38.22714681
  41.91135734  46.14958449  50.94182825  56.28808864  62.18836565
  68.64265928  75.65096953  83.2132964   91.32963989 100\.        ]Spatial distance between corresponding points in two array
[0.54052263 0.17505988 0.59108818 0.41593393 0.03548522 0.29946201
 0.84649163 0.24975051 0.90016153 0.54062043 0.00097261 0.39826495
 0.64710327 0.40655563 0.00531519 0.94567232 0.33333277 0.01713418
 0.53797027 0.48080742]Element wise comparison, ">=" will give boolean array with True where element
of p2 is greater than or equal to p1
[[ True False]
 [False False]
 [False False]
 [ True  True]
 [ True False]
 [ True  True]
 [ True  True]
 [ True False]
 [ True False]
 [False  True]
 [ True False]
 [False False]
 [ True False]
 [ True False]
 [False False]
 [ True False]
 [False False]
 [ True False]
 [False  True]
 [False False]]Element wise logical operation, "&" will give True where point of p2 is ahead
in both x and y direction from corresponding point in p1
[False False False  True False  True  True False False False False False
 False False False False False False False False]
```

## 数组运算的函数

*   Numpy 也有上述操作的函数版本，如`np.add, np.substract, np.divide, np.greater_equal, np.logical_and`等等
*   我们在上一节看到的使用+、*等运算符的数组运算是函数运算的运算符重载版本
*   函数版本的操作会给我们额外的参数来定制，其中一个常用的参数是`out`。默认情况下是`None`，这将为结果创建一个新的数组。
*   如果我们将一个形状和数据类型与预期结果匹配的数组传递给`out`，参数结果将被填充到传递的数组中。如果我们做多个操作，这将是有效的内存方式

在[19]中:

```
import numpy as npprint('Evaluate expression (x1*x2 - 3*x1 + 30) using functions')
x1 = np.linspace(0,10,20)
x2 = np.linspace(0, 10, 20)*'''Create empty output array with expected shape'''*
z = np.empty_like(x1)*'''Code is not very clean as using operator but it will perform very well memory wise'''*
np.multiply(x1, x2, out=z)
np.subtract(z, 3*x1, out=z)
np.add(z, 30, out=z)
print(z)
```

*输出*

```
Evaluate expression (x1*x2 - 3*x1 + 30) using functions
[ 30\.          28.69806094  27.9501385   27.75623269  28.11634349
  29.03047091  30.49861496  32.52077562  35.09695291  38.22714681
  41.91135734  46.14958449  50.94182825  56.28808864  62.18836565
  68.64265928  75.65096953  83.2132964   91.32963989 100\.        ]
```

# 广播

## 广播规则

当在形状不完全匹配的两个数组之间进行数组操作时，只需几个简单的步骤，就可以改变形状，使它们相互匹配，如果它们是兼容的话。

1.  **检查数组尺寸**:如果尺寸不匹配，添加到较小尺寸数组的左边
2.  **匹配每个维度上的形状**:如果任何维度上的形状不匹配，并且其中一个数组的形状为 1，则重复该操作以匹配该维度上其他数组的形状
3.  **如果尺寸和形状不匹配，则引发错误**:如果尺寸和形状在此步骤之前不匹配，则引发错误

## 让我们用自定义实现来可视化广播规则

让我们进行自定义实现，用代码来直观显示广播规则是如何工作的

```
import numpy as nparr1 = np.arange(20).reshape(10,2)
arr2 = np.random.rand(2)
arr3 = arr2.copy()print('arr1.shape: ', arr1.shape)
print('arr2.shape: ', arr2.shape)*# Step 1: Check Array Dimensions*
print('\n', 'arr1 has dimension 2 and arr2 has dimension 1, so add 1 dimension to\
left side of arr2')
*# np.newaxis is convenient way of adding new dimension*
arr2 = arr2[np.newaxis, :]
print('arr1.shape: ', arr1.shape)
print('arr2.shape: ', arr2.shape)*# Step 2: Match Shape On Each Dimension*
print('\n', 'Now in axis=0 arr1 has 10 items and arr2 has one item, so repeat it 10\
times to match arr2')
arr2 = np.repeat(arr2, 10, axis=0)
print('arr1.shape: ', arr1.shape)
print('arr2.shape: ', arr2.shape)print('\n', 'Now both array has same dimension and shape, we can multiply them')
print('arr1*arr2:\n', arr1*arr2)print('\n', 'Lets see if broadcasting also produce same result')
print('arr1*arr3:\n', arr1*arr3)
```

*输出*

```
arr1.shape:  (10, 2)
arr2.shape:  (2,)arr1 has dimension 2 and arr2 has dimension 1, so add 1 dimension toleft side of arr2
arr1.shape:  (10, 2)
arr2.shape:  (1, 2)Now in axis=0 arr1 has 10 items and arr2 has one item, so repeat it 10times to match arr2
arr1.shape:  (10, 2)
arr2.shape:  (10, 2)Now both array has same dimension and shape, we can multiply them
arr1*arr2:
 [[ 0\.          0.11111075]
 [ 1.71941377  0.33333225]
 [ 3.43882755  0.55555375]
 [ 5.15824132  0.77777525]
 [ 6.8776551   0.99999675]
 [ 8.59706887  1.22221824]
 [10.31648264  1.44443974]
 [12.03589642  1.66666124]
 [13.75531019  1.88888274]
 [15.47472397  2.11110424]]Lets see if broadcasting also produce same result
arr1*arr3:
 [[ 0\.          0.11111075]
 [ 1.71941377  0.33333225]
 [ 3.43882755  0.55555375]
 [ 5.15824132  0.77777525]
 [ 6.8776551   0.99999675]
 [ 8.59706887  1.22221824]
 [10.31648264  1.44443974]
 [12.03589642  1.66666124]
 [13.75531019  1.88888274]
 [15.47472397  2.11110424]]
```

## 让我们试几个形状的例子

您可以通过创建给定形状的数组并在它们之间进行一些操作来尝试

```
Before Broadcast        |Step 1                      | Step 2 and 3  
Shapes of arr1 and arr2 |                            | Shapes of result 
-------------------------------------------------------------------------
(3, 1, 5); (4, 1)       | (3, 1, 5); (1, 4, 1)       | (3, 4, 5)        
(10,); (1, 10)          | (10, 1); (1, 10)           | (10, 10)         
(2, 2, 2); (2, 3)       | (2, 2, 2); (1, 2, 3)       | Not Broadcastable
(2, 2, 2, 1); (2, 3)    | (2, 2, 2, 1); (1, 1, 2, 3) | (2, 2, 2, 2, 3)
```

# 广播的一些用法

## 使用广播评估线性方程

```
print("Let's evaluate equation c1*x1 + c2*x2 + c3*x3 for 100 points at once")
points = np.random.rand(100,3)
coefficients = np.array([5, -2, 11])
results = np.sum(points*coefficients, axis=1)
print('results first 10:**\n**', results[:10])
print('results.shape: ', results.shape)
```

*输出*

```
Let's evaluate equation c1*x1 + c2*x2 + c3*x3 for 100 points at once
results first 10:
 [ 6.35385279  0.85639146 12.87683079  5.99433896  4.50873972 10.44691041
  3.87407211  6.62954602 11.00386582 10.09247866]
results.shape:  (100,)
```

## 寻找数组之间的公共元素

```
np.random.seed(5)
*## Get 20 random value from 0 to 99*
arr1 = np.random.choice(50, 20, replace=False)
arr2 = np.random.choice(50, 15, replace=False)
print("arr1: ", arr1)
print("arr2: ", arr2)
print('\n', 'arr1 and arr2 are 1D arrays of length 20, 15 respectively.')
print('To make them broadcastable Change shape of arr2 to (15, 1)')
arr2 = arr2.reshape(15, 1)
print('\n', 'Then both arrays will be broadcasted to (15, 20) matrix with all possible pairs')
comparison = (arr1 == arr2)
print('\n', 'comparison.shape: ', comparison.shape)
print('\n', 'Elements of arr1 also in arr2: ', arr1[comparison.any(axis=0)])
```

输出

```
arr1:  [42 29  6 19 28 17  2 43  3 21 31  4 32  0 23  5 48 34 37 26]
arr2:  [40 37 41 48  4 20 10 18 34 28 19 32 17 22 23]arr1 and arr2 are 1D arrays of length 20, 15 respectively.
To make them broadcastable Change shape of arr2 to (15, 1)Then both arrays will be broadcasted to (15, 20) matrix with all possible pairscomparison.shape:  (15, 20)Elements of arr1 also in arr2:  [19 28 17  4 32 23 48 34 37]
```

## 查找 k 个最近邻

```
import numpy as npnp.random.seed(5)points = np.random.rand(20, 2)
print('To calculate distance between every pair of points make copy of points ')
print('with shape (20, 1, 2) which will broadcast both array to shape (20, 20, 2)', '\n')
cp_points = points.reshape(20, 1, 2)*## calculate x2-x1, y2-y1*
diff = (cp_points - points)
print('diff.shape: ', diff.shape)*## calculate (x2-x1)**2 + (y2-y1)***
distance_matrix = np.sum(diff**2, axis=2)
print('distance_matrix.shape: ', distance_matrix.shape, '\n')*## sort by distance along axis 1 and take top 4, one of which is the point itself*
top_3 = np.argsort(distance_matrix, axis=1)[:,:4]
print("Get the points with it's 3 nearest neighbors")
points[top_3]
```

*输出*

```
To calculate distance between every pair of points make copy of points 
with shape (20, 1, 2) which will broadcast both array to shape (20, 20, 2)diff.shape:  (20, 20, 2)
distance_matrix.shape:  (20, 20)Get the points with it's 3 nearest neighborsarray([[[0.22199317, 0.87073231],
        [0.20671916, 0.91861091],
        [0.16561286, 0.96393053],
        [0.08074127, 0.7384403 ]], [[0.20671916, 0.91861091],
        [0.22199317, 0.87073231],
        [0.16561286, 0.96393053],
        [0.08074127, 0.7384403 ]], [[0.48841119, 0.61174386],
        [0.62878791, 0.57983781],
        [0.69984361, 0.77951459],
        [0.76590786, 0.51841799]], [[0.76590786, 0.51841799],
        [0.62878791, 0.57983781],
        [0.69984361, 0.77951459],
        [0.87993703, 0.27408646]], [[0.2968005 , 0.18772123],
        [0.32756395, 0.1441643 ],
        [0.28468588, 0.25358821],
        [0.44130922, 0.15830987]], [[0.08074127, 0.7384403 ],
        [0.02293309, 0.57766286],
        [0.22199317, 0.87073231],
        [0.20671916, 0.91861091]], [[0.44130922, 0.15830987],
        [0.32756395, 0.1441643 ],
        [0.41423502, 0.29607993],
        [0.2968005 , 0.18772123]], [[0.87993703, 0.27408646],
        [0.96022672, 0.18841466],
        [0.76590786, 0.51841799],
        [0.5999292 , 0.26581912]], [[0.41423502, 0.29607993],
        [0.28468588, 0.25358821],
        [0.44130922, 0.15830987],
        [0.2968005 , 0.18772123]], [[0.62878791, 0.57983781],
        [0.48841119, 0.61174386],
        [0.76590786, 0.51841799],
        [0.69984361, 0.77951459]], [[0.5999292 , 0.26581912],
        [0.41423502, 0.29607993],
        [0.44130922, 0.15830987],
        [0.87993703, 0.27408646]], [[0.28468588, 0.25358821],
        [0.2968005 , 0.18772123],
        [0.32756395, 0.1441643 ],
        [0.41423502, 0.29607993]], [[0.32756395, 0.1441643 ],
        [0.2968005 , 0.18772123],
        [0.44130922, 0.15830987],
        [0.28468588, 0.25358821]], [[0.16561286, 0.96393053],
        [0.20671916, 0.91861091],
        [0.22199317, 0.87073231],
        [0.08074127, 0.7384403 ]], [[0.96022672, 0.18841466],
        [0.87993703, 0.27408646],
        [0.5999292 , 0.26581912],
        [0.76590786, 0.51841799]], [[0.02430656, 0.20455555],
        [0.28468588, 0.25358821],
        [0.2968005 , 0.18772123],
        [0.32756395, 0.1441643 ]], [[0.69984361, 0.77951459],
        [0.62878791, 0.57983781],
        [0.63979518, 0.9856244 ],
        [0.76590786, 0.51841799]], [[0.02293309, 0.57766286],
        [0.00164217, 0.51547261],
        [0.08074127, 0.7384403 ],
        [0.22199317, 0.87073231]], [[0.00164217, 0.51547261],
        [0.02293309, 0.57766286],
        [0.08074127, 0.7384403 ],
        [0.02430656, 0.20455555]], [[0.63979518, 0.9856244 ],
        [0.69984361, 0.77951459],
        [0.48841119, 0.61174386],
        [0.62878791, 0.57983781]]])
```

# …向量化…

*   在 numpy 中，向量化意味着对相同类型的数据序列执行优化操作。
*   除了具有清晰的代码结构，矢量化操作也非常高效，因为代码是编译的，避免了 python 的开销，如类型检查、内存管理等。
*   我们在上面的**广播部分看到的例子也是矢量化的好例子**

## 矢量化与循环

假设我们有一个像`a1*x + a2*x^2 + a3*x^3 ... + a10*x^10`这样的单变量 10 次多项式方程。让我们尝试仅使用 python 和 numpy 矢量化来评估大量 x 的方程，看看它们是如何比较的

```
def evaluate_polynomial_loop():
  result_loop = np.empty_like(X)
  for i in range(X.shape[0]):
    exp_part = 1
    total = 0
    for j in range(coefficients.shape[0]):
      exp_part *= X[i]
      total+=coefficients[j]*exp_part
    result_loop[i] = total
  return result_loopdef evaluate_polynomial_vect():
  ## repeates x's in 10 columns
  exponents = X[:, np.newaxis] + np.zeros((1, coefficients.shape[0]))
  exponents.cumprod(axis=1, out=exponents)
  result_vect = np.sum(exponents * coefficients, axis=1)
  return result_vect

print('Verify that both gives same result')
print('Loop:\n', evaluate_polynomial_loop()[:10])
print('Vectorization:\n', evaluate_polynomial_vect()[:10])
```

*输出*

```
Verify that both gives same result
Loop:
 [222.57782534  30.62439847  59.69953776 373.52687079 123.89007218
 179.70369976   6.49315699 321.685257    73.14575517  69.71437596]
Vectorization:
 [222.57782534  30.62439847  59.69953776 373.52687079 123.89007218
 179.70369976   6.49315699 321.685257    73.14575517  69.71437596]
```

## 比较

为了公平比较，我在两者中都使用了 numpy 数组，它的索引比 python list 快得多。通过比较，我们看到矢量化比**快了大约 80 倍**

**循环**

```
%timeit evaluate_polynomial_loop()
```

*输出*

```
113 ms ± 3.82 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

**矢量化**

```
%**timeit** evaluate_polynomial_vect()
```

*输出*

```
1.22 ms ± 75 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

# Ufunc 和 Numba

**Ufunc:** 也称为通用函数，是函数的矢量化包装。Ufunc 可以在 ndarray 上运行，并支持广播和类型转换。etc 是用 c 实现的 Ufunc 的例子。我们可以使用`np.frompyfunc`或 numba 创建自定义的 Ufunc。

Numba 是一个即时编译器，它从纯 python 数组和数字函数中生成优化的机器代码。你可以在一个函数上使用`numba.jit`装饰，使得这个函数在第一次运行时就被编译。可以使用`numba.vectorize` decorator 将 python 函数转换为 Ufunc。

让我们比较添加两个大数组的不同实现，如下所示

## 创建大数组

```
arr1 = np.arange(1000000, dtype='int64')
arr2 = np.arange(1000000, dtype='int64')
```

## 使用 Python 循环实现

```
def add_arr(arr1, arr2):
  assert len(arr1)==len(arr2), "array must have same length"
  result = np.empty_like(arr1)
  for i in range(len(arr1)):
    result[i] = arr1[i] + arr2[i]
  return result%timeit _ = add_arr(arr1, arr2)
```

*输出*

```
563 ms ± 9.34 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

## 使用`np.frompyfunc`创建 Ufunc

```
def add(a, b):
  return a+bvect_add =  np.frompyfunc(add,2,1)%timeit _ = vect_add(arr1, arr2)
```

*输出*

```
197 ms ± 9.16 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

## 使用 Numba JIT

*   “nopython=True”表示如果无法转换，将所有代码转换为机器代码，并引发错误

```
import numba
@numba.jit(nopython=True)
def add_arr_jit(arr1, arr2):
  assert len(arr1)==len(arr2), "array must have same length"
  result = np.empty_like(arr1)
  for i in range(len(arr1)):
    result[i] = arr1[i] + arr2[i]
  return result_ = add_arr_jit(arr1, arr2)
%timeit _ = add_arr_jit(arr1, arr2)
```

*输出*

```
3.8 ms ± 455 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

## 使用`numba.vectorize`创建 Ufunc

*   “numba.vectorize”将被转换的函数的签名作为参数。int64(int64，int64)'表示接受 2 个' int64 '参数并返回' int64 '

```
import numba@numba.vectorize(['int64(int64,int64)'], nopython=True)
def vect_add(a, b):
  return a+b%timeit _ = vect_add(arr1, arr2)
```

*输出*

```
2.58 ms ± 309 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

**结论**方案使用`numba.jit and numba.vectorize`效果更好。您还可以检查 numpy 矢量化与这些相比如何

# 更多探索

## 一些有用的功能

*   `**np.where**` **:** 元素方面`if .. then .. Else`
*   `**np.select**` **:** 根据多个条件从多个数组中选择值
*   `**np.concatenate, np.vstack, np.r_, np.hstack, np.c_**` **:** 按行、按列或给定轴连接多个数组
*   `**np.ravel, np.flatten**` **:** 将多维数组转换为 1D 数组
*   `**np.roll**` **:** 将数组沿给定轴做圆周移动

## 集合操作

*   `**np.unique(x)**` **:** 给出数组中唯一元素的数组
*   `**Intersect1d(x, y)**` **:** 给出两个数组共有元素的 1D 数组
*   `**Union1d(x, y)**` **:** 给出两个数组中唯一元素的 1D 数组
*   `**In1d(x, y)**` **:** 检查 x 的每个元素是否也出现在 y 上，并返回长度等于 x 的布尔值数组
*   `**Setdiff1d(x, y)**` **:** 给出不在 y 中的 x 元素
*   `**Setxor1d(x, y)**` **:** 给出在 x 或 y 中的元素，但不同时在两者中

## 从/向磁盘保存和加载数组

*   `**np.save("filename.npy", x)**` **:** 保存单个数数组到磁盘
*   `**np.load("filename.npy")**` **:** 从磁盘加载单个数数组
*   `**np.savez("filename.npz", key1=arr1, key2=arr2)**` **:** 用给定的键保存多个数组
*   `**np.savetxt("filename.npy", x)**` **:** 将单个 numpy 数组作为分隔文本文件保存到磁盘
*   `**np.loadtxt("filename.npy")**` **:** 从文本文件加载单个 numpy 数组

# 存储器交换

要使用不适合 RAM 的大型 numpy 数组，可以使用 numpy.memmap 函数将数组映射到磁盘中的文件。它将只透明地加载当前操作所需的阵列段。

*   `**np.memmap(filename, dtype, mode, shape)**` **:** 创建给定文件的内存映射数组
*   `**mmap.flush()**` **:** 将内存中的所有更改刷新到磁盘

**谢谢**

衷心感谢您阅读博客。希望它对您在 numpy 中的启动和运行有所帮助。欢迎任何意见、建议和建设性的批评。如果你喜欢这些内容，请不要忘记鼓掌[👏👏👏](https://emojipedia.org/clapping-hands-sign/)