# 数据分析:NumPy 简介

> 原文：<https://medium.com/analytics-vidhya/data-analysis-introduction-to-numpy-612e7cfc0e71?source=collection_archive---------35----------------------->

![](img/33480543b169f5dd3e6ef89b79f593d9.png)

**数据分析**定义为对**数据**进行清理、转换和建模，以发现对业务决策有用的信息的过程。**数据分析**的目的是从**数据**中提取有用的信息，并根据**数据分析**做出决策，而 **NumPy** 在数据分析中起着重要的作用。

那么，什么是 **NumPy** ？

**NumPy** 代表**数值 Python** 。基本上，它是一个用于 [Python 编程语言](https://en.wikipedia.org/wiki/Python_(programming_language))的库，增加了对大型多维[数组](https://en.wikipedia.org/wiki/Array_data_structure)和[矩阵](https://en.wikipedia.org/wiki/Matrix_(math))的支持，以及对这些数组进行操作的大量[高级](https://en.wikipedia.org/wiki/High-level_programming_language) [数学](https://en.wikipedia.org/wiki/Mathematics) [函数](https://en.wikipedia.org/wiki/Function_(mathematics))。

# 装置

要安装您可以使用" *pip install numpy "。不过，*我会推荐您使用 [Anaconda 发行版](https://www.anaconda.com/distribution)——它包括 Python、NumPy 和其他科学计算和数据科学常用的包。可以像 [Jupyter Notebook](http://jupyter.org/) 或者 [Spyder](https://pythonhosted.org/spyder/) 一样使用 Python IDE ( [集成开发环境](https://en.wikipedia.org/wiki/Integrated_development_environment))(两者默认自带 Anaconda)。

# **入门**

完成安装后，您需要在 Python IDE 中导入 numpy 库。我们导入一个库，以便在我们的程序中使用库的特性。要导入，您需要编写以下代码:

```
import numpy as np
```

这就是导入 numpy 库所要做的一切，现在你可能想知道什么是‘as NP’？用于每次使用 numpy 时将 numpy 命令写成' np.command '而不是' numpy.command '！

# NumPy 数组

数组是我们使用这个库的主要原因，数组基本上是值的集合，可以有不同的维度。在 NumPy 中，数组的维数称为数组的**秩**。一维数组称为**向量**，二维数组称为**矩阵**。 **NumPy** 中的主要数据结构是 **ndarray** ，它是 N 维**数组**的简称。

## **创建 NumPy 数组**

我们可以通过转换一个列表或列表的列表来创建一个 numpy 数组。这不是唯一的方法，我们可以建立 numpy 数组，有许多内置的方法，如 arange，linspace，zeros，one 和 random 来创建随机数数组。现在，我们将首先创建一个列表:

```
list = [1,2,3,4,5,6,7,8,9]
```

将列表转换为 numpy 数组:

```
numpy_array = np.array(my_list)
numpy_array               #1-D array
```

输出:

```
array([1, 2, 3, 4, 5, 6, 7, 8, 9]) 
```

现在，

```
list2 = [[1,2,3],[4,5,6],[7,8,9]]
numpy_twod = np.array(list2)
numpy_twod
```

输出将是一个二维 numpy 数组:

```
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
```

## **属性和方法**

您可以在 numpy 数组上使用各种属性和方法，如 **reshape** ，它返回一个包含相同数据的数组，但形状不同。在这里，我们转换了一个

```
num_arr = numpy_array.reshape(3,3)
num_arr
```

输出将是:

```
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
```

就像你可以使用几个方法和属性，例如， **arr 是 numpy 数组**

**方法:**

**arr.min()，arr.max()—** 用于查找数组中的最大和最小元素

**arr.argmin()，arr.argmax() —** 用于查找数组中最大和最小元素的索引

**arr.astype(dtype) —** 用于数据类型**的转换。**

**arr.tolist() —** 将 numpy 数组转换为 Python 列表。

**arr.sort()** —根据用户进行排序。

**属性:**

**arr.size —** 给出数组中元素的个数。

**arr.shape —** 数组所具有的决定形状的属性。

**arr.dtype —** 确定对象的数据类型。

# 索引和切片

## 支架索引和选择

挑选数组中一个或几个元素的最简单方法是:

让我们用 index : array[index]获取一个元素

```
l1 = [1,2,3,4,5,6,7,8,9,10]
arr = np.arange(l1)
arr[4] Output: 5         #will return the element in the 4th index
-------------------------------------
arr[1:5]Output: array([2, 3, 4, 5])   #Get values in a range
```

## 广播

Numpy 数组不同于普通的 Python 列表，因为它们具有广播的能力。

在 1D 阵列中:

```
arr[0:5]=100Output: array([100, 100, 100, 100, 100, 6, 7, 8, 9, 10])
---------------------------------------
new_arr = arr[3:7] 
new_arrOutput: array([100, 100, 6, 7]) #new array with sliced element  ---------------------------------------
new_arr[:] = 50  #Select all elements in array
new_arrOutput: array([50, 50, 50, 50])
--------------------------------------- 
#Now,if you check the 'arr' array, the changes also occurred in the original arrayarroutput: array([100, 100, 100, 50, 50, 50, 50, 8, 9, 10])#Data is not copied to make a new array, it's a view of the original array! This avoids memory problems! 
---------------------------------------
#to make a explicit array we have to use copy method 
#To get a copy
arr_copy = arr.copy()arr_copy
output: array([100, 100, 100, 50, 50, 50, 50, 8, 9, 10])
---------------------------------------
```

在 2D 阵列中:

**arr[row][col]** 或 **arr[row，col]** 是二维数组的两种格式，可用于返回 numpy 数组中的元素。我们也可以以 **arr[:，:]** 的方式使用索引来索引值。

**arr[row_index] —** 仅针对 row，您可以在这方面进行更多探索。

## 选择

基于条件运算符选择元素:

```
arr = np.arange(1,11) # created a array using arange
arroutput: array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
--------------------------------
arr > 6output: array([False, False, False, False,  Flase,  False,  True,  True,  True,  True], dtype=bool)
--------------------------------
bool_arr = arr>4
bool_arroutput: array([False, False, False, False,  True,  True,  True,  True,  True,  True], dtype=bool)
--------------------------------
arr[bool_arr]output: array([ 5,  6,  7,  8,  9, 10])
--------------------------------
```

太棒了。！你做了很多，这是伟大的工作

# 数字运算

## 算术

您可以轻松地用数组算法执行数组运算，或者用数组算法执行标量运算。让我们试一些例子:

```
*arr = np.arange(0,10) # create a new array
arr
output: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
------------------------------ 
arr + arr* #Additionoutput: array([ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
-----------------------------
arr — arr #Substraction*output: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
-----------------------------
arr * arr* #Muliplication*output: array([ 0, 1, 4, 9, 16, 25, 36, 49, 64, 81])
-----------------------------*
```

你也可以除法和使用指数，也有算术运算的方法:

**np.add(arr，10) —** 将为数组中的每个元素加 10

**np.add(arr1，arr 2)——**将把数组 2 加到数组 1 中。 **np.subtract()、np.divide()、np.power()和 np.multiply()** 也是如此。

您还可以让 NumPy 从数组中返回不同的值，比如:

**np.sqrt(arr) —** 将返回数组中每个元素的平方根

**np.sin(arr) —** 将返回数组中每个元素的正弦值

**np.log(arr) —** 将返回数组中每个元素的自然对数

**np.abs(arr) —** 将返回数组中每个元素的绝对值

**如果数组具有相同的元素和形状，np.array_equal(arr1，arr2) —** 将返回 **True** 。

这不仅仅限于此，还有无数的内置方法和函数，您可以根据数据集的需求和问题来探索、实现和了解它们以及它们的具体用途。这就是 numpy 入门的全部内容。

**牛逼！你应该感到骄傲！！**

> **感谢您的阅读！！**这是 NumPy 库的概述，它的基本操作和实现。希望你以后能多了解和探讨这个话题。如果你喜欢这个博客，你可以喜欢和评论，或者如果你对这个话题或文章有任何问题或疑问，你可以评论或让我知道我的电子邮件*。*