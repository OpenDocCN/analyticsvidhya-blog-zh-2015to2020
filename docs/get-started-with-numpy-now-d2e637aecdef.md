# 立即开始使用 NumPy！

> 原文：<https://medium.com/analytics-vidhya/get-started-with-numpy-now-d2e637aecdef?source=collection_archive---------15----------------------->

![](img/7b9910b082e234d668f4c84783667302.png)

简·劳格森在 [Unsplash](https://unsplash.com/search/photos/numbers?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

NumPy 是在大数据集中进行数值计算的一个非常强大的工具。这是一个 python 库，它有许多用于执行快速矢量化数组操作的内置方法。NumPy 可以用来在整个数组上执行复杂的计算，而不需要 Python for 循环，并且它在内存使用方面比纯 Python 更有效。在这篇文章中，我将讨论一些创建 numpy 数组的基本方法，然后是一些常见的操作，以及我们如何使用它们来提高性能。你可以在我的 github 库[这里](https://github.com/AhsanShihab/Hello-World-to-Data-Science/blob/master/Basics%20to%20get%20started%20with%20NumPy.ipynb)找到所有代码，我在一个 jupyter 笔记本上有几乎相同的博客。您可以下载它，运行代码，试验它们以获得更好的理解。

# 导入数字

如果您的计算机上安装了 numpy，那么您只需键入下面的代码行，就可以将该库导入到您的代码文件中。这里我们给它起了一个昵称“np ”,这样我们就可以用这个短名字来指代它，而不是每次使用时都要输入完整的长名字。

```
import numpy as np
```

我使用包含 numpy 库的 Anaconda 发行版安装 python。如果您没有安装它，只需在命令提示符下运行以下命令来安装它。

```
pip install numpy
```

![](img/ec9ba46f75928c6f3fc6aba2f9ee944c.png)

安德斯·吉尔登在 [Unsplash](https://unsplash.com/s/photos/fast?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

# 有多快？

在进入细节之前，我们先通过一点点测试说服自己，numpy 其实是一个非常有效的大数据数值计算的库。我们将以两种方式执行相同的任务，一种是使用 numpy 数组，另一种是借助 loop 的纯 python 计数器部分。我们将比较两个操作完成所需的计算时间。

我可以使用 jupyter 笔记本的神奇功能%time 来测量每次操作所需的时间，你可以在 github 的我的笔记本中找到。但是在其他 ide 中，神奇的函数就不起作用了。所以，这里有一个替代的方法。

```
import time
```

现在我们将创建一个数组和一个 python 等价列表。

```
array = np.arange(10000000)
list_1 = [x for x in range(10000000)]
```

现在我们将做一个简单的操作，平方所有的元素，从 1 到 10000000，首先以矢量化的方式，然后使用循环。我们将测量两个操作的时间。

```
tick = time.time()
array = array ** 2
tock = time.time()
print(tock - tick)
```

> 输出:
> 0.000036868686

在我的例子中，它花了 0.0449 秒(根据硬件的不同会有所不同)。现在让我们检查循环操作。

```
tick = time.time()
list_2 = [x ** 2 for x in list_1]
tock = time.time()
print(tock - tick)
```

> 输出:
> 6.000636368667

我这种情况，用了 6.0544s！几乎慢了 130 倍！因此，与循环操作相比，矢量化操作的速度有多快是显而易见的。既然您已经确信 numpy 实际上非常高效，那么让我们熟悉一下它。

# 创建 NumPy 数组

NumPy 数组可以是一维或多维的。让我们首先看看如何从 python 列表中创建 numpy 数组。我们可以通过 numpy 内置的 array()函数轻松做到这一点。

```
# Creating one dimensional NP array
list_1 = [1,2,3]
array_1 = np.array(list_1)
print(array_1)# Creating multi dimensional NP array
list_2 = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
array_2 = np.array(list_2)
print(array_2)
```

> 输出:
> 
> 数组([1，2，3])
> 
> 数组([[ 1，2，3]，
> ，
> ，【7，8，9】，
> ，【10，11，12】))

这个 array()函数接受任何类似序列的对象(包括其他数组)。在从包含其他列表的列表创建多维数组的情况下，这些列表的长度需要彼此相等。否则只会创建一个数据类型为“object”的一维数组。为了更好地理解，你可以自己尝试一下。

如果我们想通过自动生成序列中的数据来创建数组，我们可以通过 NP 的“arange()”函数来实现。

```
# arange([start,optional] stop[, step, optional], dtype=None)array_3 = np.arange(5) # One dimensional
print("array 3: \n{0}".format(array_3)) # Printing the arrayarray_4 = np.arange(2,5) # One dimensional with specified start
print("\narray 4: \n{0}".format(array_4)) # Printing the arrayarray_5 = np.arange(2,10,2) # One dimensional with specified start and steps, starts at 2, ends at 10, steps 2
print("\narray 5: \n{0}".format(array_5)) # Printing the array
```

> 输出:
> 
> 数组 3:
> [0 1 2 3 4]
> 
> 阵列 4:
> 【2 3 4】
> 
> 数组 5:
> 【2 4 6 8】

numpy 中有两个内置函数，用于创建只有 1 或 0 的数组。它们在某些情况下可能会有用。所以有内置函数给它就好了。

```
some_array_6 = np.zeros((2,3)) # Dimension is set to be 2x3
some_array_7 = np.ones((2,3))
```

默认情况下，这些值将是浮点值。如果需要整数值，也可以指定。

```
array_6 = np.zeros((2,3),’int’)
array_7 = np.ones((2,3),'int')
```

我们还有一个创建单位矩阵的函数(矩阵对角线上只有 1，其他位置只有 0)。

```
identity = np.identity(3)
identity
```

> 输出:
> 数组([[1。, 0., 0.]，
> 【0。, 1., 0.]，
> 【0。, 0., 1.]])

我们可以使用 numpy 的“random.randn()”(用于获取浮点数)或“random.randint()”(用于获取整数)函数用随机生成的数据创建数组。

```
# Float data type, 2x3 dimension
array_8 = np.random.randn(2,3)# Int data type, 3x3
array_9 = np.random.randint(low = 1, high = 10, size = (3,3)) 
```

请记住，数组的数据类型不仅限于浮点数或整数。它们可以是复数、布尔对象、字符串和其他 python 对象。如果数据类型是可转换的，我们可以很容易地转换它们。请参见下面的示例。

```
# Creating an array with strings as data type
string_array = np.array(['1','2','2.5','3.5'])# Now converting its data type to float numbers 
numeric_array = string_array.astype('float')
```

# 属性

现在我们知道了如何创建 NP 数组，让我们看看如何找到数组的不同属性。“形状”是维度的大小，即行数和列数。Dimension 是数组的轴数。“大小”表示数组包含的元素数量。要获得数组的形状，我们可以使用“shape”属性。为了找到尺寸，我们可以使用“ndim”。为了找到数据类型，我们使用“dtype”属性。为了找到数组中元素的总数，即它的大小，我们使用“size”属性。

```
print(array_2.shape)
print(array_2.ndim)
print(array_2.dtype)
print(array_2.size)
```

> 输出:
> (4，3)
> 2
> int32
> 12

您可以重塑数组。为此，您可以使用内置的调整大小功能。

```
array_2.resize(2,6)
array_2.shape
```

> 输出:
> (2，6)

# 索引和切片

## 1D 阵列

一维数组的索引和切片很简单。这就像在 python 列表中索引元素一样。

```
# Creating a new array first for demonstration
base_array = np.arange(10)
base_array
```

> 输出:
> 数组([0，1，2，3，4，5，6，7，8，9])

```
base_array[5] 
# Indexing in python starts at 0, so index 5 is the 6th element
```

> 输出:
> 5

现在，如果需要选择数组的一部分，可以像 python 列表一样使用切片。

```
sliced_array = base_array[2:6] # The upper bound is excluded
sliced_array
```

> 输出:
> 数组([2，3，4，5])

这个切片数组只是一个视图，不是新的副本。所以如果你改变了什么，它会反映在主数组上。

```
sliced_array[0] = 20
sliced_array[1] = 30
sliced_array
```

> 输出:
> 数组([20，30，4，5])

```
base_array
```

> 输出:
> 数组([ 0，1，20，30，4，5，6，7，8，9])

要创建一个副本，以便切片数组中的更改不会更改主数组，请使用 *copy()* 函数。

```
base_array = np.arange(5)
copied_array = base_array.copy()
```

## 多维数组

多维数组的索引有选项。你可以通过递归调用得到一个单独的元素，就像你在列表列表中所做的那样，或者你可以只传递一个逗号分隔的索引列表。

```
# Creating a 2D array first
base_array = np.array([[1,2,3],[4,5,6],[7,8,9]])
base_array
```

> 输出:
> 数组([[1，2，3]，
> ，【4，5，6】，
> ，【7，8，9]])

```
# Getting the first row
base_array[0]
```

> 输出:
> 数组([1，2，3])

```
# Getting the element of the first column of the first row
base_array[0][0]
```

> 输出:
> 1

```
# or we can pass the indices separated by comma
base_array[0,0]
```

> 输出:
> 1

您可能已经猜到了如何分割多维数组。

```
base_array[:2] # selecting every row upto the 2nd row
```

> 输出:
> 数组([[1，2，3]，
> [4，5，6]])

```
base_array[1:] #selecting every row after the first one
```

> 输出:
> 数组([[4，5，6]，
> ，[7，8，9]])

```
base_array[:,:2]
#selecting every column upto the 2nd one, had to put an empty ':' to indicate 'take all rows'
```

> 输出:
> 数组([[1，2]，
> ，【4，5】，
> ，【7，8]])

你会在我的 github repo 中提供的 jupyter 笔记本中找到更多的例子。

## 布尔索引

还有另一种非常有趣的索引类型，叫做布尔索引。我们可以传递一个布尔值数组作为索引，它将只给出布尔值为' True '的行。让我们看看下面的例子。

```
# Creating a boolean array first
boolean_array = np.array([True, False, True])# Using boolean indexing
base_array[boolean_array]
```

> 输出:
> 数组([[1，2，3]，
> ，[7，8，9]])

布尔索引非常有用，可以用非常有创造性的方式使用。让我们看一个简单的例子。

假设我们有四种不同的东西要混合(糖、盐、香料、水或其他)。我们尝试了 10 种不同的组合。有些被证明是好的组合，有些被证明是坏的组合。我们将组合存储在一个数组中。每一列代表成分，每一行是不同的混合。

```
combinations = np.random.randint(low = 1, high = 10, size = (10,4)) # Randomly creating an array for demonstrationcombinations
#output will change every time the code is run, as they are random data
```

> 输出:
> 数组([[6，6，8，3]，
> ，【9，8，2，7】，
> ，【3，5，4，2】，
> ，
> ，【9，2，6，4】，
> ，【7，7，7，8】，
> ，【7，9，3，7】，
> ，【4，7，5，6】，
> ，【8，6

```
results = np.array(['g','g','vg','g','b','b','b','vg','b','b']) 
# Randomly created the result, assume g = good, vg = very good, b = bad
```

因此，我们希望找到非常好或仅仅好(在结果中用‘g’表示)的组合，或者除了坏组合之外的任何组合。下面是我们如何使用布尔索引来做到这一点。

```
very_good = results == 'vg'
combinations[very_good]
```

> 输出:
> 数组([[3，5，4，2]，
> [4，7，5，6]])

```
condition = results=='b'
combinations[~condition] 
# using ~ before a condination will yield every other results except the given condition
```

> 输出:
> 数组([[6，6，8，3]，
> ，【9，8，2，7】，
> ，【3，5，4，2】，
> ，【6，7，6，2】，
> ，【4，7，5，6】))

在布尔索引中，我们必须确保布尔索引的长度与其索引轴的长度相匹配。

# 操作

## 简单的算术运算

两个数组的简单加法或减法将执行逐元素的加法或减法。您也可以用标准方式将每个元素乘以或除以一个标量值。

```
print(base_array)
```

> 输出:
> [[1 2 3]
> 【4 5 6】
> 【7 8 9】]

```
base_array + base_array
```

> 输出:
> 数组([[ 2，4，6]，
> ，【8，10，12】，
> ，【14，16，18])

```
base_array * 10
```

> 输出:
> 数组([[10，20，30]，
> ，【40，50，60】，
> ，【70，80，90]])

```
base_array + 10
```

> 输出:
> 数组([[11，12，13]，
> ，【14，15，16】，
> ，【17，18，19]])

## 数组乘法

数组相乘有两种方法。一个是两个数组之间简单的元素乘法。简单地将两个数组加上乘法符号，就像在普通乘法运算中一样，就可以实现这一点。

```
base_array * base_array
```

> 输出:
> 数组([[ 1，4，9]，
> ，【16，25，36】，
> ，【49，64，81]])

但是如果要进行矩阵乘法，可以使用 numpy 的“*dot()”*函数。(对于不熟悉矩阵乘法的人来说，有点复杂。查看此[链接](https://www.mathsisfun.com/algebra/matrix-multiplying.html)了解矩阵乘法在数学上是如何完成的。)

```
base_array.dot(base_array)
# np.dot(array, array), alternate syntax
```

> 输出:
> 数组([[ 30，36，42]，
> ，【66，81，96】，
> ，【102，126，150]])

## 公制转置

我们可以通过使用*”来转置一个数组(交换它的行和列)。*师后阵。

```
base_array.T
```

> 输出:
> 数组([[1，4，7]，
> ，【2，5，8】，
> ，【3，6，9]])

对于更复杂的多维数组，我们可以使用“transpose()”函数，并按照我们想要的顺序传递轴元组作为参数。例如，array.transpose((1，2，0))将使第二个轴优先，第三个轴其次，第一个轴最后。(记住 python 索引，第一个轴的索引是 0，第二个是 1，第三个是 2。)

```
array = np.arange(16).reshape((2, 2, 4))
array
```

> 输出:
> 数组([[[ 0，1，2，3]，
> [ 4，5，6，7]]，
> 
> [[ 8，9，10，11]，T29，[12，13，14，15]])

```
array.transpose((1, 2, 0))
```

> 输出:
> 数组([[[ 0，8]，
> ，【1，9】，
> ，【2，10】，
> ，【3，11])，
> 
> [[ 4，12]，
> ，【5，13】，
> ，【6，14】，
> ，【7，15]]])

# 应用函数

有一些通用函数可以帮助我们执行快速的元素操作。例如，np.sqrt()对数组的每个元素执行平方根运算，np.exp()函数返回一个数组，其中包含每个元素的指数值。

```
np.sqrt(base_array)
```

> 输出:
> 数组([[1。，1.41421356，1.73205081]，
> 【2。，2.23606798，2.44948974]，
> 【2.64575131，2.82842712，3。]])

```
np.exp(base_array)
```

> 输出:
> 数组([[2.71828183e+00，7.38905610e+00，2.00855369e+01]，
> ，【5.45981500e+01，1.48413159e+02，4.03428793e+02]，
> ，【1.0966316 e+03，2.988

在 numpy 中有很多这样的函数，帮助执行不同的数学计算，如求对数、平方、正弦、余弦、正切等。以一种快速的方式。在我的笔记本中，我提到了一些你可能经常会发现有用的功能。

有一些数学函数可以计算数组的统计值。例如，np.mean()返回给定数组的平均值，max()给出最大值，std()给出数据的标准差。

```
array.mean()
# np.mean(array), alternate syntax
```

> 输出:
> 5.0

# 结论

我希望这篇文章能让你对 numpy 有一个基本的了解，并为你开始做好准备。我建议你从我在这本笔记本中提供的基础知识开始使用 numpy。在初始阶段，当你需要做一些事情的时候，你可能总是首先想到在你的数组中应用一个 for 循环，至少我是这么做的。但是请记住，如果你这样做，你就失去了使用矢量化运算的优势。但是不用担心。如果你坚持练习，你会养成使用矢量化运算的习惯。当你遇到一个问题，或者你需要执行一个没有在这个笔记本上列出的操作时，只需简单地进行谷歌搜索或者在 stackoverflow 中搜索。numpy 有一个很好的机会来执行你的操作，而不需要 for 循环，你知道，这将是一个更好的性能选择。