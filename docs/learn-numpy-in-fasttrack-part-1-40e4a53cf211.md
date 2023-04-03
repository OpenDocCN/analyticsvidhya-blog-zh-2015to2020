# 在快速通道中学习 Numpy 第 1 部分

> 原文：<https://medium.com/analytics-vidhya/learn-numpy-in-fasttrack-part-1-40e4a53cf211?source=collection_archive---------11----------------------->

# Numpy 数组对象

# NumPy 数组

**python 对象:**

1.  高级数字对象:整数、浮点
2.  容器:列表(免费插入和追加)，字典(快速查找)

**Numpy 提供:**

1.  针对多维数组的 Python 扩展包
2.  更接近硬件(效率)
3.  为科学计算而设计(方便)
4.  也称为面向阵列的计算

[1]:

```
**import** numpy **as** npa **=** np.array([0, 1, 2, 3])print(a)​print(np.arange(10))[0 1 2 3]
[0 1 2 3 4 5 6 7 8 9]
```

**为什么有用:**提供快速数字运算的高效内存容器。

[2]:

```
*#python lists*L **=** range(1000)**%**timeit [i******2 **for** i **in** L]2.64 ms ± 747 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

[3]:

```
a **=** np.arange(1000)**%**timeit a******28.74 µs ± 1.22 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
```

# 1.创建数组

** 1.1.阵列的手动构造**

[4]:

```
*#1-D*​a **=** np.array([0, 1, 2, 3])​a
```

[4]:

```
array([0, 1, 2, 3])
```

[5]:

```
*#print dimensions*​a.ndim
```

[5]:

```
1
```

[6]:

```
*#shape*​a.shape
```

[6]:

```
(4,)
```

[7]:

```
len(a)
```

[7]:

```
4
```

[8]:

```
*# 2-D, 3-D....*​b **=** np.array([[0, 1, 2], [3, 4, 5]])​b
```

[8]:

```
array([[0, 1, 2],
       [3, 4, 5]])
```

[9]:

```
b.ndim
```

[9]:

```
2
```

[10]:

```
b.shape
```

[10]:

```
(2, 3)
```

[11]:

```
len(b) *#returns the size of the first dimention*
```

[11]:

```
2
```

[12]:

```
c **=** np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])​c
```

[12]:

```
array([[[0, 1],
        [2, 3]], [[4, 5],
        [6, 7]]])
```

[13]:

```
c.ndim
```

[13]:

```
3
```

[14]:

```
c.shape
```

[14]:

```
(2, 2, 2)
```

** 1.2 用于创建数组的函数**

[15]:

```
*#using arrange function*​*# arange is an array-valued version of the built-in Python range function*​a **=** np.arange(10) *# 0.... n-1*a
```

[15]:

```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

[16]:

```
b **=** np.arange(1, 10, 2) *#start, end (exclusive), step*​b
```

[16]:

```
array([1, 3, 5, 7, 9])
```

[17]:

```
*#using linspace*​a **=** np.linspace(0, 1, 6) *#start, end, number of points*​a
```

[17]:

```
array([0\. , 0.2, 0.4, 0.6, 0.8, 1\. ])
```

[18]:

```
*#common arrays*​a **=** np.ones((3, 3))​a
```

[18]:

```
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
```

[19]:

```
b **=** np.zeros((3, 3))​b
```

[19]:

```
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])
```

[20]:

```
c **=** np.eye(3)  *#Return a 2-D array with ones on the diagonal and zeros elsewhere.*​c
```

[20]:

```
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
```

[21]:

```
d **=** np.eye(3, 2) *#3 is number of rows, 2 is number of columns, index of diagonal start with 0*​d
```

[21]:

```
array([[1., 0.],
       [0., 1.],
       [0., 0.]])
```

[22]:

```
*#create array using diag function*​a **=** np.diag([1, 2, 3, 4]) *#construct a diagonal array.*​a
```

[22]:

```
array([[1, 0, 0, 0],
       [0, 2, 0, 0],
       [0, 0, 3, 0],
       [0, 0, 0, 4]])
```

[23]:

```
np.diag(a)   *#Extract diagonal*
```

[23]:

```
array([1, 2, 3, 4])
```

[24]:

```
*#create array using random*​*#Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).*a **=** np.random.rand(4)​a
```

[24]:

```
array([0.21441103, 0.98131389, 0.18318141, 0.90358733])
```

[25]:

```
a **=** np.random.randn(4)*#Return a sample (or samples) from the “standard normal” distribution.  ***Gausian****​a
```

[25]:

```
array([-0.7362603 ,  0.18632049, -0.60449908,  0.54301321])
```

**注:**

对于来自 N(\mu，\sigma)的随机样本，使用:

sigma * np.random.randn(…) + mu

# 2.基本数据类型

您可能已经注意到，在某些情况下，数组元素显示时带有一个**尾随点(例如 2。vs 2)** 。这是由于所使用的**数据类型**的不同:

[26]:

```
a **=** np.arange(10)​a.dtype
```

[26]:

```
dtype('int32')
```

[27]:

```
*#You can explicitly specify which data-type you want:*​a **=** np.arange(10, dtype**=**'float64')a
```

[27]:

```
array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
```

[28]:

```
*#The default data type is float for zeros and ones function*​a **=** np.zeros((3, 3))​print(a)​a.dtype[[0\. 0\. 0.]
 [0\. 0\. 0.]
 [0\. 0\. 0.]]
```

[28]:

```
dtype('float64')
```

**其他数据类型**

[29]:

```
d **=** np.array([1**+**2j, 2**+**4j])   *#Complex datatype*​print(d.dtype)complex128
```

[30]:

```
b **=** np.array([**True**, **False**, **True**, **False**])  *#Boolean datatype*​print(b.dtype)bool
```

[31]:

```
s **=** np.array(['Ram', 'Robert', 'Rahim'])​s.dtype
```

[31]:

```
dtype('<U6')
```

**每个内置数据类型都有一个唯一标识它的字符代码。**

布尔型

I '(带符号)整数

u '—无符号整数

浮点型

c’—复数浮点

m '时间增量

m '日期时间

o’(Python)对象

s '，' a '(字节)字符串

统一码

原始数据(无效)

**了解更多详情**

[https://docs . scipy . org/doc/numpy-1 . 10 . 1/user/basics . types . html](https://docs.scipy.org/doc/numpy-1.10.1/user/basics.types.html)

# 3.索引和切片

**3.1 分度**

数组的项目可以像其他 **Python 序列(如列表)**一样被访问和分配:

[32]:

```
a **=** np.arange(10)​print(a[5])  *#indices begin at 0, like other Python sequences (and C/C++)*5
```

[33]:

```
*# For multidimensional arrays, indexes are tuples of integers:*​a **=** np.diag([1, 2, 3])​print(a[2, 2])3
```

[34]:

```
a[2, 1] **=** 5 *#assigning value*​a
```

[34]:

```
array([[1, 0, 0],
       [0, 2, 0],
       [0, 5, 3]])
```

**3.2 切片**

[35]:

```
a **=** np.arange(10)​a
```

[35]:

```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

[36]:

```
a[1:8:2] *# [startindex: endindex(exclusive) : step]*
```

[36]:

```
array([1, 3, 5, 7])
```

[37]:

```
*#we can also combine assignment and slicing:*​a **=** np.arange(10)a[5:] **=** 10a
```

[37]:

```
array([ 0,  1,  2,  3,  4, 10, 10, 10, 10, 10])
```

[38]:

```
b **=** np.arange(5)a[5:] **=** b[::**-**1]  *#assigning*​a
```

[38]:

```
array([0, 1, 2, 3, 4, 4, 3, 2, 1, 0])
```

## 4.副本和视图

切片操作在原始数组上创建一个视图，这只是访问数组数据的一种方式。因此，原始数组不会被复制到内存中。可以使用 **np.may_share_memory()** 来检查两个数组是否共享同一个内存块。

**修改视图时，原始数组也被修改:**

[39]:

```
a **=** np.arange(10)a
```

[39]:

```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

[40]:

```
b **=** a[::2]b
```

[40]:

```
array([0, 2, 4, 6, 8])
```

[41]:

```
np.shares_memory(a, b)
```

[41]:

```
True
```

[42]:

```
b[0] **=** 10b
```

[42]:

```
array([10,  2,  4,  6,  8])
```

[43]:

```
a  *#eventhough we modified b,  it updated 'a' because both shares same memory*
```

[43]:

```
array([10,  1,  2,  3,  4,  5,  6,  7,  8,  9])
```

[44]:

```
​​a **=** np.arange(10)​c **=** a[::2].copy()     *#force a copy*c
```

[44]:

```
array([0, 2, 4, 6, 8])
```

[45]:

```
np.shares_memory(a, c)
```

[45]:

```
False
```

[46]:

```
c[0] **=** 10​a
```

[46]:

```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

# 5.花式索引

NumPy 数组可以用切片索引，也可以用布尔或整数数组**(掩码)**。这种方法叫做**花式分度**。它创建副本而不是视图。

**使用布尔掩码**

[47]:

```
a **=** np.random.randint(0, 20, 15)a
```

[47]:

```
array([10, 11, 16, 13,  8,  7, 17, 16, 18, 16, 18, 11, 12,  5, 19])
```

[48]:

```
mask **=** (a **%** 2 **==** 0)
```

[49]:

```
extract_from_a **=** a[mask]​extract_from_a
```

[49]:

```
array([10, 16,  8, 16, 18, 16, 18, 12])
```

**使用掩码索引对于给子数组分配新值非常有用:**

[50]:

```
a[mask] **=** **-**1a
```

[50]:

```
array([-1, 11, -1, 13, -1,  7, 17, -1, -1, -1, -1, 11, -1,  5, 19])
```

**用整数数组索引**

[51]:

```
a **=** np.arange(0, 100, 10)​a
```

[51]:

```
array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
```

[52]:

```
*#Indexing can be done with an array of integers, where the same index is repeated several time:*​a[[2, 3, 2, 4, 2]]
```

[52]:

```
array([20, 30, 20, 40, 20])
```

[53]:

```
*# New values can be assigned*​a[[9, 7]] **=** **-**200​a
```

[53]:

```
array([   0,   10,   20,   30,   40,   50,   60, -200,   80, -200])
```