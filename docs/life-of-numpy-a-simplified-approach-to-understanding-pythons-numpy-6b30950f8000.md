# NumPy 的生命:理解 Python 的 NumPy 的简化方法。

> 原文：<https://medium.com/analytics-vidhya/life-of-numpy-a-simplified-approach-to-understanding-pythons-numpy-6b30950f8000?source=collection_archive---------15----------------------->

![](img/84ee8e6b7ea459ff864b114f1a4f3723.png)

# 为什么是 NumPy？

在深入这个 python 库的功能之前，让我回答几个基本问题。NumPy 对于任何一个数据科学家来说是多么的不可或缺？为什么我们必须使用矩阵，即使我们有像 python 中的列表这样的内置功能。答案很简单。

*   NumPy 操作的执行速度比内置的 python 列表快 5-100 倍；
*   它还提供了许多额外的选项和操作，我们将在本文中介绍。

# 我们开始吧！

**定义** : Numpy 是 Python 编程语言的一个库，增加了对大型多维数组和矩阵的支持，以及对这些数组进行操作的大量高级数学函数。

![](img/5dd947df4a0516d61e51f3662012b46a.png)

## **1。导入 NumPy 并创建矢量:**

这里我们将导入 NumPy 库并创建一个一维数组，也称为 vector。

```
#Load Library
import numpy as np#Create a vector as a Row
vec_row = np.array([7,8,9])#Create vector as a Column
vec_column = np.array([[7],[8],[9]])
```

## 2.数组的类型:

这里我们将创建一些二维数组，默认情况下这些数组在这个库中是可用的。此外，我们将讨论其中每一项的重要性。那么，让我们放大到编码部分。

```
# Python program to demonstrate different types of array
import numpy as np# Creating an array from list with datatype double
a1 = np.array([[1.1,2.1,1.13],[5.2,4.24,2.89]] ,dtype = 'double')
print ("Array with datatype double:\n", a1)# An array from tuple
a2 = np.array(( 7, 5, 2, 6, 8))
print ("\nArray created using passed tuple:\n", a2)# A 2X3 array with all zeros
a3 = np.zeros(( 2, 3) ,dtype='int64')
print ("\nAn array initialized with all zeros:\n", a3)# A 2X3 array with all ones
a4 = np.ones(( 2, 3) ,dtype='int64')
print ("\nAn array initialized with all ones:\n", a4)# A constant value array of complex type
a5 = np.full((3, 2), 19, dtype = 'complex')
print ("\nAn array initialized with all 19s and complex dtype.", a5)# a sequence of integers from 0 to 30 with steps of 4
a6 = np.arange(10, 50, 4)
print ("\nA sequential array with stepsize 4:\n", a6)# Create a sequence of 12 values in range 2 to 5
a7 = np.linspace(2, 5, 12)
print ("\nA sequential array with 12 values between 2 and 5:\n", a7)# An array with random values
a8 = np.random.randint(5,20,size=6)
print ("\nA random array:\n", a8)
```

除了基本的一个，这些数组很少需要一些解释，如下所示:

*   np.full((x，y)，3)返回一个 x，y 维的数组，所有元素为 3；
*   np.arange()和 np.linspace()的区别在于，前者你可以自己指定步长，而后者会根据要求的元素数量创建一个序列；
*   np.random.randint(low，high，size)返回一个数组，其整数元素~大小在从低到高的范围内(包括低但不包括高)。

## 3.描述数组:

在本节中，我们将讨论描述数组及其元素的所有函数。在分析大型数据集时，这些工具变得非常方便，我们必须在将数组元素输入任何算法之前确保其格式正确。

```
# Describing an array
# Creating a 2-D array which is also known as matrix
b = np.array( [[ 1, 2, 1], [ 4, 2, 1]] )#Printing type of matrix
print("Matrix type: ", type(b))#Printing Matrix dimensions (axes)
print("No. of dimensions: ", b.ndim)#Printing shape of matrix
print("Shape of matrix: ", b.shape)#Printing size (total number of elements) of matrix
print("Size of matrix: ", b.size)#Printing type of elements in matrix
print("matrix stores elements of type: ", b.dtype)
```

## 4.阵列上的操作:

在最后一节中，我们将介绍可以在 NumPy 阵列上执行的所有基本和高级操作。

**a)基本操作**:这将涵盖所有数组的基本操作，如两个数组的加法和减法，标量和矩阵乘法。

```
#Creating a 2X3 and 3X2 array
c=np.array([[1,2,3],[3,2,1]])
d=np.array([[3,4],[5,7],[9,6]])
e=np.array([[3,2,1],[1,2,3]])#Addition and subtraction of two matrices
c1=e-c
c2=e+c
print("Sum of two matrices:\n",c2)
print("\nDifference of two matrices:\n",c1)#Scalar multiplication
c3=5*c
print("\nElements of matrix c multiplied by 5:\n",c3)#Matrix multiplication
c4=np.dot(c,d)
print("\nProduct of two matrices:\n",c4)#Matrix element wise multiplication
c5=np.multiply(c,e)
print("\nElement wise multiplication of two matrices:\n",c4)# Reshaping 2X3 array to 3X2 array
c1=c.reshape(3,2)
print ("\nOriginal array:\n", c)
print ("Reshaped array:\n", c1)# Flatten array
d1 = d.flatten()
print ("\nOriginal array:\n", d)
print ("Fattened array:\n", d)
```

**注意:** np.multiply()将一个数组的元素与另一个数组的元素相乘，而 np.dot()返回两个矩阵的点积(理论上也称为矩阵乘法)。此外，对于加法、减法和乘法，两个阵列的维数需要相同，而对于点积，必要条件是第一矩阵的列数必须等于第二矩阵的行数，并且所得矩阵的维数将是(第一矩阵的行数，第二矩阵的列数)

**b)高级操作**:在这里我们将利用这个库的一些高级功能。

```
#Creating a square matrix
d=np.array([[5,3,3],[3,5,2],[7,5,3]])#Transpose the matrix
print("\nOriginal matrix:\n",d)
print("\nTranspose of matrix:\n",d.T)#Calculate the Determinant
print("\nDeterminant of matrix:\n",np.linalg.det(d))#Diagonal elements
print("\nPrint the Principal diagonal:\n",d.diagonal())
print("\nPrint the diagonal one above the Principal diagonal:\n",d.diagonal(offset=1))
print("\nPrint the diagonal one below Principal diagonal:\n",d.diagonal(offset=-1))#Calculate its inverse
print("\nInverse of matrix:\n",np.linalg.inv(d))
```

**秩、迹、特征值和特征向量:**

*   **秩:**给定矩阵中线性无关的行数；
*   **迹:**对角线元素之和；
*   **特征值和特征向量:**特征向量广泛应用于机器学习库中。直观地给定由矩阵 A 表示的线性变换，特征向量是当应用该变换时，仅改变比例(而不是方向)的向量。更正式地说

> **Av=Kv**
> 
> *这里 A 是方阵，K 包含特征值，v 包含特征向量。*

```
e=np.array([[5,3,3],[3,3,2],[7,2,3]])#Calculate the Rank
print("\nRank of matrix:\n",np.linalg.matrix_rank(e))#Trace of matrix
print("\nTrace of matrix:\n",e.trace())#Calculate the Eigenvalues and Eigenvectors of that Matrix
e_values ,e_vectors=np.linalg.eig(e)
print("\nEigenvalues of matrix:\n",e_values)
print("\nEigenvectors of matrix:\n",e_vectors)
```

## 4.奖励内容:

呀！！。锦上添花。我们将用本库中给出的一些统计函数来完成这项工作。

```
#Create a Matrix
e = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("\nOriginal Matrix:\n",e)#Max
print("\nMax value: ",np.max(e))#Max in each column
print("\nMax value: ",np.max(e,axis=1))
#axis=0 for max in each row#Mean
print("\nMean: ",np.mean(e))#Standard Dev.
print("\nStandard Deviation: ",np.std(e))#Variance
print("\nVariance: ",np.var(e))
```

# 结论

NumPy 是任何有抱负的数据科学家必须学习的不可或缺的工具之一，因为它在所有高级算法中的重要性和适用性。本文内容全面，但并不详尽，因为它涵盖了 NumPy 库最常用的功能。