# Spark MLlib 中的数据类型

> 原文：<https://medium.com/analytics-vidhya/data-types-in-spark-mllib-966b4800f893?source=collection_archive---------4----------------------->

![](img/a6d5d148f8ace4ec1328413228132017.png)

[*图片来源*](https://www.scnsoft.com/blog/big-data-quality)

# 局部向量

MLlib 支持两种类型的局部向量:密集和稀疏。当大多数数字为零时，使用稀疏向量。要创建稀疏向量，您需要提供向量的长度——非零值的索引，它们应该是严格递增的非零值。

```
from pyspark.mllib.linalg import Vectors## Dense Vector
print(Vectors.dense([1,2,3,4,5,6,0]))
# >> DenseVector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0])### SPARSE VECTOR 
### Vectors.sparse( length, index_of_non_zero_values, non_zero_values)
### Indices values should be strictly increasingprint(Vectors.sparse(10, [0,1,2,4,5], [1.0,5.0,3.0,5.0,7]))
# >> SparseVector(10, {0: 1.0, 1: 5.0, 2: 3.0, 4: 5.0, 5: 7.0})print(Vectors.sparse(10, [0,1,2,4,5], [1.0,5.0,3.0,5.0,7]).toArray())
# >> array([1., 5., 3., 0., 5., 7., 0., 0., 0., 0.])
```

# 标记点

标记点是一个局部向量，其中一个标签被分配给每个向量。你必须解决监督的问题，你有一些目标对应的一些功能。标注点与将矢量作为一组要素和与之关联的标注提供的情况完全相同。

```
from pyspark.mllib.regression import LabeledPoint# set a Label against a Dense Vector
point_1 = LabeledPoint(1,Vectors.dense([1,2,3,4,5]))# Features 
print(point_1.features)# Label
print(point_1.label)
```

# 局部矩阵

局部矩阵存储在一台机器上。MLlib 支持密集矩阵和稀疏矩阵。在稀疏矩阵中，非零项值以列主顺序存储在压缩稀疏列(CSC)格式中。

```
# import the Matrices
from pyspark.mllib.linalg import Matrices# create a dense matrix of 3 Rows and 2 columns
matrix_1 = Matrices.dense(3, 2, [1,2,3,4,5,6])print(matrix_1)
# >> DenseMatrix(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], False)print(matrix_1.toArray())
"""
>> array([[1., 4.],
          [2., 5.],
          [3., 6.]])
"""# create a sparse matrix
matrix_2 = Matrices.sparse(3, 3, [0, 1, 2, 3], [0, 0, 2], [9, 6, 8])print(matrix_2)
# SparseMatrix(3, 3, [0, 1, 2, 3], [0, 0, 2], [9.0, 6.0, 8.0], False)print(matrix_2.toArray())
"""
>> array([[9., 6., 0.],
          [0., 0., 0.],
"""
```

# 分布式矩阵

分布式矩阵存储在一个或多个 rdd 中。选择正确的分布式矩阵格式非常重要。迄今为止，已经实现了四种类型的分布式矩阵:

*   **行矩阵**:每行是一个局部向量。您可以在多个分区上存储行。像随机森林这样的算法可以使用行矩阵来实现，因为该算法划分行以创建多个树。一棵树的结果不依赖于其他树。因此，我们可以利用分布式架构，对大数据的随机森林等算法进行并行处理

```
# Distributed Data Type - Row Matrix
from pyspark.mllib.linalg.distributed import RowMatrix# create RDD
rows = sc.parallelize([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])# create a distributed Row Matrix
row_matrix = RowMatrix(rows)print(row_matrix)
# >> <pyspark.mllib.linalg.distributed.RowMatrix at 0x7f425884d7f0>print(row_matrix.numRows())
# >> 4print(row_matrix.numCols())
# >> 3
```

*   **索引行矩阵:**它类似于行矩阵，行存储在多个分区中，但以有序的方式存储。为每一行分配一个索引值。它用于顺序很重要的算法中，如时间序列数据。它可以从索引行的 RDD 创建

```
# Indexed Row Matrixfrom pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix# create RDD
indexed_rows = sc.parallelize([
    IndexedRow(0, [0,1,2]),
    IndexedRow(1, [1,2,3]),
    IndexedRow(2, [3,4,5]),
    IndexedRow(3, [4,2,3]),
    IndexedRow(4, [2,2,5]),
    IndexedRow(5, [4,5,5])
])# create IndexedRowMatrix
indexed_rows_matrix = IndexedRowMatrix(indexed_rows)print(indexed_rows_matrix.numRows())
# >> 6print(indexed_rows_matrix.numCols())
# >> 3
```

*   **坐标矩阵:**可以从 MatrixEntry 的 RDD 创建坐标矩阵。只有当矩阵的维数都很大时，我们才使用坐标矩阵。

```
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry# Create an RDD of coordinate entries with the MatrixEntry class:
matrix_entries = sc.parallelize(
[MatrixEntry(0, 5, 2), 
MatrixEntry(1, 1, 1), 
MatrixEntry(1, 5, 4)])# Create an CoordinateMatrix from an RDD of MatrixEntries.
c_matrix = CoordinateMatrix(matrix_entries)# number of columns
print(c_matrix.numCols())
# >> 6# number of rows
print(c_matrix.numRows())
# >> 2
```

*   **分块矩阵:**在一个分块矩阵中，我们可以在不同的机器上存储一个大矩阵的不同子矩阵。我们需要指定块的尺寸。就像下面的例子，我们有 3X3，对于每个块，我们可以通过提供坐标来指定一个矩阵。

```
# import the libraries
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix# Create an RDD of sub-matrix blocks.
blocks = sc.parallelize(
[((0, 0), Matrices.dense(3, 3, [1, 2, 1, 2, 1, 2, 1, 2, 1])),
 ((1, 1), Matrices.dense(3, 3, [3, 4, 5, 3, 4, 5, 3, 4, 5])),
 ((2, 0), Matrices.dense(3, 3, [1, 1, 1, 1, 1, 1, 1, 1, 1]))])# Create a BlockMatrix from an RDD of sub-matrix blocks  of size 3X3
b_matrix = BlockMatrix(blocks, 3, 3)# columns per block
print(b_matrix.colsPerBlock)
# >> 3# rows per block
print(b_matrix.rowsPerBlock)
# >> 3# convert the block matrix to local matrix
local_mat = b_matrix.toLocalMatrix()# print local matrix
print(local_mat.toArray())
"""
>> array([[1., 2., 1., 0., 0., 0.],
          [2., 1., 2., 0., 0., 0.],
          [1., 2., 1., 0., 0., 0.],
          [0., 0., 0., 3., 3., 3.],
          [0., 0., 0., 4., 4., 4.],
          [0., 0., 0., 5., 5., 5.],
          [1., 1., 1., 0., 0., 0.],
          [1., 1., 1., 0., 0., 0.],
          [1., 1., 1., 0., 0., 0.]])
"""
```

*原载于 2019 年 10 月 28 日*[*https://www.analyticsvidhya.com*](https://www.analyticsvidhya.com/blog/2019/10/pyspark-for-beginners-first-steps-big-data-analysis/?utm_source=av&utm_medium=feed-articles&utm_campaign=feed)*。*