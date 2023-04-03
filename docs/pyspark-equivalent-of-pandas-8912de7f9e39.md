# Pyspark 相当于熊猫

> 原文：<https://medium.com/analytics-vidhya/pyspark-equivalent-of-pandas-8912de7f9e39?source=collection_archive---------5----------------------->

作为 Pandas 的狂热用户和 Pyspark 的初学者(我现在仍然是),我总是在寻找一篇关于 Pyspark 中 Pandas 的等价功能的文章或栈溢出帖子。我想我会为自己和任何对我有用的人创建一个。这也可能有点冗长。

![](img/f2f8c3d87784b13baadb940836b249b6.png)

**注意:**可能有一个更有效的版本，你可能需要查找，但这可以完成工作。

**1:通过引用列表将缺失的列添加到数据帧:**

假设您有一个如下所示的数据帧，pandas 中的数据帧命名为`pandas_df`，spark 中的数据帧命名为`spark_df`:

```
 ---+---+---+---+
|  A|  B|  C|  D|
+---+---+---+---+
| 24|  3| 56| 72|
|  0| 21| 19| 74|
| 41| 10| 21| 38|
| 96| 20| 44| 93|
| 39| 14| 26| 81|
+---+---+---+---+
```

现在我们有了一个要添加到 dataframe 中的列列表，默认值为 0。

`cols_to_add = ['Col1','Col2']`

在 pandas 中，我们可以使用下面的`reindex`功能:

```
print(pandas_df.reindex(columns=pandas_df.columns.union(cols_to_add,sort=False),fill_value=0)) A   B   C   D  Col1  Col2
0  24   3  56  72     0     0
1   0  21  19  74     0     0
2  41  10  21  38     0     0
3  96  20  44  93     0     0
4  39  14  26  81     0     0
```

在 Pyspark 中，我们可以使用下面的`lit`函数和`alias` 做同样的事情:

```
import pyspark.sql.functions as F
spark_df.select("*",
        *[F.lit(0).alias(i) for i in cols_to_add]).show()+---+---+---+---+----+----+
|  A|  B|  C|  D|Col1|Col2|
+---+---+---+---+----+----+
| 24|  3| 56| 72|   0|   0|
|  0| 21| 19| 74|   0|   0|
| 41| 10| 21| 38|   0|   0|
| 96| 20| 44| 93|   0|   0|
| 39| 14| 26| 81|   0|   0|
+---+---+---+---+----+----+
```

**2:索引/子集化数据帧:**

假设我们有一些索引，我们想在这些索引中包含一个数据帧的子集。

使用上面同样的数据帧，我们可以使用`.iloc[]`作为熊猫数据帧。假设起点和终点如下:

`start_row , end_row = 2,4`

```
print(df.iloc[start_row-1:end_row])

    A   B   C   D
1   0  21  19  74
2  41  10  21  38
3  96  20  44  93
```

对于 Pyspark，同样的事情可以通过分配一个`row_number()`然后使用`between`函数来实现。

```
(spark_df.withColumn("Row",F.row_number()
         .over(Window.orderBy(F.lit(0))))
         .filter(F.col("Row")
         .between(start_row,end_row)).drop("Row")).show()+---+---+---+---+
|  A|  B|  C|  D|
+---+---+---+---+
|  0| 21| 19| 74|
| 41| 10| 21| 38|
| 96| 20| 44| 93|
+---+---+---+---+
```

**3:熊猫和 Pyspark 列中值的条件赋值**

假设我们必须创建一个包含 3 个条件的条件列，其中:

如果 A 列小于 20，赋值`Less`，否则如果 A 列在 20 和 60 之间，赋值`Medium`，否则如果 A 列大于 60，赋值`More`否则赋值`God Knows`

在 pandas 中，推荐的方法是使用`numpy.select`，这是一种矢量化的方法，而不是使用速度较慢的`apply`。

```
import numpy as np
cond1 , cond2 , cond3 = df['A'].lt(20) , df['A'].between(20,60) , df['A'].gt(60)
value1 , value2 , value3 = 'Less' , 'Medium' , 'More'out = df.assign(New=np.select([cond1,cond2,cond3],[value1,value2,value3],default='God Knows'))
print(out) A   B   C   D     New
0  24   3  56  72  Medium
1   0  21  19  74    Less
2  41  10  21  38  Medium
3  96  20  44  93    More
4  39  14  26  81  Medium
```

在 Pyspark 中，我们可以使用带有`selectExpr`的 SQL `CASE`语句

```
expr = """CASE 
          WHEN A < 20 THEN 'Less' 
          WHEN A BETWEEN 20 AND 60 THEN 'Medium'
          WHEN A > 60 THEN 'More'
          ELSE 'God Knows'
          END AS New"""
spark_df.selectExpr("*",expr).show()+---+---+---+---+------+
|  A|  B|  C|  D|   New|
+---+---+---+---+------+
| 24|  3| 56| 72|Medium|
|  0| 21| 19| 74|  Less|
| 41| 10| 21| 38|Medium|
| 96| 20| 44| 93|  More|
| 39| 14| 26| 81|Medium|
+---+---+---+---+------+
```

**4:使用熊猫系列中的列表或 Pyspark 列中的数组:**

有时，您可能会得到一个列表，如下所示:

```
 version   timestamp      Arr_col
0      v1  2012-01-10  ['-A','-B']
1      v1  2012-01-11  ['D-','C-']
```

对于这类列的任何操作，例如替换子字符串等。理想的方法是使用一个理解列表，这样我们就可以在熊猫中使用以下内容:

```
output = pandas_df.assign(Arr_col=
                 [[arr.replace('-','')  for arr in i]
                 for i in df['Arr_col']]) version   timestamp Arr_col
0      v1  2012-01-10  [A, B]
1      v1  2012-01-11  [D, C]
```

在 PySpark 2.4+中，我们可以访问像`[transform](https://spark.apache.org/docs/latest/api/sql/index.html#transform)` [](https://spark.apache.org/docs/latest/api/sql/index.html#transform)这样的高阶函数，所以我们可以像这样使用它们:

```
spark_df.withColumn("Arr_col",
        F.expr("transform(Arr_col,x-> replace(x,'-',''))")).show()# or for lower versions , you can use a udf:
from pyspark.sql.types import ArrayType,StringType
def fun(x):
    return [i.replace('-','') for i in x]
myudf = F.udf(fun,ArrayType(StringType()))
spark_df.withColumn("Arr_col",myudf("Arr_col")).show()+-------+----------+-------+
|version| timestamp|Arr_col|
+-------+----------+-------+
|     v1|2012-01-10| [A, B]|
|     v1|2012-01-11| [D, C]|
+-------+----------+-------+
```

感谢阅读。希望这对你有用。我会试着在将来想出更多这样的场景。