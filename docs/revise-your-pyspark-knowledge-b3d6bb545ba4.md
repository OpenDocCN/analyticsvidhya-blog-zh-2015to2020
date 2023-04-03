# 复习你的 PySpark 知识

> 原文：<https://medium.com/analytics-vidhya/revise-your-pyspark-knowledge-b3d6bb545ba4?source=collection_archive---------17----------------------->

![](img/93d01e0d674d4d6528c2c1e236fb1dca.png)

麦克斯韦·尼尔森在 T2 的照片

作为一名数据工程师，在一段时间后，你可能不得不在不同的编程/脚本语言之间切换。根据我的经验，我通常需要根据项目需求在纯 SQL 查询、Java、Shell 脚本、Scala 和 Python 之间切换。

这些直接的转换可能需要一些时间才能让我们上手，有时我们需要寻找直接函数是否可用于该编程语言以及语法。所以我写下了一些关键点，或者我们可以称之为 PySpark 的“作弊代码”,这可能会在你长时间休息后过渡到 PySpark 时有所帮助。

> **初始化 spark session**

```
from pyspark.sql import SparkSession 
spark = SparkSession \
        .builder \
        .appName("PySpark Example") \
        .config("spark.some.config.option", "your-value") \
        .getOrCreate()
```

> ***创建数据帧***

**来自 RDDs**

```
from pyspark.sql.types import *
```

*推断模式*

```
sc = spark.sparkContext 
lines = sc.textFile("E:\products.csv") 
parts = lines.map(lambda l: l.split(",")) 
products = parts.map(lambda p: Row(prod_id=int(p[0]), \
        product_name=p[1], price=int(p[2]))) 
products_df = spark.createDataFrame(products)
```

*指定模式*

```
schemaString = "prod_id product_name price" 
fields = [StructField(field_name, StringType(), True) \
        for field_name in schemaString.split()] 
schema = StructType(fields) 
products_df = spark.createDataFrame(products, schema)
```

**来自火花数据源**

```
#TEXT files 
df = spark.read.text("sales.txt")#JSON files
df1 = spark.read.json("customer.json") 
df2 = spark.read.load("students.json", format="json")#Parquet files 
df3 = spark.read.load("transactions.parquet")
```

> ***向数据帧添加新列***

```
df = df.withColumn('name',df.customer.name) \
        .withColumn('mailId',df.customer.mailId) \
        .withColumn('address',df.customer.address) \
        .withColumn('PhoneNumber',explode(df.contactNo.number))
```

> ***更新数据帧的列名***

```
df = df.withColumnRenamed('mailId', 'Email')
```

> ***从数据帧*** 中删除列

```
df = df.drop("Email", "address")
df = df.drop(df.Email).drop(df.address)
```

> ***检查你的数据帧***

```
df.show()             
#Display the content of dfdf.head(3)                 
#Return first n rowsdf.tail(5)                 
#Return last n rowsdf.first()                 
#Return first rowdf.take(2)                
#Return the first n rowsdf.schema                 
#Return the schema of dfdf.printSchema()        
#Print the schema of dfdf.columns              
#Return the columns of dfdf.dtypes                 
#Return df column names and data typesdf.describe().show()    
#Compute summary statisticsdf.count()              
#Count the number of rows in dfdf.distinct().count()   
#Count the number of distinct rows in df
```

> ***在数据帧上写查询***

**选择查询**

```
df.select("StudentID").show()           
#Show all records in StudentID columndf.select("StudentID",explode("contactInfo").alias("PhoneNumber")) \
        .select("PhoneNumber.type","StudentID","age").show()
#Show all records in  StudentID, age and PhoneNumber typedf.select(df["StudentID"],df["age"]+ 1).show()
#Show all records in StudentID and age with 1 added to the entries of agedf.select(df.address.substr(1,4).alias("street")).show() 
#Return substrings of  address
```

**条件查询**

```
df.select(df["age"] > 15).show()       
#Show all records where age > 15df.filter(df["age"] > 15).show()
#Show all records where age > 15df[df.StudentID.isin(1234,1235)].show()   
#Show StudentID if in the given valuesdf.select("FirstName",df.department.like("%D4%") \
        .alias("D4_dept")).show()
#Show FirstName and D4_dept as TRUE if department is like D4df.select("FirstName",df.department.startswith("Civil") \
        .alias("Civilian")).show() 
#Show FirstName and Civilian as TRUE if department starts with Civildf.select("StudentID",df.department.endswith("Engineering") \
        .alias("Engineer")).show()
#Show StudentID and Engineer as TRUE if department ends with Engineering
```

**按查询分组**

```
df.groupBy("department").count().show()
#Return count of members for each department group
```

**分类数据帧**

```
df.sort(df.age.desc()).show() 
df.sort("age", ascending=False).show() 
df.orderBy(["age","city"],ascending=[0,1]).collect()
```

> ***修改数据帧中的值***

```
df.na.fill(50).show() 
#Replace null valuesdf.na.replace(10, 20).show()
#Replace one value with another and return a new dfdf.na.drop().show()
#Drop rows with null values and return a new dfdf = df.dropDuplicates()
#Drop duplicated rows and return a new df
```

> ***修改数据帧的分区数量***

```
df.repartition(10).rdd.getNumPartitions()
#Can increase or decrease the number of partitionsdf.coalesce(1).rdd.getNumPartitions()
#Can decrease the number of partitions
```

> ***将数据帧注册为视图***

```
customer_df.createTempView("customer")
product_df.createOrReplaceTempView("product")
student_df.createGlobalTempView("student")#Now we can write spark.sql queriesspark.sql("select * from product").show()
```

> ***获取数据帧的输出***

```
new_rdd = df.rdd
#Convert df into RDDdf.toJSON().collect()
#Return df into JSON formatdf.toPandas()
#Return df as Pandas DataFramedf.select("StudentID","department").write.save("E:\f1.parquet")
#Write content of df in f1.parquet filedf.select("StudentID","department").write. \
        save("E:\f2.json", format="json")
#Write content of df in f2.json file
```

我希望所有这些命令和函数列表将有助于解决您在开发中的障碍，如果您觉得缺少什么，也可以在评论中随意添加。