# PySpark 中使用 DataFrame 和 SQL 的模式定义和排序

> 原文：<https://medium.com/analytics-vidhya/schema-definition-and-ranking-using-dataframe-and-sql-in-pyspark-6492c6c06668?source=collection_archive---------20----------------------->

就分析用例而言，排名是 Spark 中最重要的概念之一。这篇博客将会讨论在 Spark 中实现等级、密集等级和行数。除此之外，还将介绍如何在数据之上创建一个模式。

> rank 和 denseRank 的区别在于，当出现平局时，denseRank 不会在排名序列中留下空白。也就是说，如果你用 denseRank 对一场比赛进行排名，有三个人并列第二，你会说这三个人都是第二名，下一个人是第三名。

**问题陈述:**根据‘TX’状态客户总数排名前 25 个城市。

**先决条件:** Hortonworks 或 Cloudera VM

**执行步骤:**

1.  将数据从 MySQL 导入 HDFS
2.  火花执行

**从 MySQL 导入数据到 HDFS**

该解决方案要求从 MySQL 导入 CSV 格式的“客户”表。

> *查看我的帖子“*关于如何将数据从 MySQL 导入 HDFS 的 Sqoop、HDFS、Hive 和 Spark *的用例，以及如何将数据加载到 MySQL 的更多信息。*

```
**#Create folder in HDFS**
hdfs dfs -mkdir /user/hdfs/spark_usecase**#Customers Table: CSV Format** sqoop import --connect jdbc:mysql://localhost:3306/retail_db --username retail_dba --password hadoop --table customers --target-dir /user/cloudera/spark_usecase/customers;
```

**使用数据帧的火花执行**

所需数据现已导入 HDFS。使用 Spark，现在通过在读取时定义模式来加载数据。

```
**#Spark shell** pyspark**#Reading CSV data and defining schema** customers = spark.read.csv('/user/cloudera/spark_usecase/customers', schema='id int, fname string, lname string, email string, password string, street string, city string, state string, zipcode string')**#Importing required function(s)** from pyspark.sql.functions import count, col**#Filtering TX customers, finding # of customers per city** cus_count = customers.where(customers.state == 'TX').groupBy('state', 'city').agg(count('id').alias('cus_count')).orderBy('cus_count', ascending=False).limit(25)**#Importing required function(s)** from pyspark.sql import Window**#Creating window based on state and order by # of customers**
window = 
Window.partitionBy('state').orderBy(col('cus_count').desc())**#Importing required function(s)** from pyspark.sql.functions import rank, dense_rank, row_number**#Finding rank, dense rank and row number on data** result = cus_count.withColumn('rank', rank().over(window)).withColumn('dense_rank', dense_rank().over(window)).withColumn('row_num', row_number().over(window))**#Printing result** result.show(25)**#Writing result in JSON into HDFS** result.write.json('/user/cloudera/spark_usecase/output')
```

**使用 SQL 执行 Spark】**

通过以下方法使用 Spark SQL 可以实现相同的用例。

```
**#Spark shell** pyspark**#Reading CSV data and defining schema** customers = spark.read.csv('/user/cloudera/spark_usecase/customers', schema='id int, fname string, lname string, email string, password string, street string, city string, state string, zipcode string')**#Create a global temporary view with this DataFrame** customers.createTempView('customers')**#Create a global temporary view with this DataFrame**
result = spark.sql('select state, city, cus_count, rank() over (partition by state order by cus_count desc) as rank, dense_rank() over (partition by state order by cus_count desc) as dense_rank, row_number() over (partition by state order by cus_count desc) as row_num from (select state, city, count(*) as cus_count from customers where state = "TX" group by state, city order by cus_count desc limit 25)')**#Printing result** result.show(25)**#Writing result in JSON into HDFS** result.write.json('/user/cloudera/spark_usecase/output')
```

**结果**

“休斯顿”排名第一，拥有 91 名客户，“布朗斯维尔”和“普莱诺”排名第十，拥有相同数量的客户。对于城市“圣贝尼托”,可以观察到等级和密集等级之间的差异，在该城市中，dense rank 在进行等级划分时没有留下间隙。

注意:下面的结果是基于上面提到的 MySQL 中加载的数据。

```
+-----+--------------------+---------+----+----------+-------+
|state|                city|cus_count|rank|dense_rank|row_num|
+-----+--------------------+---------+----+----------+-------+
|   TX|             Houston|       91|   1|         1|      1|
|   TX|              Dallas|       75|   2|         2|      2|
|   TX|         San Antonio|       53|   3|         3|      3|
|   TX|             El Paso|       42|   4|         4|      4|
|   TX|          Fort Worth|       27|   5|         5|      5|
|   TX|              Austin|       25|   6|         6|      6|
|   TX|              Laredo|       23|   7|         7|      7|
|   TX|            Amarillo|       19|   8|         8|      8|
|   TX|          Sugar Land|       17|   9|         9|      9|
|   TX|         Brownsville|       16|  10|        10|     10|
|   TX|               Plano|       16|  10|        10|     11|
|   TX|              Irving|       15|  12|        11|     12|
|   TX|          San Benito|       13|  13|        12|     13|
|   TX|     College Station|       13|  13|        12|     14|
|   TX|          Carrollton|       12|  15|        13|     15|
|   TX|             Weslaco|       12|  15|        13|     16|
|   TX|North Richland Hills|       12|  15|        13|     17|
|   TX|             Mission|       11|  18|        14|     18|
|   TX|             Del Rio|       11|  18|        14|     19|
|   TX|            Richmond|       10|  20|        15|     20|
|   TX|            Mesquite|       10|  20|        15|     21|
|   TX|           Harlingen|       10|  20|        15|     22|
|   TX|                Katy|       10|  20|        15|     23|
|   TX|               Pharr|        9|  24|        16|     24|
|   TX|          San Marcos|        9|  24|        16|     25|
+-----+--------------------+---------+----+----------+-------+
```

**参考**

查看我的帖子“我是如何完成 CCA Spark 和 Hadoop Developer (CCA175)认证的”来了解我的学习细节。此外，Spark 关于方法定义的文档。