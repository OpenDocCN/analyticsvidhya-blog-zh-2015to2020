# 火花 SQL 和数据帧

> 原文：<https://medium.com/analytics-vidhya/spark-sql-and-dataframes-72e58fe90f94?source=collection_archive---------31----------------------->

在第 1 章 [*Apache Spark 初级读本*](/analytics-vidhya/apache-spark-primer-ca1a6d060fc8) 中，我们已经讲述了 Spark 架构和组件的基础知识。

Spark SQL、数据帧和数据集是结构化数据处理的 Spark 组件。数据帧和数据集在概念上与 RDBMS 中的表相同。数据集是“类型化”的，即在编译时检查其数据类型，因此可用于基于 JVM 的语言，如 Java 和 Scala，但不能用于 Python 或 r。数据帧是“非类型化”的，即在运行时检查其数据类型。使用 Spark SQL，您可以创建、查询、删除表、数据库和视图。我们将通过下面的 PySpark 代码片段探索数据帧和 Spark SQL。

```
*### Import required packages* from pyspark.sql import SparkSession,context
from pyspark.conf import SparkConf
from pyspark.sql.functions import broadcast*### Set configurations* conf = SparkConf().setAll([('spark.sql.shuffle.partitions', '10'), ('spark.app.name', '"SourSparkApp"'), ('spark.executor.cores', '2'), ('spark.cores.max', '2'), ('spark.driver.memory','2g')])*### Create SparkSession instance* spark = SparkSession.builder.config(conf=conf).getOrCreate()*### Read data from a Database* df=spark.read.format('jdbc').options(url="jdbc:oracle:thin:USER_NAME/PASSWORD@database_connection:1521/db_name",dbtable="my_db_table",driver="oracle.jdbc.driver.OracleDriver").option("numPartitions", 10).load()### Read data from a text file on local FS on edge node
df = spark.read.text("file:///home/path/to/my_data.txt")### Or Read data from a Json
df = spark.read.load("/hdfs/path/to/my_data.json", format="json")### Or Read data from a CSV/Delimited file
df = spark.read.load("/hdfs/path/to/my_data.csv", format="csv", sep=";", inferSchema="true", header="true")### Or Read ORC file
df = spark.read.orc("/hdfs/path/to/my_data.orc")### Or Read from Hive 
df = spark.sql("select * from databasename.tableName")### Or Read Parquet file
df = spark.read.load("/hdfs/path/to/my_data.parquet")### Take a look at Schema(columns and types) of the dataframe
df.printSchema()*### Apply filter i.e. Narrow transformation to create new Dataframe* df_filter = df.filter(df.APPLICATION=='My_app')### Convert Spark DF to Pandas DF (This should be performed on small dataset as it will collect the data on the driver)
pandas_df = df_filter.toPandas()*### Apply groupBy i.e. Wide transformation to create a new Dataframe* df_group=df_filter.groupBy("CURRENCY","TYPE").agg({"AMOUNT":"sum"}).withColumnRenamed("sum(AMOUNT)","sum_AMOUNT")### To improve the performance if df_group is used multiple times later, cache it in memory
df_group.cache()*### Create a temporary view to execute SQL statements on it* df.createOrReplaceTempView("TEMP_DATA_TABLE")
*### Create a Global view to share data across different sessions* df.createGlobalTempView("GLOBAL_TEMP_DATA_TABLE")*### Create a Dataframe using a SparkSQL query* spark_table = spark.sql("select CURRENCY,TYPE,sum(AMOUNT)  FROM global_temp.GLOBAL_TEMP_DATA_TABLE where APPLICATION='My_app' group by CURRENCY,TYPE")### Hint to Broadcast the table when joining with another table
broadcast(spark.table("spark_table")).join(spark.table("new_table"), "key")*### Perform Action i.e. Showing the SQL query results* print(spark_table.show())*### Perform Action i.e. Write the results to HDFS as a csv* df_group.write.partitionBy("CURRENCY").save('/dev/datalake/sour_spark_v2/data_agg_v2', format='csv', mode='append')
*### Store it as Parquet file* df_group.write.parquet("df_group.parquet")
*### Store it as Hive Table* sqlContext.sql("create table mytable as select * from TEMP_DATA_TABLE");
### Store the results back to RDBMS
df_group.write.jdbc(url="jdbc:oracle:thin:USER_NAME/PASSWORD@database_connection:1521/db_name",dbtable="my_db_table",driver="oracle.jdbc.driver.OracleDriver", tablename, mode="append", properties=props)*### Stop the Spark driver* spark.stop()
```

我们将通过 spark-submit 执行这段代码。Spark 将把它转换成逻辑计划，然后转换成优化的物理 DAG 计划，最后转换成执行的 RDD 字节码。

数据帧操作与 Spark SQL 的性能是相同的，因为两者都执行相同的底层 RDD 转换计划。

如果我们从数据库中读取数据并执行一些转换。如果数据帧上有过滤器，Spark 会将过滤器推送到数据库，而不是从数据库中取出所有数据，然后通过 ***查询下推*** 应用过滤器。

除了查询下推之外，您还可以让 ***谓词下推*** ，其中您可以让分离谓词的过滤数据到达它们各自的分区。

因此，在本章中，我们介绍了结构化 Spark APIs，即数据帧、数据集和 Spark SQL。

在下一章中，我们探讨 [*火花调谐和*](/@sourabhpotnis/spark-tuning-and-debugging-fe32fded8454) 调试。