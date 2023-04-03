# Azure 数据砖节点到节点通信加密

> 原文：<https://medium.com/analytics-vidhya/azure-data-bricks-node-to-node-communication-encryption-ea0701e133ab?source=collection_archive---------19----------------------->

文章讨论了如何通过加密来保护 spark 节点间的通信。许多公司都想保护 spark 及其处理过程。这可以通过 azure data bricks 引入的加密节点到节点通信的新特性来实现。以下是步骤。

*   为全局或集群范围初始化脚本创建一个初始化脚本
*   创建 azure 数据块集群。基本默认一个就够了。此群集仅用于创建初始化脚本。
*   创建一个 scala 记事本，剪切并粘贴下面的代码

```
dbutils.fs.put("/databricks/scripts/set-encryption.sh","""
#!/bin/bash
keystore_file="$DB_HOME/keys/jetty_ssl_driver_keystore.jks"
keystore_password="gb1gQqZ9ZIHS"

# Use the SHA256 of the JKS keystore file as a SASL authentication secret string
sasl_secret=$(sha256sum $keystore_file | cut -d' ' -f1)

spark_defaults_conf="$DB_HOME/spark/conf/spark-defaults.conf"
driver_conf="$DB_HOME/driver/conf/config.conf"

if [ ! -e $spark_defaults_conf ]; then
touch $spark_defaults_conf
fi

if [ ! -e $driver_conf ]; then
touch $driver_conf
fi

# Authenticate
echo "spark.authenticate true" >> $spark_defaults_conf
echo "spark.authenticate.secret $sasl_secret" >> $spark_defaults_conf

# Configure AES encryption
echo "spark.network.crypto.enabled true" >> $spark_defaults_conf
echo "spark.network.crypto.saslFallback false" >> $spark_defaults_conf

# Configure SSL
echo "spark.ssl.enabled true" >> $spark_defaults_conf
echo "spark.ssl.keyPassword $keystore_password" >> $spark_defaults_conf
echo "spark.ssl.keyStore $keystore_file" >> $spark_defaults_conf
echo "spark.ssl.keyStorePassword $keystore_password" >> $spark_defaults_conf
echo "spark.ssl.protocol TLSv1.2" >> $spark_defaults_conf
echo "spark.ssl.standalone.enabled true" >> $spark_defaults_conf
echo "spark.ssl.ui.enabled true" >> $spark_defaults_conf

head -n -1 ${DB_HOME}/driver/conf/spark-branch.conf > $driver_conf

echo " // Authenticate">> $driver_conf

echo " \"spark.authenticate\" = true" >> $driver_conf
echo " \"spark.authenticate.secret\" = \"$sasl_secret\"" >> $driver_conf

echo " // Configure AES encryption">> $driver_conf
echo " \"spark.network.crypto.enabled\" = true" >> $driver_conf
echo " \"spark.network.crypto.saslFallback\" = false" >> $driver_conf

echo " // Configure SSL">> $driver_conf

echo " \"spark.ssl.enabled\" = true" >> $driver_conf
echo " \"spark.ssl.keyPassword\" = \"$keystore_password\"" >> $driver_conf
echo " \"spark.ssl.keyStore\" = \"$keystore_file\"" >> $driver_conf
echo " \"spark.ssl.keyStorePassword\" = \"$keystore_password\"" >> $driver_conf
echo " \"spark.ssl.protocol\" = \"TLSv1.2\"" >> $driver_conf
echo " \"spark.ssl.standalone.enabled\" = true" >> $driver_conf
echo " \"spark.ssl.ui.enabled\" = true" >> $driver_conf
echo " }" >> $driver_conf

mv $driver_conf ${DB_HOME}/driver/conf/spark-branch.conf

""",true)
```

*   上面的代码将创建一个名为`set-encryption.sh.`的文件
*   shell 脚本基本上创建一个密钥，然后在每个节点上启用 AES 加密。创建 spark 集群时，脚本在所有节点上运行。
*   验证并确保我们已经创建了脚本文件

```
display(dbutils.fs.ls("dbfs:/databricks/scripts/set-encryption.sh"))
```

# 创建新群集—对节点通信应用加密

*   创建一个全新的集群
*   选择您自己的配置或保留默认值
*   在创建新集群之前，请转到高级部分
*   选择初始化脚本部分
*   指定初始化脚本路径

```
dbfs:/databricks/scripts/set-encryption.sh
```

*   现在启动集群
*   创建集群通常需要额外的几分钟时间。

# 测试集群

*   现在，让我们在集群运行后对其进行测试
*   进入工作区，创建一个 scala 记事本
*   类型

```
sc.version
```

*   让我们通过读取样本 json 文件来做一些数据帧测试

```
val df = spark.read.json("/databricks-datasets/samples/people/people.json")
display(df)
```

*   现在让我们做更多的数据帧读取

```
val diamonds = spark.read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv")
display(diamonds)
```

*   如果你能看到细胞运行，那么我们都设置好了。我还没有做上述性能测试，但很快就会到来。
*   还要记住，使用上面的方法会影响性能。请适当调整您的集群

谢谢你。请让我们知道您的意见和想法。

原文：<https://github.com/balakreshnan/Accenture/blob/master/cap/databricksnodetonodeencryption.md>