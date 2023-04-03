# Spark 加密/解密专栏，适用于 PII、GDPR 合规性、隐私和安全性。

> 原文：<https://medium.com/analytics-vidhya/spark-encrypt-columns-for-pii-gdpr-compliance-and-security-3bf17bf59636?source=collection_archive---------9----------------------->

![](img/62d7ce5a717a29cbedd6af117054d942.png)

# 能够加密 Spark Scala 数据帧中的列

本教程是用 Azure 数据砖构建的。本文分为两部分，第一部分是使用 SHA 散列法，第二部分是使用 AES 加密法，展示我们如何加密和解密 PII、GDPR 和其他合规性安全数据。

首先，让我们得到所有需要的进口

```
import scala.collection.JavaConverters._ 
import com.microsoft.azure.eventhubs._ 
import java.util.concurrent._ 
import scala.collection.immutable._ 
import scala.concurrent.Future 
import scala.concurrent.ExecutionContext.Implicits.global 
import org.apache.spark.sql._ 
import org.apache.spark.sql.functions._ 
import org.apache.spark.sql.types._ 
import org.apache.spark.sql.functions._ 
import org.apache.spark.sql.functions._
```

为测试配置存储帐户信息

```
spark.conf.set( "fs.azure.account.key.xxxxx.blob.core.windows.net", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
```

现在为三角洲湖设立检查站

```
val checkpointLocationproduct = "wasbs://xxxxx@xxxxxx.blob.core.windows.net/deltaidi/delta_custdata/_checkpoints/etl-from-jsonproduct" 
val deltapathproduct = "wasbs://xxxxxx@xxxxxxx.blob.core.windows.net/deltaidi/delta_productdim"
```

将数据文件读入数据帧

```
val dfproduct = spark.read.option("header","true").option("inferSchema","true").csv("wasbs://xxxxx@xxxxx.blob.core.windows.net/productdim.csv")
```

显示并确保数据集尚未加密。

```
display(dfproduct)
```

现在是导入安全库的时候了

```
import java.security.MessageDigest
```

让我们做一些改变

```
def removeAllWhitespace(col: Column): Column = { regexp_replace(col, "\\s+", "") }val df1 = dfproduct.withColumn("customercleaned", removeAllWhitespace(dfproduct("customername")))
```

显示结果

```
display(df1)import org.apache.spark.sql.types._ 
import org.apache.spark.sql.functions._ 
import javax.xml.bind.DatatypeConverter
```

创建加密函数

```
def sha256Hash(text: String) : String = String.format("%064x", new java.math.BigInteger(1, java.security.MessageDigest.getInstance("SHA-256").digest(text.getBytes("UTF-8"))))
```

构建自定义项

```
val encryptUDF = udf(sha256Hash _)
```

运行加密

```
val df2 = df1.withColumn("encryptedcust", encryptUDF(col("customername").cast(StringType)))
```

显示数据并查看它是否被加密

```
display(df2)
```

现在是时候检查 AES 了。

```
import java.security.MessageDigest 
import java.util import javax.crypto.Cipher 
import javax.crypto.spec.SecretKeySpec 
import org.apache.commons.codec.binary.Base64 
import java.security.DigestException; 
import java.security.InvalidAlgorithmParameterException; 
import java.security.InvalidKeyException; 
import java.security.NoSuchAlgorithmException; 
import java.security.SecureRandom; 
import java.util.Arrays; 
import javax.crypto.BadPaddingException; 
import javax.crypto.Cipher; 
import javax.crypto.IllegalBlockSizeException; 
import javax.crypto.NoSuchPaddingException; 
import javax.crypto.spec.IvParameterSpec; 
import javax.crypto.spec.SecretKeySpec;
```

让我们创建 AES 加密代码:

```
def encrypt(key: String, value: String): String = { 
val cipher: Cipher = Cipher.getInstance("AES/ECB/PKCS5Padding") cipher.init(Cipher.ENCRYPT_MODE, keyToSpec(key)) org.apache.commons.codec.binary.Base64.encodeBase64String(cipher.doFinal(value.getBytes("UTF-8"))) 
} def decrypt(key: String, encryptedValue: String): String = { 
val cipher: Cipher = Cipher.getInstance("AES/ECB/PKCS5PADDING") cipher.init(Cipher.DECRYPT_MODE, keyToSpec(key)) 
new String(cipher.doFinal(org.apache.commons.codec.binary.Base64.decodeBase64(encryptedValue))) 
} def keyToSpec(key: String): SecretKeySpec = { 
var keyBytes: Array[Byte] = (SALT + key).getBytes("UTF-8") 
val sha: MessageDigest = MessageDigest.getInstance("SHA-1") 
keyBytes = sha.digest(keyBytes) keyBytes = util.Arrays.copyOf(keyBytes, 16) 
new SecretKeySpec(keyBytes, "AES") 
} private val SALT: String = "jMhKlOuJnM34G6NHkqo9V010GhLAqOpF0BePojHgh1HgNg8^72k"
```

设置要测试的键值

```
val key = "123456789"import org.apache.spark.sql.functions.lit
```

创建带键的列以加快处理速度

```
val df3 = df1.withColumn("key", lit(key))
```

创建加密 udf 函数

```
val encryptUDF1 = udf(encrypt _)
```

创建解密 udf 函数

```
val decryptUDF = udf(decrypt _)
```

现在运行加密的数据框，查看列是否加密

```
val df4 = df3.withColumn("encryptedcust", encryptUDF1(col("key").cast(StringType),col("customername").cast(StringType)))
```

显示结果

```
display(df4)
```

检查列名 encryptedcust，数据应该加密

现在来解密

```
val df5 = df4.withColumn("deencryptedcust", decryptUDF(col("key").cast(StringType),col("encryptedcust").cast(StringType)))
```

显示结果

```
display(df5)
```

查找列名 deencryptedcust，应该显示常规名称而不是加密名称。

很好，祝你在保护你的专栏上玩得开心，然后给德尔塔湖回信，我们已经加密了数据。

*最初发表于*[*【https://github.com】*](https://github.com/balakreshnan/EventDrivenDL/blob/master/ScalaEncrypt.md)*。*