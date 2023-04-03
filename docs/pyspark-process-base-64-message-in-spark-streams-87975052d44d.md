# 火花流中的 PySpark 进程 base 64 消息

> 原文：<https://medium.com/analytics-vidhya/pyspark-process-base-64-message-in-spark-streams-87975052d44d?source=collection_archive---------17----------------------->

![](img/666cacc0d54a72e862ac6d97f8747a62.png)

# 带有 base 64 编码消息的事件中心消息。

# 用例

*   物联网中心和事件中心使用 Avro 格式，并使用 base 64 编码的消息存储消息
*   使用结构化流解析数据
*   解析 base 64 编码的 body 列

# 先决条件

*   Azure 订阅
*   创建活动中心名称空间
*   选择标准，因为模式注册表在 basic 中不可用
*   创建一个带有 1 个分区的活动中心
*   创建一个名为 sample1 的消费者组
*   创建 Azure 数据砖块工作区
*   火花 3.0
*   创建活动中心集群
*   从 Maven: com.microsoft.azure 安装事件中心库 jar:azure-event hubs-spark _ 2.12:2 . 3 . 17

# 模拟器创建数据并将其发送到事件中心

*   [https://eventhubdatagenerator.azurewebsites.net/](https://eventhubdatagenerator.azurewebsites.net/)
*   复制事件中心连接字符串
*   复制事件中心的名称并粘贴到此处
*   让 JSON 消息保持原样
*   将消息数量更改为 500
*   单击提交
*   等待几秒钟加载数据

# Azure 数据块代码

```
from azure.schemaregistry import SchemaRegistryClient 
from pyspark.sql.types import * 
from pyspark.sql.functions import * 
from pyspark.sql.functions import unbase64,base64 
from pyspark.sql.functions import * 
from pyspark.sql.types import * 
import json
```

*   设置活动中心配置
*   配置为从头开始读取流。

```
# Start from beginning of stream
startOffset = "-1"

# End at the current time. This datetime formatting creates the correct string format from a python datetime object
#endTime = dt.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

# Create the positions
startingEventPosition = {
  "offset": startOffset,  
  "seqNo": -1,            #not in use
  "enqueuedTime": None,   #not in use
  "isInclusive": True
}
```

设置连接字符串

> connectionString = " Endpoint = sb://XXXXXX . service bus . windows . net/；SharedAccessKeyName = adbaccessSharedAccessKey = xxxxxxxEntityPath=eventhubname "

设置事件中心的配置以读取消息

> conf = { }
> conf[' event hubs . connectionstring ']= sc。_ JVM . org . Apache . spark . event hubs . eventhubsutils . encrypt(connectionString)
> conf[" event hubs . consumer group "]= " sample 1 "
> conf[" event hubs . starting position "]= JSON . dumps(startingEventPosition)

现在读这条小溪

> df = spark \
> 。readStream \
> 。格式(" eventhubs") \
> 。选项(**conf) \
> 。负载()

创建一个列来转换 base 64 列

> df = df.withColumn("body "，df["body"]。cast("string "))

现在让我们解析主体内部的数据

> df1 = df . select(get _ JSON _ object(df[' body ']，" $。传感器 id”)。别名(' sensor_id ')，
> get_json_object(df['body']，" $。传感器 _ 温度”)。别名(' sensor_temp ')，
> get_json_object(df['body']，" $。传感器状态”)。别名('传感器状态')
> )

显示数据集

> 显示器(df1)

上述逻辑可以应用于基数为 64 的 ant 源。

*原载于*[*https://github.com*](https://github.com/balakreshnan/Accenture/blob/master/Streams/PysparkStreams.md)*。*