# 工业 IOT-从 Kepware 数据收集到 Azure Data Explorer

> 原文：<https://medium.com/analytics-vidhya/industrial-iot-kepware-data-collection-to-azure-data-explorer-454d7ce720c1?source=collection_archive---------28----------------------->

为工业传感器、plc、控制系统和其他基于过程的系统(如 MES 和其他车间系统)提供数据存储。该系统还应具有查询从车间各种传感器收集的时间序列数据的能力。该系统可以是他们的仪表板、OEE 和根本原因分析以及其他基于质量的用例的数据源。

提供一个单一的商店，在这里我们可以集中所有的大批量生产操作，包括过程和工业传感器数据(时间序列)。很容易获得数据并存储在基于云的存储中，但这不允许我们使用查询数据进行实时和批量使用。

![](img/4b337b770a41c09b9c129694441d4777.png)

使用的组件

*   机器人
*   生产线——制造产品的生产线
*   KepWare —物联网网关
*   Azure 物联网中心—提供从 kepware 到云的安全数据通信
*   Azure 数据浏览器—云历史存储
*   仪表板—可视化工具

现在我们可以从数据收集器或网关(行业术语)收集数据。我选择的网关是 PTC Kepware。还有其他方法，如基于 OPC UA 的工具来收集数据。

像 Kepware 这样的工具能够从运行在车间的旧系统中收集数据，这是改造用例所急需的。

在新工厂中，OPC UA 可用作工业 4.0 的工业标准来收集数据。

首先让我们安装并配置 Kepware。

还要配置要连接的设备，并订阅需要发送到云的标签。

以下是 Kepware 如何向物联网中心发送数据的示例。

```
[ 
{ "timestamp": 1586780606000, 
"values": [ 
{ "id": "Channel1.Device1.Tag1", "v": 250, "q": true, "t": 1586780606000 }, 
{ "id": "Channel1.Device1.Tag2", "v": 220, "q": true, "t": 1586780606001 }, 
{ "id": "Channel1.Device1.Tag3", "v": 150, "q": true, "t": 1586780606002 } 
] 
}, 
{ "timestamp": 1586780606030, 
"values": [ 
{ "id": "Channel1.Device1.Tag1", "v": 255, "q": true, "t": 1586780606031 }, 
{ "id": "Channel1.Device1.Tag2", "v": 223, "q": true, "t": 1586780606032 }, 
{ "id": "Channel1.Device1.Tag3", "v": 156, "q": true, "t": 1586780606033 } 
] 
}, 
{ "timestamp": 1586780606040, 
"values": [ 
{ "id": "Channel1.Device1.Tag1", "v": 251, "q": true, "t": 1586780606041 }, 
{ "id": "Channel1.Device1.Tag2", "v": 229, "q": true, "t": 1586780606041 }, 
{ "id": "Channel1.Device1.Tag3", "v": 153, "q": true, "t": 1586780606042 } 
] 
}, 
{ "timestamp": 1586780606060, 
"values": [ 
{ "id": "Channel1.Device1.Tag1", "v": 252, "q": true, "t": 1586780606061 }, 
{ "id": "Channel1.Device1.Tag2", "v": 224, "q": true, "t": 1586780606062 }, 
{ "id": "Channel1.Device1.Tag3", "v": 158, "q": true, "t": 1586780606063 } 
] 
} 
]
```

现在，通过物联网中心获得流量，然后创建一个消费者群体来支持 Azure Data Explorer。

一旦进入数据浏览器，就应该创建数据模型来推送数据。

Azure Data explorer 允许我们通过编写 Kusto 查询来绘制图表和趋势。没有点击即走的环境。

但是 Azure data explorer 允许我们从其他业务系统获取数据，然后将这些数据与时间序列数据无缝地结合起来，以实现查询功能。因此，我们可以从资产管理系统或 ERP 中获取订单数据、资产管理数据，并将它们连接起来并构建查询。

Azure data explorer 还允许我们使用第三方工具或自定义网站来扩展和创建自己的仪表板。

Azure Data explorer 还内置了许多时间序列功能，用于时间序列分析，如预测、异常检测，甚至机器学习，如自动聚类和购物篮分析。

要启动 Azure data explorer，首先创建存储数据的架构

```
// Create table command //////////////////////////////////////////////////////////// 
.create table ['kepwaresample_stage'] (['values']:dynamic, ['timestamp']:datetime) // Set 0d retention on stage table so that the data is deleted after its transformed 
.alter-merge table kepwaresample_stage policy retention softdelete = 0d // Create mapping command //////////////////////////////////////////////////////////// .create-or-alter table ['kepwaresample_stage'] ingestion json mapping 'kepwaresample_stage_mapping' '[{"column":"values","path":"$.values","datatype":"dynamic"},{"column":"timestamp","path":"$.timestamp","transform":"DateTimeFromUnixMilliseconds"}]' //create function to extract the data from JSON .create-or-alter function TransformKepWareLogs() { kepwaresample_stage | mv-expand values | project msg_timestamp=timestamp, metric_timestamp=unixtime_milliseconds_todatetime(tolong(values.t)), metric_id=tostring(values.id), 
metric_value=tostring(values.v), 
metric_quality=tobool(values.q) 
} //create the final table that will hold the extracted data .create table kepwaresample (msg_timestamp: datetime, metric_timestamp: datetime, metric_id: string, metric_value: string, metric_quality: bool) //Create update policy to bind the stabing table, function, and the destination table .alter table kepwaresample policy update @'[{"IsEnabled": true, "Source": "kepwaresample_stage", "Query": "TransformKepWareLogs()", "IsTransactional": true, "PropagateIngestionProperties": true}]' // Ingest data into table command /////////////////////////////////////////////////////////// .ingest async into table ['kepwaresample_stage'] ('https://kkgkstrldkustodemo00.blob.core.windows.net/pbp-20200413-temp-e5c334ee145d4b43a3a2d3a96fbac1df/1586805347662_kepwaresample.json?sv=2018-03-28&sr=c&sig=uvob%2BuNmKN1FeDFo983Ldft0Z%2BNputQhYQYad9nZWbE%3D&st=2020-04-13T18%3A15%3A47Z&se=2020-04-17T19%3A15%3A47Z&sp=rwdl') with (format='multijson',ingestionMappingReference='kepwaresample_stage_mapping',ingestionMappingType='Json',tags="['229fee5c-508d-4f26-99ae-3f2d007c813f']") //from the above run get the id and substitute below here .show operations 2d8b2cbc-2bf1-496b-99f0-75ed6fb1ee8f kepwaresample | limit 100
```

既然数据已经准备好用于实时和其他批处理用例。

等待几分钟，看看数据是否在流动。如果数据是流动的，那么我们都是。

为了存储长期数据，我们可以连续导出到 blob 或 ADLS gen2 存储，以便长期访问。

*   创建存储帐户 ADLS Gen2
*   创建一个名为 opcoutput 的容器
*   获取 Azure data explorer 配置的存储帐户名、密钥以及容器名
*   配置 Azure 数据浏览器

如果您已经创建了表，这里是 drop table 命令

```
.drop external table opcExternalLongTerm
```

现在创建外部表来长期存储

```
.create external table opcExternalLongTerm (msg_timestamp: datetime, metric_timestamp: datetime, metric_id: string, metric_value: string, metric_quality: bool) kind=blob partition by "metric_id="metric_id, bin(TimeStamp, 1d) dataformat=csv ( h@'https://xxxxxx.blob.core.windows.net/opcoutput;xxxxxxxxxxxxxxxxxxxxxxxxx' )
```

现在在 Azure Data Explorer 中配置连续导出数据

```
.create-or-alter continuous-export opccontinousexport over (kepwaresample) to table opcExternalLongTerm with (intervalBetweenRuns=1h, forcedLatency=10m, sizeLimit=104857600) <| kepwaresample
```

去看出口

```
.show continuous-export opccontinousexport exported-artifacts | where Timestamp > ago(1h)
```

要查找故障，请使用以下命令

```
.show continuous-export opccontinousexport failures
```

要启用和禁用，请使用下面的

```
.enable continuous-export opccontinousexport .disable continuous-export opccontinousexport
```

要查询数据，请使用:

```
external_table("opcExternalLongTerm") | take 100
```

*最初发表于*[*【https://github.com】*](https://github.com/balakreshnan/IIoT-AI/blob/master/IIoT/kepwaretoadx.md)*。*