# 利用结构化流探索简单、可扩展、容错的流处理

> 原文：<https://medium.com/analytics-vidhya/easy-scalable-fault-tolerant-stream-processing-with-structured-streaming-8f340be7ed29?source=collection_archive---------24----------------------->

![](img/efb2faabd4c0410fd074794646e1686b.png)

## 流处理的复杂性:

![](img/51453503f4ca9e1da2258726d05b52c7.png)

## **什么是结构化流？**

![](img/418fab1d63e961387b63f389504d0889.png)

## **解剖流字数:**

![](img/e6c340c1ee9c8bd45fc26875a04f1953.png)![](img/2dd7650b814f18a2f9e16e7c24ad4552.png)![](img/da8615132b7d8a2fb84d8a1865fce286.png)![](img/0c5b3990a76f01b13c76bf7935bbf5c1.png)![](img/781bf3e0e40bc3cac1755cbec3352628.png)

## 利用时间:

**活动时间-**

*   许多用例需要按事件时间汇总统计数据。

例如，在 1 小时窗口内，每个系统中的错误数量是多少？

*   从数据中提取事件时间，处理延迟的、无序的数据。
*   DStream APIs 对于事件时的东西是不够的。

**事件时间聚合-**

*   窗口只是结构化流中另一种类型的分组。

每小时记录数-**parsed data . group by(window(" timestamp "，" 1 小时"))。count()**

每 10 分钟每个设备的平均信号强度-**parsed data . group by(" device "，window("timestamp "，" 10 mins "))。avg("信号")**

*   支持 UDAFs！

## **聚合的有状态处理:**

![](img/f1fb6b932314be3ac3ab173562fb47ee.png)

## **自动处理迟交数据:**

![](img/c3f259c76c6f9c70d2abb394659d257a.png)

## 水印:

![](img/308f392b12d87def4be70b0d09e8de8b.png)

## 关注点的彻底分离:

![](img/4eac67aa0ba5a9a5f08cfb2bd3db7831.png)

## 其他有趣的操作:

![](img/16b5ae3847644edf899bc65c35a32b91.png)

## 流式重复数据删除:

![](img/f7ccf7d592f5f9008ceb7513cb5e3aa8.png)

## **带水印的流式重复数据删除:**

![](img/b7099ead2fe4862a7d08e6c4d074bc18.png)

## **任意有状态操作:**

![](img/fd0fd461a19ee2c8e5209b35945191dc.png)

## **MapGroupsWithState:如何使用？**

1.定义数据结构

2.定义使用新数据更新每个分组键状态的函数

3.对分组数据集使用用户定义的函数

***user actions . group by key(_。关键)。mapGroupsWithState(updateStateFunction)***

**它适用于批处理和流式查询**

*批量查询时，每组只调用一次函数，没有先验状态*

## **FlatMapGroupsWithState:**

![](img/89bd7dab9afad3b7f87c7d0a7fedfbe2.png)

## **监控流查询:**

![](img/0efe600266b7b7c3df2f7356fcac5644.png)

## **故障恢复和存储系统要求:**

即使机器出现故障，结构化流也能保持其结果有效。为此，它对输入源和输出接收器提出了两个要求:

输入源必须是*可重放的*，以便在作业崩溃时可以重新读取最近的数据。例如，像 Amazon Kinesis 和 Apache Kafka 这样的消息总线是可重放的，文件系统输入源也是如此。

输出接收器必须支持*事务更新*，这样系统才能让一组记录原子地出现。到目前为止，结构化流为文件接收器实现了这一点。

## **带检查点的容错:**

![](img/75d9a02b8f1058d39ddb99c0ad9a2c0c.png)

## **支持的源&汇:**

![](img/37aec5ebbd163f36ddc541a98eed8f17.png)

## **性能基准:**

![](img/1084eb1fca5f8862a6ef571fdbf901f0.png)

## **更多卡夫卡支持:**

![](img/654db4a5ffde1bc56f117f2a7e41f960.png)

# 感谢阅读💜