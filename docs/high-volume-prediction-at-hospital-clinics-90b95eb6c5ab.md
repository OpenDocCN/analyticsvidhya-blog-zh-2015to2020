# 医院诊所的高容量预测

> 原文：<https://medium.com/analytics-vidhya/high-volume-prediction-at-hospital-clinics-90b95eb6c5ab?source=collection_archive---------1----------------------->

![](img/ac65cdc048d571b0f896d0301ed23f32.png)

最近，我的一个顾客想出了一个主意。他希望构建一个高级分析和人工智能解决方案，来测量和预测医院诊所的高负载。动机是减少诊所排队长度和患者总等待时间。诊所运营经理将能够通过采取数据驱动的决策，实时更好地分配和利用人力资源，及早做出响应，以防止或减少高负载情况。

[例如:系统可以通过在高负荷发生前两个小时从子科室#3 调换两个护士、一个医生和一个秘书来提前建议加强子科室#2，并且这样实际上防止子科室#2 的病人等待很长时间。]

**难题/问题定义+潜在特征:**

许多医院诊所正在经历意想不到的沉重的病人负担。经常会出现这样的情况，病人要经历很长的等待时间，而一些工作人员却无所事事。在许多情况下，再分配发生得太晚了。经过几天在诊所内的调查，我们考虑并指出以下原因:

1.  **缺少医生** —在许多情况下，意外地，医生被紧急呼叫到急诊室，这明显扰乱了预先计划的分配并引入了高容量瓶颈。
2.  **患者资料** —年龄、既往诊断、入院史、母语:这些参数似乎强烈影响患者在诊所停留的总时间。
3.  **“不请自来的患者”** —这些患者占每天就诊总数的 10%以上。这明显延长了“饥饿”时间并影响了一般临床的高负荷和排队延迟。
4.  **复杂流程—** 许多诊所包括非平凡站(例如，挂号= >准备= >咨询检查 1 = >医生检查= >咨询检查 2 = >回诊医生= >下次就诊挂号)。
5.  **天气**——可能导致患者和服务提供者延迟到达。
6.  **公共交通** —可能导致患者和服务提供者延迟到达。
7.  **各种服务商**和昂贵的设备。(医生、护士、技术员、秘书)。

总之，我们得出结论，这些是主要的意外和不确定因素，这些因素使得预先规划流程和临时实时响应对诊所运营经理来说如此具有挑战性，我们认为我们可以开始并验证这些功能作为我们机器学习模型的候选。

**建议的解决方案:** 具有高容量预测(发生前 60-120 分钟)的实时可视化系统，具有高度重要测量值的高级分析跟踪。

**高音量定义建议:** 我们不能——不能逃避它。我们必须弄清楚**高负荷**是什么意思，那么让我们:

*   以分钟为单位决定我们的预测时间，并将其标记为***pt =【60，90，120】***
*   定义等待站并标记为 ***st = [reg，ref1..3、*【医生总数】**
*   等待列表中的最短停留时间为 ***wt = [0…720]***

现在我们可以将我们的预测结果标记为 ***p(n，pt，st，wt)***

例如:p(n，120，' ref2 '，0)是从现在起 120 分钟后，预计等待至少 0 分钟的“参考-2”等待列表的预测等待患者数。(这意味着该时间点的所有等待患者)

请注意:我们还没有定义“n ”,因为这个参数将是我们实时特征向量列表的一个组成部分，只有在数据探索和数据准备阶段之后才会被发现。

现在，如果以下至少有一项为真，我们将把该预测标记为高负载:

*   p(n，x1，x2，0])预测结果高于该小时的实际平均值加上三倍标准差(现在是+ x1)，以及函数(x2)。
*   p(n，x1，x2，60])的结果是超过 25%的来自‘x1’的等待列表达到了至少 60 分钟的延迟。

**概念验证:** 我们选择了一家我们检查过的诊所，然后采取了以下步骤:

1.  高级架构和数据摄取。

2.数据探索。

3.数据准备和特征工程。

4. [AutoML](https://en.wikipedia.org/wiki/Automated_machine_learning) —回归模型选择&训练。

5.DeployML —回归模型部署。

6.可视化—实时流式数据集可视化，包括 ML 结果。

# 高级架构和数据接收:

我们的客户在本地环境中托管了应用程序。第一步，我们希望将数据持续复制到他们的 azure 订阅中。

我们建议将所有相关的数据源合并到 DMZ 环境中的专用 sql 服务器中，然后[将数据复制到](https://docs.microsoft.com/en-us/azure/sql-database/replication-to-sql-database) [azure sql](https://azure.microsoft.com/en-us/services/sql-database/) 中。

[数据工厂](https://docs.microsoft.com/en-us/azure/data-factory/connector-sql-server)和 [SQL 数据同步](https://docs.microsoft.com/en-us/azure/sql-database/sql-database-sync-data)可能是此同步作业的更好选择，因为它们提供托管服务。

![](img/79a86b83f7f2cfe8a38de025c2f713bf.png)

第二步是(近)实时地将数据流式传输到可视化数据集。我们提供了循环逻辑应用程序来查询数据并将其传输到 EventHub。

![](img/192e22fe45df258788a56ec1d8f07533.png)

在这一点上，我们创建了一个新的 [Azure Stream Analytics](https://azure.microsoft.com/en-us/services/stream-analytics/) 服务来连接这些点，并为可视化平台提供从事件中心到流数据集的访问。你可以在这里找到更多关于[的细节。](https://docs.microsoft.com/en-us/azure/stream-analytics/stream-analytics-power-bi-dashboard)

# 数据探索:

我们将历史 queueDB 和 DemographicsDB 数据上传到 azure blobs 进行数据探索。我选择了 [Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/) ，在我看来，这是默认的托管平台，适合大数据探索和准备。

在这一点上，我考虑过使用时序服务，比如 [Kusto](https://azure.microsoft.com/en-us/services/data-explorer/) ，但是为了简单起见，我选择了 Databricks。

这个平台的另一个主要优势是多语言编程支持。正如附件中的笔记片段所示，我用 sql 写了几个部分，用 python 写了一些其他部分，当然还有我喜欢的 spark 编程语言——Scala。
[这里](https://www.reddit.com/r/IAmA/comments/31bkue/im_matei_zaharia_creator_of_spark_and_cto_at/cq02nbk/)是 Matei Zaharia(Spark 的创建者和 Databricks 的 CTO)关于他为什么选择 Scala 而不是其他编程语言来创建 Spark 的回应。

![](img/e8e4579a685eb3405497a98d16b82746.png)

按星期几计划的访问

![](img/c5b9d91fa1ff8f9d9db54330602e5c63.png)

一周中各天的实际访问量(每小时汇总)

![](img/be2db2dfe09d4a21cecf0362fbde306a.png)

按小时计算的实际访问量与一周中的某一天进行比较

![](img/9e7a30cb494716bdaf8bb58d5d433c40.png)

诊所负荷热图

![](img/0e035af235784435d73b936bd67613fd.png)

按日期统计的总住院时间异常

![](img/00934efd4df1c3331a22b3abc5ab3df9.png)

诊所工作时间内住院时间长

# 数据准备和特征工程:

现在我们已经消化并探索了这些数据。接下来我们可以做的是特征工程。

特征工程基本上是一种从现有数据中寻找特征或数据的技术。有几种方法可以做到这一点。更多的时候，这只是一个常识问题。

在这一阶段，我们考虑将每天(周日-周五)的工作时间划分为 15 分钟的时段，在每个时段中标记 0 或 1，以表示患者正在等待。

![](img/8adcb682ecdce6ccd407b88327f94848.png)

由患者元数据加入的实际检入-检出

下一件事是按日期列分组，并通过合计每个时段进行聚合。

![](img/c8627ac163fe2e2fcf91e56cfb9265ad.png)

数据分为 15 分钟时段

现在，我们想在每个桶中添加一个带有患者“人口统计”数据的向量，例如:性别、年龄、住院时间、以前的诊断、以前的入院天数等。

![](img/e2b2e4174a7a9820f30153c974a3312b.png)

添加每个特征的向量

现在再按日期分组，把所有这些向量加起来。

![](img/fd4aa0e32e7a20f05d17f0e64229fd45.png)

按每个时段(列)的日期按所有向量分组

在这一点上，我想对 bucket 中的每个向量进行平面映射，这样每一列将被分成 8 列(向量中的每个单元格一列)。在分离之后，我们需要重新排列列来组织特征 *f1，..，fn* 带有相关标签。

我划分了数据集，并用 [pandas](https://pandas.pydata.org/) 重新排列它(从 spark 数据帧切换到 Pandas 数据帧相当容易，因为您只需要导入 Pandas 并调用 df.toPandas())

![](img/62f8fe0ec1587c03ccdae31051cfc8de.png)

要从 spark 数据帧转换到 panda 数据帧的 toPandas()

# 模型培训和自动化:

因此，在将数据集适当地排列为特征和标签后，我们可以尝试执行几种经典的[监督学习](https://en.wikipedia.org/wiki/Supervised_learning)算法，并对这些回归技术的性能进行比较。

为了这个任务，我尝试了 [Azure Automated ML UI](https://azure.microsoft.com/en-us/blog/simplifying-ai-with-automated-ml-no-code-web-interface/) ，它现在是 Azure 机器学习服务 workspace 的集成部分。你可以从总体上了解更多关于 [AutoML](https://en.wikipedia.org/wiki/Automated_machine_learning) 流程的信息。

## Azure 自动化 ML:

![](img/58357b1f1550d2d982d54be7617d93a9.png)

自动机器学习性能范围

![](img/e0dc6d2a2223c09f92b3022a1556798f.png)

生成按绩效排序的回归模型列表

![](img/e851dddfeb642da3020d265eb06ea46d.png)

我也尝试了[快速矿工](https://docs.rapidminer.com/latest/server/cloud/microsoft-azure.html)-
**快速矿工:**

![](img/2c45f52acf364c69df53d1c1fadfc552.png)![](img/1df589e02d2fc94a32f94464e3b32a63.png)![](img/b7e1cc31a49136ab0b7e98dcaf0933f7.png)![](img/ac40ce728fce8ddf25ab63329daee86d.png)

预测结果与实际结果

**Azure ML 工作室:**

在这一步，我将准备好的数据集上传到 azure ml studio，并毫不费力地创建并运行了实验。

![](img/74bffb798fcc86c82a84308f2bdead7c.png)![](img/c8589a3c9e4cfe921da8add6e778c5c7.png)

预测结果与现实结果(标签)

![](img/7ebc618b4897be822a1554db63fee82d.png)

## ML 部署:

部署部分是通过将预测的实验作为 web 服务部署在一个 [AKS](https://docs.microsoft.com/en-us/azure/aks/) (Azure Kubernetes 服务)上。

![](img/ab3bbb4052c8e0eb0c31e8d6dd05f03b.png)

ML 部署完成后，我们使用另一个 LogicApp 将在线数据和预测数据合并到一个 sql 视图中，然后将视图作为流数据集传输到前面提到的流中。
(LogicApp =>EventHub =>Stream Analytics =>power bi)

![](img/eaabcb179f80b7cef320f4ad8c726bbc.png)

60 分钟和 120 分钟预测结果流

更糟糕的是，我们已经为客户构建了另一个视图，其中保存了所需参数的最后八个快照。每个 15 分钟时段的每一行。

![](img/abb0cd82b4b63e75270427851d7046f0.png)

# 实时仪表板— Power BI:

![](img/8f5b54df92f11073bb7a0f6b7b471181.png)

实时仪表板加上预测结果

![](img/889977799238ecffca655dbeca1a1c1b.png)

实时仪表板加上预测结果

![](img/3ae40506c1941bac81a5140b40ce306f.png)

预测比较

![](img/7b44955142b8a6d685aaa494696f4dcb.png)

## 后续步骤:

展望未来，我们将整合来自[开放天气](https://openweathermap.org/current)和 Moovit 的 API。这将通过天气和公共交通信息丰富我们的数据，这可能会提高模型的准确性。

开放天气非常简单，每分钟您可以免费接听 60 个电话。

![](img/2bcf8abfacf2be73f232fb779a49e9d6.png)

**完整源代码:** 数据库笔记本源代码可以在这里找到。

我要感谢我的同事 [Kfir Gur Ari](https://il.linkedin.com/in/kfirgurari) 帮助我完成 PowerBI 可视化部分，并感谢**[catlin Esanu](https://www.linkedin.com/in/catalinesanu)**、**[Guy bertenal](https://il.linkedin.com/in/guybe)、**、**帮助我建造高层建筑并为我提供建议。**

**Avi Paz，
云解决方案架构师-数据&AI
Avi.Paz@microsoft.com**