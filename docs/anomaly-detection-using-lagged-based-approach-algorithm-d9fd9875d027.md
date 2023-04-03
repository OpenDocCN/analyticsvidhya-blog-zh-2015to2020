# 基于滞后逼近算法的异常检测

> 原文：<https://medium.com/analytics-vidhya/anomaly-detection-using-lagged-based-approach-algorithm-d9fd9875d027?source=collection_archive---------12----------------------->

![](img/95be119421a3e7b53f4e4f21fe8cc881.png)

在这篇文章中，我将介绍一种帮助我们预测数据异常的技术。

[**异常**](https://en.wikipedia.org/wiki/Anomaly_detection) 是这些数据模式内的意外变化，或者不符合预期数据模式的事件，被认为是异常。换句话说，异常是对常规业务的偏离。

**状态监测** (CM)是监测机械状态参数(振动、温度等)的过程。)，以便识别指示发展中故障的显著变化。

**技术方面:** 从业务上了解数据后，我们偶遇一个新的算法。因此，这种模式的异常行为可能会告诉我们失败的原因。

***那么主要问题来了，“我们如何才能找到未来的异常数据”***

在进行探索性数据分析后，我们发现有 10 个参数正在影响输出变量。

**所以为了找到未来的异常数据点，我写了一个算法**。

**发现未来异常的算法:**

1.  在这种方法中，我们将把我们的问题转换成一个“回归问题”。那么“输出”参数将成为因变量，其他 10 个参数将成为自变量。
2.  根据时间戳对数据集进行排序。将时间戳转换成 pandas [datetime](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html) 格式并排序。
3.  取每个自变量的时滞，分别计算与因变量的相关性。检查代码片段。

3.对所有重要参数使用上述代码，找出相应“输出”参数和滞后自变量之间的相关性。

**例如**，假设自变量命名为“Y”，因变量命名为 X。假设我们发现(X-10)值与当前变量“Y”高度相关，这意味着“Y”与过去的 X-10 值高度相关，这意味着晚上 10.00 的“Y”值与晚上 9.55 的“X”值高度相关(10*30 秒[数据之间的时间差])。

4.同样，我们也可以找到其他参数的滞后相关性。

5.开发基于“回归”的机器学习模型，将滞后数据作为自变量，将“输出”参数作为因变量，并部署该模型。

6.使用这种方法，我们将通过以前的数据点来找到未来未知变量的“输出”值。

7.一旦我们有了未来的数据点，我们可以使用 Z 值、IQR 等来判断数据是否异常。

**结束语:** 如果大家有任何疑问，欢迎在下面评论。由于我不想增加文章的长度，所以我没有为这种方法编写太多的代码。我希望你们喜欢这个有趣的异常检测之旅。

和我连线-[https://www.linkedin.com/in/paritosh-kumar-3605a913b/](https://www.linkedin.com/in/paritosh-kumar-3605a913b/)

**参考资料**:
[https://www.appliedaicourse.com/](https://www.appliedaicourse.com/)
[https://towardsdatascience . com/how-to-use-machine-learning-for-anomaly-detection-and-condition-monitoring-6742 f 82900 D7](https://towardsdatascience.com/how-to-use-machine-learning-for-anomaly-detection-and-condition-monitoring-6742f82900d7)