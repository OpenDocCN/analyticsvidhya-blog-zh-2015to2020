# 机器学习在多大程度上得到了应用？

> 原文：<https://medium.com/analytics-vidhya/how-big-tech-use-machine-learning-446d2d3bf686?source=collection_archive---------19----------------------->

## K inda 深刻的见解

让我们从什么是机器学习开始。

**机器学习**是人工智能的一个子集，它为系统提供了自动学习和根据经验改进的能力，而无需显式编程。

*   机器学习是数据科学家用来分析数据的方法，以便自动化系统的分析建模部分。
*   该系统从大量数据中学习，识别模式，然后在最少的人工干预下做出预测。

机器学习的普遍用途是 Play Store 和 App Store 推荐、谷歌地图、电子邮件过滤、谷歌翻译、谷歌搜索等等。让我们详细看看 5 个机器学习用例:

1.  App Store 和 Play Store 推荐
2.  交通和通勤
3.  Gmail 的电子邮件过滤
4.  谷歌搜索
5.  用于客户支持查询的聊天机器人等等。

**1。App Store 和 Play Store 推荐:**与 DeepMind 合作的 Google play 采用了三种主要模式:

*   候选生成器
*   重新排序
*   多目标优化模型

他们尝试了 3 种不同的解决方案，其中包括使用 LSTM(长短期记忆)，这获得了显著的准确性增益，但导致了服务延迟，因为 LSTM 在计算上是敲诈性的。第二个解决方案是用用于序列到序列预测的转换器模型代替 LSTM，并且在 NLP 中产生了显著的结果。这不仅提高了效率，也增加了培训成本。第三个也是最后一个解决方案是实现一个有效的附加注意模型，该模型适用于序列特征的任何组合，同时产生较低的计算成本。

**候选生成器**是一个深度检索模型，可以分析超过百万个 app，检索出最合适的。它从用户先前安装的内容中学习。它还了解到一种偏好，即下载量是其他应用的 10 倍以上的应用。为了纠正这种偏差，他们在模型中实施了加权。它基于每个应用程序的印象安装率与整个 Play store 的印象安装率的中位数进行比较。因此，安装率低于中值的应用程序的重要性将超过小于 1 的权重。

对于每个应用程序，都有一个**重新排序器**，这是一个用户偏好模型，可以从多个维度预测用户的偏好。通常，许多推荐系统使用二元分类法来解决排名问题，这种方法一次只能对一个项目进行排名，无法捕捉应用程序可能相似或不相似的上下文。这个问题的解决方案是 Reranker 模型，在这个模型中，它了解了同时显示给用户的一对应用程序的重要性。然后，用户选择下载的应用将被分配一个正面或负面的标签，该模型将试图减少排名中的反转次数，从而提高应用的整体相对排名。

接下来，这些预测是多目标优化模型的输入，该模型的解决方案为用户提供了最合适的候选。他们使用的算法试图在许多指标之间找到一个折衷，并沿着折衷曲线找到合适的点。

参考: [DeepMind 博客](https://deepmind.com/blog/article/Advanced-machine-learning-helps-Play-Store-users-discover-personalised-apps)

**2。优步如何使用机器学习:**优步从生产 3 个模型到生产 10，000 个模型，现在可能更多。他们允许他们的数据科学家在 GCP、Tensorflow、Keras 训练模型，并以稳健的方式为所有这些模型提供服务。优步必须收集大量数据，以便找到最佳路线，预测不断变化的市场需求，应对潜在的欺诈行为等等。优步使用**米开朗基罗**开源组件像**[HDFS](http://hadoop.apache.org/)[Spark](https://spark.apache.org/)[萨姆扎](http://samza.apache.org/)[卡珊德拉](http://cassandra.apache.org/)[ml lib](https://spark.apache.org/mllib/)[XGBoost](https://github.com/dmlc/xgboost)[tensor flow](https://www.tensorflow.org/)。**

**![](img/d0d7f5c0b4de1de717e2e65c5f9d4009.png)**

**图 1 .米开朗基罗:优步的机器学习平台**

**如果你听说过 **UberEATS** ，那么你会更兴奋地知道，UberEATS 有几个模型在米开朗基罗上运行，比如覆盖送餐时间预测、搜索排名、搜索自动完成和餐馆排名。**

**![](img/574e1c354ad70220614ea39ac64cc6b8.png)**

**图 2 . Uber eats 应用程序拥有一个由基于米开朗基罗的机器学习模型支持的估计交付时间功能。**

**你可以在米开朗基罗官方博客下面的链接中对他有更深入的了解。**

**参考:[优步工程 Youtube 视频](https://www.youtube.com/watch?v=DOwDIHzN5bs)，[米开朗基罗博客](https://eng.uber.com/michelangelo-machine-learning-platform/)**

****3。Gmail 的邮件过滤功能:**如果你使用 Gmail 已经有一段时间了，那么你一定已经注意到它在过滤垃圾邮件和重要邮件方面有多好。像谷歌这样的科技巨头使用 TensorFlow 来帮助 Gmail 用户过滤额外的垃圾邮件。借助 TensorFlow，他们每天成功拦截大约 1 亿封额外的垃圾邮件。借助 TensorFlow，他们成功阻止了数百万难以识别的垃圾邮件，如基于图像的垃圾邮件、嵌入内容的电子邮件以及来自新创建的域的邮件，这些邮件在合法流量中含有少量垃圾邮件。Gmail 团队设法在收件箱中获得 0.05%的垃圾邮件，这意味着大约 99%的准确率。**

**参考:[谷歌云博客](https://cloud.google.com/blog/products/g-suite/ridding-gmail-of-100-million-more-spam-messages-with-tensorflow)， [Gmail 博客](https://gmail.googleblog.com/2015/07/the-mail-you-want-not-spam-you-dont.html)**

****如果你读到最后并且喜欢这篇文章，可以考虑查看参考文献，以获得更多关于机器学习的见解。****

****感谢您的阅读:)****