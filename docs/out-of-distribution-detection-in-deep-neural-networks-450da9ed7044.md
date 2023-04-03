# 深度神经网络中的非分布检测

> 原文：<https://medium.com/analytics-vidhya/out-of-distribution-detection-in-deep-neural-networks-450da9ed7044?source=collection_archive---------0----------------------->

> 让人工智能系统变得健壮和可靠

深度神经网络通常使用 [**封闭世界假设**](https://en.wikipedia.org/wiki/Closed-world_assumption) 进行训练，即假设测试数据分布与训练数据分布相似。然而，当在现实世界的任务中使用时，这种假设不成立，导致他们的性能显著下降。虽然这种性能下降对于产品推荐这样的宽容应用来说是可以接受的，但是在医学和家用机器人这样的不宽容领域使用这样的系统是危险的…