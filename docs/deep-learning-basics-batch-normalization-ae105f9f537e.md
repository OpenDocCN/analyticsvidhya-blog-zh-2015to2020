# 深度学习基础——批量标准化

> 原文：<https://medium.com/analytics-vidhya/deep-learning-basics-batch-normalization-ae105f9f537e?source=collection_archive---------6----------------------->

# **什么是批量正常化？**

批量标准化批量标准化层之间的网络激活，使得批量的平均值为 0，方差为 1。批处理规范化通常编写如下:

![](img/cebec8e614fdecc442e99bccc6a693da.png)

[https://py torch . org/docs/stable/generated/torch . nn . batch norm 2d . html](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)

> 通过小批量计算每个维度的平均值和标准差，γ和β是可学习的…