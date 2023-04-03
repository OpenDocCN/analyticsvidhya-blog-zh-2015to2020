# 知道在梯度下降中优化学习率的更好方法

> 原文：<https://medium.com/analytics-vidhya/a-better-way-to-estimate-learning-rates-in-gradient-descent-5f30de053da?source=collection_archive---------23----------------------->

批量梯度下降是任何早期机器学习课程或书籍中教授的最重要的思想之一。作为迭代优化函数参数的最基本实现之一，它成为任何有抱负的数据科学家最喜爱的工具。

作为一名本科生，我也喜欢深入研究算法，因此提出了一个看似独特的想法来估计梯度下降的超参数调整算法中的“学习率”。