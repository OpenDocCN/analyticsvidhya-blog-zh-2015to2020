# 在 TensorFlow 1 和 2 中使用 TensorBoard 的基础知识

> 原文：<https://medium.com/analytics-vidhya/basics-of-using-tensorboard-in-tensorflow-1-2-b715b068ac5a?source=collection_archive---------3----------------------->

## 停止使用 Matplotlib 来绘制你的损失——可视化图表和模型，过滤器，损失…

![](img/17930755b828c7850d061a5b1fa8d7c9.png)

编辑自 [Pankaj Patel](https://unsplash.com/@pankajpatel?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的一张照片

TensorBoard 是一个强大的可视化工具，直接内置于 TensorFlow 中，允许您在 ML 模型中找到洞察力。

TensorBoard 可以可视化任何东西，从标量(例如，损失/准确性)，到图像…