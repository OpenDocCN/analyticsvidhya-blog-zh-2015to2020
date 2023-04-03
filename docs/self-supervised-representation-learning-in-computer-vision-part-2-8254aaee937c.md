# 计算机视觉中的自我监督表示学习——第二部分

> 原文：<https://medium.com/analytics-vidhya/self-supervised-representation-learning-in-computer-vision-part-2-8254aaee937c?source=collection_archive---------2----------------------->

[**本系列的第 1 部分**](/@gauss_hat/self-supervised-representation-learning-in-computer-vision-part-1-215af945d23a) 着眼于表象学习，以及自我监督学习如何缓解学习图像表象时的数据低效问题。

这是通过“**对比学习**”实现的，这是一种用于学习相似性和区别的设计范例。这种范式归结为让一个模型理解，相似的事物在表示上应该靠得更近，不相似的事物应该离得更远。