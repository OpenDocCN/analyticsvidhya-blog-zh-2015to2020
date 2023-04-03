# “样本内”准确性的问题

> 原文：<https://medium.com/analytics-vidhya/problem-with-in-sample-accuracy-cedb68f8531d?source=collection_archive---------30----------------------->

![](img/d4e00e730aa7dffe031d23c202eb4f51.png)

斯蒂芬·道森在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

每当我们建立一个机器学习模型时，很明显我们会计算所建立模型的准确性(尽管不是全部)。

许多人在衡量预测准确性时犯了一个巨大的错误。他们用他们的*训练数据*进行预测，并将这些预测与*训练数据*中的目标值进行比较。我们可以使用***train _ test _ split***来解决这个问题。