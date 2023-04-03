# 不平衡数据下的随机森林分类器

> 原文：<https://medium.com/analytics-vidhya/a-random-forest-classifier-with-imbalanced-data-7ef4d9ebedb8?source=collection_archive---------3----------------------->

我最近用来自 [Pump It Up 的数据集完成了一个项目:从 DRIVENDATA](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) 中挖掘地下水位。我将遍历随机森林分类器，这是我测试的分类器之一，在调整其超参数后，我发现它的性能最好。

我不会在这里深入讨论，但是在数据准备好用于模型之前，有大量的数据清理和特征选择工作要做。有许多丢失的数据以及几个重复或接近重复的特征，再加上少量带有…的重复实例