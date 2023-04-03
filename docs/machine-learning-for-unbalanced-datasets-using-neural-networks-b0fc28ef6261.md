# 基于神经网络的不平衡数据集机器学习

> 原文：<https://medium.com/analytics-vidhya/machine-learning-for-unbalanced-datasets-using-neural-networks-b0fc28ef6261?source=collection_archive---------1----------------------->

## 在不平衡数据集的情况下，神经网络可以用于二分类吗？

有几种方法可以解决不平衡的数据集:从内置的[逻辑回归](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)和 [sklearn 估计器](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)中的 *class_weight* 到手动过采样，以及 [SMOTE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html) 。我们将研究神经网络是否可以作为一种可靠的现成解决方案，以及可以调整哪些参数来实现更好的性能。