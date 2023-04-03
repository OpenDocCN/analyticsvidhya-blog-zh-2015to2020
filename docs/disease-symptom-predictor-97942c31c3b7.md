# 疾病预测者

> 原文：<https://medium.com/analytics-vidhya/disease-symptom-predictor-97942c31c3b7?source=collection_archive---------12----------------------->

这是我的深度学习项目；根据数据中给出的症状对 41 种疾病进行数据分析和预测。我使用了 Pandas、NumPy、Seaborn、TensorFlow、Keras 和 Scikit learn 库来分析数据和建立模型。

![](img/c3d5c9a3d1697b021d1ff49659418cce.png)

在 pandas 库的帮助下，我将 CSV 作为“数据”变量导入并分析了它。

![](img/258f0a339b4a690300094b860a3f1467.png)

data.head()。转置()

![](img/856b6d2238626f546675aba1b57f7d89.png)

data . description()。转置()

![](img/be27ce6894e8275758906d56ad7cacd7.png)

data.isna()。总和()

在分析数据之后，我们可以看到在“症状 _5”特征之后大部分是 NaN 值，所以我移除了那些特征，并用 0 替换剩余的 NaN 值。

![](img/e38963e66e3ced3cc5072085640151e9.png)

现在是特征工程，我用数值代替了症状，用数字(0 到 40)代替了疾病。

![](img/35473bd0f05abe4b455049738d86a349.png)![](img/20616485405541b95d7a818f65a860c5.png)![](img/10dcb8c1d98741649c71aaee474a3026.png)

data.head(15)

![](img/bb8b06d7221b246b17b40fbcc183d371.png)

sns.countplot(数据['疾病'])

我将数据分为 x 和 y 变量，并将它们分为训练测试数据，即“x_train”、“x_test”、“y_train”、“y_test”。在通过人工神经网络发送之前，我对它们进行了缩放，以使数据标准化。我在“y_train”和“y_test”上进行了一次性编码。

![](img/d22d5f4a3c171ab849afed83e98f7661.png)

我建立了一个人工神经网络对这些疾病进行分类，并做了一个分类报告，对 41 种疾病的平均准确率为 91%。

![](img/9bbf969af5de43372af49378ad443a48.png)![](img/91133e4cdaa83b43c1137351a1872599.png)

结论

我做这些都是为了练习人工神经网络，以下是我在做这个项目时学到的所有东西:

> *学会了如何使用以及何时使用人工神经网络。*
> 
> *了解了深度学习相对于其他基本机器学习算法的优势和劣势。*

Github 项目:[疾病预测](https://github.com/codename-hyper/Disease-Prediction)