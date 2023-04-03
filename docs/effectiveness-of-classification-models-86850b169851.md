# 分类模型的有效性

> 原文：<https://medium.com/analytics-vidhya/effectiveness-of-classification-models-86850b169851?source=collection_archive---------10----------------------->

![](img/a4ff7390e7219a35483a51eaba1fa58f.png)

由:[https://apartamento 702 . com . br/o-maior-evento-de-divulgacao-de-software-livre-na-America-Latina-acontece-neste-sabado-em-natal/](https://apartamento702.com.br/o-maior-evento-de-divulgacao-de-software-livre-na-america-latina-acontece-neste-sabado-em-natal/)

机器学习是人工智能概念的一部分，旨在提供模型，使机器能够执行将由人执行的任务。这些模型由预定义的规则组成，允许计算机根据先前的数据和用户使用的数据做出决策。

有一些用于分类的模型，每个分类问题都需要一个特定的模型。为了找出哪个模型对我们的分类问题最有效，有一些评估指标可以应用于我们的模型。

在我们看到这些指标之前，我们需要理解围绕它们的基本概念。为了获得度量应用的必要变量，我们需要执行一个混淆矩阵，这是一个表，其中显示了我们的模型的错误和命中，并与预期结果进行了比较。下图显示了混淆矩阵的工作原理:

![](img/7a4ba161ff86fbb66fa8da4438ff051a.png)

作者:[http://rasbt . github . io/mlx tend/user _ guide/evaluate/mission _ matrix _ files/mission _ matrix _ 1 . png](http://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix_files/confusion_matrix_1.png)

*   **真阳性(TP):** 阳性类的正确分类；
*   **假阴性(FN):** 当预期值为阳性类别时，模型预测为阴性类别的错误；
*   **假阳性(FP):** 期望值为负类时模型预测为正类的误差；
*   **真否定(TN):** 否定类的正确分类。

## 评级模型评估指标

通过对混淆矩阵中获得的所有术语进行计数，我们可以将这些值应用于分类模型中的评估指标。

*   **准确性:**这将表明我们模型的整体性能。使用我们的模型得出的所有正确评级；
*   **精度:**这将指示在我们的模型做出的所有正面类别评级中，有多少是正确的；
*   **回忆:**这将指示在所有作为期望值的正面等级评定中，有多少是正确的；
*   **F1-得分:**表示准确度和召回率之间的调和平均值。

## 密码

下面是实现其中一些评估指标并将它们应用于一些分类模型的代码部分。应用的指标有:

**精度:**

**召回**

**F1-得分**

**功能总结**

下面的代码片段总结了在前面实现的函数中获得的结果，以及精度、召回率和 F 分数指标的结果。

## 结果

被评估的模型有 J48、朴素贝叶斯、C50、SVM、One R、JRip、Random Forest、SMO，所有数据都被执行并比较结果，以了解每个模型在应用数据中的有效性。下面我们可以看到每个代码片段生成了什么，以及为比较模型之间的度量而生成的结果。最后，我们会有一个模型的比较图表。

代码和数据库在:【https://github.com/icdatanalysis/data-analysis 

请记住，每个问题都需要一个特定的模型来解决，因此根据您想要解决的问题，模型之间的结果可能会有很大的不同。

## 参考

默克里诺。《联合国大会 1984 年选民登记册基础上的 J48 号决定执行算法》。2013 年。https://prezi.com/rx9kqu3rm6ye/arvore-de-decisao-j48/>。acesso em:2019 年 11 月 11 日。

坎迪亚戈洛伦佐。朴素贝叶斯分类算法。 2019。disponível em:<https://www . organic digital . com/blog/algoritmo-de-classicacao-naive-Bayes/>。acesso em:2019 年 11 月 11 日。

罗斯，彼得。 **OneR:最简单的方法。** 2000。http://www.soc.napier.ac.uk/~peter/vldb/dm/node8.html>。acesso em:2019 年 11 月 11 日。

何塞尼多·科斯塔·达·席尔瓦。**阿普伦多·艾姆·弗洛雷斯塔·阿莱托利亚。** 2018。disponível em:<https://medium . com/machina-sapiens/o-algoritmo-da-Flores ta-aleat % C3 % B3 RIA-3545 F6 babdf 8>。acesso em:2019 年 11 月 11 日。

J·普拉特..**利用序列最小优化快速训练支持向量机。** 1998。disponível em:<http://WEKA . SourceForge . net/doc . dev/WEKA/classifiers/functions/smo . html>。acesso em:2019 年 11 月 11 日。