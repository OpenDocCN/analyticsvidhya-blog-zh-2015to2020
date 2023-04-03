# 意大利的 Covid19 传输预测 SIR 模型的 python 教程

> 原文：<https://medium.com/analytics-vidhya/covid19-transmission-forecast-in-italy-a-python-tutorial-for-sri-model-8c103c0a95b9?source=collection_archive---------9----------------------->

在[上一部分](/@andrea.castiglioni/covid-19-corona-virus-analysis-in-italy-with-python-and-spread-prediction-d787d46bfc6f)中，我们看到了前 12 天意大利新冠肺炎扩散的数据分析。

在这一部分中，我们将实现一个更好的算法来拟合数据，即所谓的 SIR 模型。 [**SIR 模型**](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) 是最简单的房室模型之一，很多模型都是这种基本形式的衍生。该模型由三个部分组成: **S** 表示可接受的 **s** 的数量， **I** 表示感染的 **i** 的数量， **R** 表示恢复(或免疫)的个体的数量。该模型合理地预测了…