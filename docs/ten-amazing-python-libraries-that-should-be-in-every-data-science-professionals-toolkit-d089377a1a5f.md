# 十个令人惊叹的 Python 库，应该是每个数据科学专业人员的工具包

> 原文：<https://medium.com/analytics-vidhya/ten-amazing-python-libraries-that-should-be-in-every-data-science-professionals-toolkit-d089377a1a5f?source=collection_archive---------9----------------------->

![](img/abb82335d5e747aa6f1f2b414fea9743.png)

来源:https://unsplash.com/photos/sfL_QOnmy00

> 作为一名职业数据科学家不同于作为一名爱好者。虽然后者可以通过试错来试验和学习，但专业人员需要快速得出结果！这是我在工作中最常用的方法。我已经包括了可视化库、特征工程和统计测试库、低代码 autoML 库、ML 调试库，最后还有一个预测库和一个深度学习库。虽然这比标准的(Pandas、Numpy、Sklearn 和 matplotlib)更有效，但我希望读者会觉得这很有用。

## 薄层

如果你从 Tableau 来，错过了那些很酷的地图，leav 是你的新朋友。它不仅利用了矢量和光栅图层，还可以生成交互式地图，渲染后可以放大。

[官方文件](https://python-visualization.github.io/folium/modules.html)在这里。
[这款 Kaggle 笔记本](https://www.kaggle.com/daveianhickey/how-to-folium-for-maps-heatmaps-time-analysis)展示了它在地图、热图和时间分析方面的应用。

## 海生的

虽然 seaborn 主要用于探索性分析和建模结果，但它也为专业视觉思考者提供高维度图。它的声明式 API 可以让您专注于绘图中不同元素的含义，而不是如何绘制它们的细节。

[官方示例画廊](https://seaborn.pydata.org/examples/index.html)是一个值得探索的地方。

## 散景

专为浏览器内可视化而构建，如果制作的图形需要嵌入到 PPT 和幻灯片中，请使用它。如果您需要在 web 应用程序中嵌入绘图，或者跨应用程序和服务器导出，请选择此选项。

从这里开始[。](https://towardsdatascience.com/data-visualization-with-bokeh-in-python-part-one-getting-started-a11655a467d4)

## 功能工具

数据科学专业人员在功能工程上花费的时间是巨大的。如果您还没有，建议您这样做。该库可生成用于分析的特征，并可在执行特征工程时派上用场。但是，建议将其与手头的业务问题结合使用。

[这里的](/@rrfd/simple-automatic-feature-engineering-using-featuretools-in-python-for-classification-b1308040e183)是一个惊人的资源开始。
[这本 Kaggle 笔记本](https://www.kaggle.com/willkoehrsen/featuretools-for-good)告诉你怎么做。

## 统计模型

如果您怀念 Python 中 R 结果的简单性，那么这个库就是为您准备的。
以其进行统计测试和数据探索的能力而闻名，它还包含回归和时间序列模型。

[这里的](https://www.statsmodels.org/stable/index.html)是官方文档。

## PyCaret

一个低代码的 autoML 库，允许它的用户用几行代码构建多个模型，这个 gem 非常适合那些我们都在做的快速而肮脏的原型制作。

官方文件是[这里是](https://pycaret.org/)。
我用 PyCaret [写了一个 Kaggle 笔记本提交给泰坦尼克号生存预测这里](https://www.kaggle.com/kritidoneria/titanic-using-pycaret-100-lines-of-code/comments)。

## ELI5

*我不管你的模型是干什么的，把我当五岁小孩解释给我听。这个库是最简单的解释能力，很像它的 Reddit 启发的名字。这是一种非常方便的调试 ML 模型的方法，通常用于解释深度学习预测。*

本 [Kaggle 笔记本](https://www.kaggle.com/kritidoneria/explainable-ai-eli5-lime-and-shap)中提到了一个工作实例。

## PyTorch

深度学习在专业设置中的应用并不广泛，但对于不确定的目标，有时尤其是 NLP 和图像数据，它确实很方便。

[这个](https://pytorch.org/)解释了 PyTorch 支持的功能、环境和特性。

## Catboost

CatBoost 通常被称为 XGBoost 的后继者，它利用有序提升，可以独立处理分类数据，从而得到更好的结果。它最适合涉及高度异构数据的设置。

[Catboost 有什么特别之处？](/@hanishsidhu/whats-so-special-about-catboost-335d64d754ae)可以回答你的任何问题。

## 骗子

不够或者没有测试数据？没问题。需要匿名化的数据？没问题。Faker 生成假数据供你摆弄。

[官方文档](https://pypi.org/project/Faker/)在这里，快速入门的例子是[这个](https://www.geeksforgeeks.org/python-faker-library/)。