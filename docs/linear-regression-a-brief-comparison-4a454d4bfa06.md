# 线性回归:一个简单的比较

> 原文：<https://medium.com/analytics-vidhya/linear-regression-a-brief-comparison-4a454d4bfa06?source=collection_archive---------25----------------------->

什么是回归，这个术语你们很多人可能听过，但它到底是什么？用最简单的话来说，这是一个统计过程，旨在建立投入和产出之间的工作关系。输入是一个与输出有特定关系的实体，最简单的回归形式是线性回归，旨在绘制输入和输出之间的简单线性关系。

![](img/e6e14d39a9b40228b0c4e9232d3b75ec.png)

线性回归方程。[来源](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.mathematics-monster.com%2Flessons%2Fhow_to_convert_a_linear_equation_from_general_form_to_slope_intercept_form.html&psig=AOvVaw29Ml2RaAlhXvST2CAzAthW&ust=1592432669488000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCNjzwpKwh-oCFQAAAAAdAAAAABAR)

这里 y 只是依赖于输入 x 的输出，因此 y 是因变量，x 是自变量。因此，对于遵循关系式的每个 x 值，存在唯一的 y，它依赖于 y 的值。

让我们考虑一个例子来更好地理解线性回归。当处理可以直接相互映射的数据时，它们之间具有特定的线性关系，(如在这种情况下，以年为单位的经验(x)和获得的工资(y))，我们可以执行线性回归，以获得因变量和自变量之间的最佳关系。

![](img/516c6aed930226279ab32391d7212e2e.png)

简单线性回归。薪资 vs 经验[来源](https://www.kaggle.com/kindlerking/linear-regression-using-gradient-descent)

以上是简单线性回归的一个例子，还存在更多形式的回归，但是在继续讨论这个话题之前，让我们先来探讨当一个简单的线性模型不能捕捉数据中最真实的关系时该怎么办？让我们来研究这样一个场景。

![](img/871e2e609bfc5ed28c33c832fca708cc.png)

我们的数据(x =级别，y =工资)

在上面的数据中，我们发现 x 和 y 之间没有明确的关系，不像以前随着经验的增加，工资也增加了，也就是说，虽然工资(y)取决于水平(x ),但明确定义的简单关系是不可能建立的。

![](img/490ce3453646a44efb68a517e77f7386.png)

线性回归(x =级别，y =工资)

> 由于这两个变量之间缺乏适当的相关性，上述模型是不合适的，因此引导我们找到一种更好的替代方法来解决上述问题。*多项式回归*是对旨在处理这种复杂关系的线性回归的改进。

![](img/3ace07e8a6a0d510442dd4d678733212.png)

多项式回归方程([来源](/analytics-vidhya/machine-learning-project-3-predict-salary-using-polynomial-regression-7024c7bace4f)

由于预测所依赖的 x 系数的性质，多项式回归仍被视为线性模型。因此，通过预处理数据以生成多项式 x 并将其拟合到线性回归模型而生成的高阶方程给我们提供了期望的预测 y，其与先前的模型相比拟合得更好。

![](img/347a5f7ccd1c9b4b47024f3c14d33a90.png)

多项式预处理+线性回归

![](img/7509a892c34232f2efd9adf9c61844f6.png)

多项式线性回归[代码](https://www.kaggle.com/kindlerking/linear-vs-polynomial-regression)

与简单的线性回归相比，上述模型更适合这种非线性复杂数据的情况，因此表明模型的选择在很大程度上取决于用例特定的方法。最后，让我们使用两个著名的指标来比较线性方法和多项式方法。

1.  **均方根误差:**用最简单的术语来说，就是由模型或估计量预测的值与观察到的值之间的差异，或常用的差异度量。均方根越低，模型预测越好。

![](img/db67270d601c75fd923a1a0b8f97ea46.png)

RMSE 的公式[来源](https://www.google.com/url?sa=i&url=https%3A%2F%2Ftowardsdatascience.com%2Fwhat-does-rmse-really-mean-806b65f2e48e&psig=AOvVaw1lXKWldXe06lCWUZysH5J8&ust=1592456293699000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCKj0po-IiOoCFQAAAAAdAAAAABAD)

2. **R2 分数:**在统计学中，决定系数通常被称为 **R2** 或**‘R 平方’**是对 y(预测值)总变异的解释变异的度量。).R2 分数越高，我们模型的预测就越好。

![](img/a8a0ef0f8b28f9ff51e01ee25c044cb6.png)

R 平方的公式[来源](https://i.stack.imgur.com/xb1VY.png)

根据我们的模型，我们发现*多项式回归与线性回归*相比，具有更高的 R2 分数和更低的 RMSE，因此表明对于具有如此直接趋势的复杂数据而言，这是首选方法。

**注意:**多项式回归预处理中的次数“n”是预处理将进行到 x 系数的次数，尽管增加预处理的次数将增加曲线拟合我们的数据并进行预测的方式，但它也会通过过度拟合来扭曲数据，因此非常低的次数(简单线性回归)将使数据欠拟合，如我们所知，而高的次数将使数据过拟合。在大多数情况下，度的最佳值在(2-3)之间，你将不得不摆弄它，用你的直觉得到最好的可能结果，而不会使你的数据有太大的偏差。

![](img/d9f60dc064b4b06bfeb88dcd948fbf88.png)

n=10 的度数严重过度拟合数据。**必须避开**

访问:[https://www . ka ggle . com/kindler king/linear-vs-polynomial-regression](https://www.kaggle.com/kindlerking/linear-vs-polynomial-regression)获取代码。这只是一系列回归文章的开始，敬请关注更多。