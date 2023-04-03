# NEAs —探索哨兵数据

> 原文：<https://medium.com/analytics-vidhya/neas-exploring-sentry-data-fd9dde7ecab9?source=collection_archive---------14----------------------->

我在上一篇文章中介绍了一些小行星和 NEA 的基础知识。在本帖中，我们将检查一些 Near 近地天体数据，并尝试理解与潜在危险/危险近地天体(近地天体)相关的一些指标。我们开始吧。

我们的探索从一些通过 CNEOS API 公开可用的数据开始，具体来说，我将检查下面的[数据集](https://cneos.jpl.nasa.gov/sentry/)。

设置以下值“随时观察”、“任何撞击概率”、“任何巴勒莫标度”和“任何 H”会返回一个数据库查询，在编写本文时，该查询会生成一个 990 行的数据集。我…