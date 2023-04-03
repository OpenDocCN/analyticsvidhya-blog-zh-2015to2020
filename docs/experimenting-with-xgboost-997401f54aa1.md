# 使用 XGBoost 进行实验

> 原文：<https://medium.com/analytics-vidhya/experimenting-with-xgboost-997401f54aa1?source=collection_archive---------18----------------------->

![](img/cfd8bec7e39ac5af28a8c39a3c69c9ae.png)

萨法尔·萨法罗夫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 介绍

这是一篇旨在通过一些机器学习的简单例子来了解 XGBoost API 的文章。我将调整一些超级参数，并使用其内置的交叉验证。

# 数据集

我们将使用我从 Kaggle 获得的 [NYC Air Bnb 数据集。](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)

# 数据集的基本步骤

我们不会在这里深入探讨 EDA，因为我们的目标是熟悉 XGBoost 的 API。但是，我们需要做一些数据处理，因为我们的数据集混合了分类数据和数值数据。首先，让我们导入必要的库。

```
import xgboost as xgbimport pandas as pdimport numpy as npfrom sklearn.model_selection import train_test_splitfrom sklearn.metrics import mean_absolute_errorfrom sklearn.pipeline import Pipelinefrom sklearn.preprocessing import StandardScalerfrom sklearn.pipeline import FeatureUnionfrom sklearn.base import BaseEstimator, TransformerMixinfrom sklearn.preprocessing import Imputerfrom sklearn.preprocessing import LabelEncoderfrom sklearn.pipeline import FeatureUnionfrom sklearn.model_selection import GridSearchCV
```

通过简单地使用一些基本的 Pandas 功能，我们可以清楚地看到有许多缺失的功能，特别是在“last_review”和“reviews_per_month”列中。我们会放弃他们。

```
airbnb.isnull().sum()id                                    0
name                                 16
host_id                               0
host_name                            21
neighbourhood_group                   0
neighbourhood                         0
latitude                              0
longitude                             0
room_type                             0
price                                 0
minimum_nights                        0
number_of_reviews                     0
last_review                       10052
reviews_per_month                 10052
calculated_host_listings_count        0
availability_365                      0
dtype: int64
```

之后，我们使用 **info()** 函数告诉我们每一列的类型。

```
airbnb.info()<class 'pandas.core.frame.DataFrame'>
RangeIndex: 48895 entries, 0 to 48894
Data columns (total 14 columns):
id                                48895 non-null int64
name                              48879 non-null object
host_id                           48895 non-null int64
host_name                         48874 non-null object
neighbourhood_group               48895 non-null object
neighbourhood                     48895 non-null object
latitude                          48895 non-null float64
longitude                         48895 non-null float64
room_type                         48895 non-null object
price                             48895 non-null int64
minimum_nights                    48895 non-null int64
number_of_reviews                 48895 non-null int64
calculated_host_listings_count    48895 non-null int64
availability_365                  48895 non-null int64
dtypes: float64(2), int64(7), object(5)
memory usage: 5.2+ MB
```

由于我们有几个分类列，我们将使用熊猫。分类函数并返回数字列，每个数字代表列中不同的值。

# XGBoost API

经过这几个简单的步骤，我们可以说我们已经准备好应用 XGBoost 及其 API 了。此外，我们将通过谷歌的 Collab 内置结构来比较它的速度和性能，以及 GPU 和常规 CPU。

首先，我们需要将表格数据转换成数据矩阵。

DMatrix 是 XGBoost 开发人员创建的一种数据结构，这也是 XGBoost 速度超快的原因之一。也就是说，它实现是特定的，且是为这个特定算法精心设计的。

之后，我们需要确定一些需要传递给模型的参数。XGBoost 有很多参数，但我们将重点关注几个简单但仍然重要的参数。

```
# The model's individual parametersgeneral_params = {"verbosity": 1,"booster": "gbtree"} #just to get familiar with the notationstree_params = {"eta": 0.1,"max_depth": 8,"min_child_weight": 1,"tree_method": "gpu_hist"}learning_task_parameters = {"objective": "reg:squarederror","eval_metric" : "rmse"}model_params = {**general_params, **tree_params, **learning_task_parameters}
```

第一个字典 *general_params* 由已经设置为这些值的默认参数组成。

第二个参数 tree_params 更重要。在其中，我们定义了算法的学习率，is max_depth，它的 min_child_weight(所有这些定义了模型的学习和它的最终结果，比如过拟合或欠拟合)以及“tree_method”。这一点很重要，因为在本文中，我们将 XGBoost 的性能与 GPU 和常规 CPU 进行比较。此参数可以留空，它将使用其“自动”默认值。

因此，接下来我们需要做的就是实例化一个与这个模型完全相同的模型，但是没有“tree_method”参数。

现在，我们只需要测试两者的性能。

# 使用 GPU

```
model = xgb.train(model_params, d_train, num_boost_round= num_boost_round, evals = [(d_test, "test")], early_stopping_rounds = 10)0]	test-rmse:260.538
Will train until test-rmse hasn't improved in 10 rounds.
[1]	test-rmse:252.566
[2]	test-rmse:246.073
[3]	test-rmse:240.912
[4]	test-rmse:236.7
[5]	test-rmse:232.965
[6]	test-rmse:230.032
[7]	test-rmse:227.433
[8]	test-rmse:225.417
[9]	test-rmse:223.836
[10]	test-rmse:222.405
[11]	test-rmse:221.197
[12]	test-rmse:220.515
[13]	test-rmse:219.941
[14]	test-rmse:219.523
[15]	test-rmse:219.903
[16]	test-rmse:219.719
[17]	test-rmse:219.644
[18]	test-rmse:219.46
[19]	test-rmse:219.341
[20]	test-rmse:219.396
[21]	test-rmse:219.492
[22]	test-rmse:219.109
[23]	test-rmse:219.02
[24]	test-rmse:219.108
[25]	test-rmse:218.846
[26]	test-rmse:218.614
[27]	test-rmse:218.538
[28]	test-rmse:218.545
[29]	test-rmse:218.546
[30]	test-rmse:218.606
[31]	test-rmse:218.779
[32]	test-rmse:218.611
[33]	test-rmse:218.729
[34]	test-rmse:218.807
[35]	test-rmse:218.746
[36]	test-rmse:218.826
[37]	test-rmse:219.02
Stopping. Best iteration:
[27]	test-rmse:218.538

CPU times: user 1.17 s, sys: 747 ms, total: 1.92 s
Wall time: 1.95 s
```

现在我们将使用 XGBoost 的 CV 内置函数，并计算它的性能。

```
%%timecv_results = xgb.cv(model_params, d_train, num_boost_round = num_boost_round, seed = 41, nfold = 10, metrics = {"rmse"}, early_stopping_rounds = 10)CPU times: user 10.3 s, sys: 6.07 s, total: 16.4 s
Wall time: 16.4 s
```

# 不带 GPU

我们将执行与之前相同的步骤。

```
%%timemodel = xgb.train(model_params, d_train, num_boost_round= num_boost_round, evals = [(d_test, "test")], early_stopping_rounds = 10)0]	test-rmse:260.663
Will train until test-rmse hasn't improved in 10 rounds.
[1]	test-rmse:252.803
[2]	test-rmse:246.229
[3]	test-rmse:241.064
[4]	test-rmse:236.891
[5]	test-rmse:233.325
[6]	test-rmse:230.704
[7]	test-rmse:228.544
[8]	test-rmse:226.999
[9]	test-rmse:225.487
[10]	test-rmse:224.685
[11]	test-rmse:224.073
[12]	test-rmse:223.353
[13]	test-rmse:222.934
[14]	test-rmse:222.601
[15]	test-rmse:222.435
[16]	test-rmse:222.544
[17]	test-rmse:222.532
[18]	test-rmse:222.712
[19]	test-rmse:222.187
[20]	test-rmse:222.263
[21]	test-rmse:221.973
[22]	test-rmse:221.898
[23]	test-rmse:221.9
[24]	test-rmse:221.668
[25]	test-rmse:221.591
[26]	test-rmse:221.584
[27]	test-rmse:221.649
[28]	test-rmse:221.907
[29]	test-rmse:221.829
[30]	test-rmse:222.147
[31]	test-rmse:221.988
[32]	test-rmse:221.957
[33]	test-rmse:221.249
[34]	test-rmse:221.144
[35]	test-rmse:221.161
[36]	test-rmse:221.081
[37]	test-rmse:221.175
[38]	test-rmse:221.169
[39]	test-rmse:221.251
[40]	test-rmse:221.247
[41]	test-rmse:221.255
[42]	test-rmse:221.246
[43]	test-rmse:221.293
[44]	test-rmse:221.149
[45]	test-rmse:220.766
[46]	test-rmse:220.965
[47]	test-rmse:221.119
[48]	test-rmse:221.181
[49]	test-rmse:221.177
[50]	test-rmse:221.272
[51]	test-rmse:221.318
[52]	test-rmse:221.366
[53]	test-rmse:220.959
[54]	test-rmse:220.975
[55]	test-rmse:220.934
Stopping. Best iteration:
[45]	test-rmse:220.766

CPU times: user 6.14 s, sys: 124 ms, total: 6.26 s
Wall time: 3.46 s
```

现在，通过交叉验证。

```
%%timecv_results = xgb.cv(model_params, d_train, num_boost_round = num_boost_round, seed = 41, nfold = 10, metrics = {"rmse"}, early_stopping_rounds = 10)CPU times: user 50.1 s, sys: 126 ms, total: 50.3 s
Wall time: 25.7 s
```