# 优化 Sklearn 集成(具有均匀权重)

> 原文：<https://medium.com/analytics-vidhya/optimizing-sklearn-ensembles-with-homogeneous-weights-2db698a72b7a?source=collection_archive---------8----------------------->

每个人都知道 Python 的 Scikit-Learn 包的不可思议的潜力。在它的资源中，有一种回归器通过回归器委员会的投票(平均)来起作用。那就是[投票回归量](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)。

在这篇短文中，我将展示一种选择最佳回归变量的方法，将委员会或回归变量与遗传算法相结合。

第一，遗传算法会优化什么？—回归变量集中每个回归变量的参数，然后它将寻找最佳回归变量集来整合委员会。所以让我们开始吧。

我将制作一个名为**的 Python 类 ensemble_search。**

```
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR 
from sklearn.ensemble import AdaBoostRegressor as ADA
from sklearn.ensemble import BaggingRegressor as BAG
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import RANSACRegressor as RAN
from sklearn.linear_model import PassiveAggressiveRegressor as PAR
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.datasets import make_regression
from sklearn import preprocessing
```

以上是将要使用的包和回归模型。回归模型越多越好。

该类将使用您的数据的训练/测试部分、您的群体大小以及您希望它运行的最大时期数进行初始化。

下面显示了该类的完整代码以及如何在一些回归数据上运行它的示例。每种方法都在后面解释。这里是 github 中[代码](https://github.com/hugoabreu1002/Optimizaded_Ensemble_Sklean)的链接。

密码

**gen_population** 如其所言，用于生成人口，这是一个列表的列表。填充列表中的每个列表都由另一个列表组成，现在是 regeressors 名称、对象及其参数(这是一个字典)。群体(列表)中的个体(列表)是许多模型的串联，这些模型在 for 循环中进一步参数化。模型可以重复。

群体中的每个个体都用一些随机参数初始化。回归量的数量也是一个需要优化的参数。

**set_fitness** 方法用于为群体中的每个个体赋予一个适应值，这里使用的是预测回归和真实 y set (y_test)之间的 MAE(平均绝对误差)。

**next_population** 方法的作用是对种群执行交叉。在调用这个方法之前，群体必须从最好的个体到最差的个体进行排序。然后，该方法将每个回归变量的参数混合到群体的个体中。

**early_stop** 是在算法完成所有历元之前停止算法，只有在历元上的适应度达到最小值时才停止算法。最后 2 个二阶导数的平均值大于零，而一阶导数的平均值接近零。

**search_best** 是一种将所有东西粘在一起的方法，生成给定大小的种群，为种群中的每个个体设置适应度，对个体进行排序以了解最佳和最差，并执行交叉。所有这些步骤都在一个循环中完成，而没有达到最大次数或停止条件。

最后，为了检验算法是否有效，我们对 make_regression 样本进行了一次测试。