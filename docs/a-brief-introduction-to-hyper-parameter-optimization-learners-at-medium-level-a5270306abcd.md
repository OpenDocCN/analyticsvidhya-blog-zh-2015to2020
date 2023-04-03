# 超参数优化简介(中等水平的学习者)

> 原文：<https://medium.com/analytics-vidhya/a-brief-introduction-to-hyper-parameter-optimization-learners-at-medium-level-a5270306abcd?source=collection_archive---------16----------------------->

![](img/8d4b4ac8173afcc7a76f6f3e04944390.png)

**来自谷歌图片**

首先让我们了解什么是超参数:-

**超参数**是算法的参数，其值与模型的行为和性能成正比。

不要混淆超参数和模型参数之间的区别。

**模型参数**是模型内部的参数，这些内部的东西(参数)可以从给定的数据(我们可以说是训练数据)中计算出来。

当你看到这两件事的一些例子时，你会更清楚这种区别，

所以，在这里我将通过考虑**线性回归**、**随机森林**、 **K-Means 聚类**算法来展示例子(假设你使用过这些算法)。

# 超参数:-

**线性回归:-**

在线性回归中，拟合截距是一个超参数，

也就是说，

fit_intercept =在函数形式中是否包含β项。

**随机森林:-**

在随机森林中，超参数是

1.  n _ 估计量
2.  标准

这解释了，

n_estimators =森林中的树木数量，

criterion =“Gini”或“信息增益”(假设您使用了这些算法)。

**K 均值聚类:-**

在 K 均值中，超参数为

1.  初始化

也就是说，

init =质心的初始化方法。

# 模型参数:-

**线性回归:-**

在线性回归中，当我们用训练数据训练模型时，将形成最佳拟合线，

y =α+β* X

其中 X =输入数据，

Y =标记数据(训练时)

如果你想了解更多关于线性回归的知识，请点击链接。

这里β是模型参数之一。

不幸的是，**随机森林**和 **K-Means 聚类**没有模型参数可以向你解释😢。

对于**支持向量机**，支持向量是模型参数。

所以现在，我希望你对模型参数和超参数之间的区别有所了解。

现在让我们进入主要部分，

1.  **一个算法的超参数优化有什么用？**

a)正如我们上面讨论的，超参数是对算法性能有很大控制的参数。因此，通过优化超参数，使我们的数据以优化的方式适合算法，从而提高算法的精度。

我们可以将**超参数优化**定义为“将超参数的优化值分配给算法以获得更高精度的过程”。

2.**我们如何将超参数优化应用到算法中？**

a)这里我们将讨论一些最流行的超参数优化技术，即随机搜索、网格搜索。

a.**随机搜索:-**

在这种技术中，随机搜索将建立超参数值的网格，并选择该超参数值的随机组合来训练和评分模型。

只需浏览下面的代码片段就能理解**随机**搜索:-

```
from sklearn.model_selection import RandomizedSearchCV
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform, truncnorm, randint# get iris data
iris = datasets.load_iris()
X = iris.data
y = iris.target
model_params = {
    'n_estimators': randint(4,200),
    'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    'min_samples_split': uniform(0.01, 0.199)
}
rf_model = RandomForestClassifier()
clf = RandomizedSearchCV(rf_model, model_params, n_iter=100, cv=5, random_state=1)
model = clf.fit(X, y)
from pprint import pprint
pprint(model.best_estimator_.get_params())
```

不要为代码而烦恼，比如“那是什么东西？”诸如此类的事情。因为所有这些东西，你都会在练习的时候得到。

**网格搜索:-**

在这种技术中，网格搜索将用超参数值的所有可能组合来训练算法。

在这里，我们可以通过使用**交叉验证**技术来衡量性能。

使用 K **实现交叉验证的最佳方法之一——折叠交叉验证**

这项技术使我们能够确保训练好的模型是由数据集中数据的所有模式和行为训练出来的。

只需浏览下面的代码片段就可以理解网格搜索:-

```
import pandas as pd 
import numpy as np
dataset = pd.read_csv(r"D:/Datasets/winequality-red.csv", sep=';')
  X = dataset.iloc[:, 0:11].values 
  y = dataset.iloc[:, 11].values
  from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=0)
  from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators=300, random_state=0)
  grid_param = {     
'n_estimators': [100, 300, 500, 800, 1000],     
'criterion': ['gini', 'entropy'],     
}
  gd_sr = GridSearchCV(
estimator=classifier,                      
param_grid=grid_param,                      
scoring='accuracy',                      
cv=5)
gd_sr.fit(X_train, y_train)best_parameters = gd_sr.best_params_ 
print(best_parameters)
best_result = gd_sr.best_score_ 
print(best_result)
```

我再说一遍，先不要纠结代码，先试着学习核心概念。在编码的时候，你会很容易熟悉这些东西。

# 一些重要注意事项:-

→如果你对自己的计算能力和成本没问题，那就去网格搜索吧。因为网格搜索比随机搜索需要更多的计算能力。

随机搜索将只选择超参数值的随机组合，而网格搜索将通过包括所有可能的组合来选择最佳超参数值。这就是为什么网格搜索比随机搜索需要更多的计算能力。

→尽管网格搜索需要更多的计算能力，但它会给出比随机搜索更准确的结果

→但是随机搜索会以较低的计算能力和成本更好地给出准确的结果。

## 快乐阅读！✌✌