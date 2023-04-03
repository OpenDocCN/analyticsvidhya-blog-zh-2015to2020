# 特征工程——如何找到特征重要性分数(增长黑客)

> 原文：<https://medium.com/analytics-vidhya/feature-engineering-how-to-find-feature-importance-score-of-all-features-and-then-build-model-a8508a3edb33?source=collection_archive---------11----------------------->

## 或者特征 imp。IrisDataset 上的分数

特征工程是一个非常容易被误解的术语，当我第一次开始制作模型时，我也对它感到困惑，并认为这意味着在决定构建哪些特征和跳过哪些特征时的一种方法或一系列实践或工具。

迷茫？

但随着我更多地围绕它工作，并开始处理真实世界的数据点，我意识到决定选择什么特征，然后对它们进行特征工程是一个“根据数据集决定”的问题，而不仅仅是缩放特征集或在其上运行一些功能。

有时，您将两个特征组合起来，使新特征(如每户的房间数和每邻居的房间数)成为每邻居的房间数，有时，您从另一个特征中减去一个特征，而其他时候，您只从 30 个特征中选择最好的 8 个，因为其他 22 个特征对您的决策影响不到 0.01%。

有人称之为艺术多于科学，我认为他们是对的。

特征工程是一个多样化的大得多的主题，但是有一个小而简洁的技巧是我在处理一些模型时发现的，并且发现它对我试图构建的模型非常有用。

# 这是我的诀窍

每当我在分类器上运行模型并训练我的分类器时，我通常更喜欢将随机森林作为首选分类器，不仅因为它是透明的，可以向我显示机器是如何破坏我的模型的。(见下文)

[](/@madanflies/decisiontree-classifier-working-on-moons-dataset-using-gridsearchcv-to-find-best-hyperparameters-ede24a06b489) [## 决策树分类器—使用 GridSearchCV 处理 Moons 数据集，以找到最佳超参数

### 决策树是一种很好的分类方法，不像随机森林，它们是透明的或白盒…

medium.com](/@madanflies/decisiontree-classifier-working-on-moons-dataset-using-gridsearchcv-to-find-best-hyperparameters-ede24a06b489) 

还因为我可以检查整个模型的特征重要性分数，我可以选择几个模型，然后计算它们的重要性分数，以找到机器实际挖掘的内容。

例如，如果我正在对住房数据集进行要素重要性评分，我的分数将为

```
Rooms (unit) 0.09805484876235299
Neighbourhood (km) 0.021686162123673226
Median income ($) 0.44874930874833113
Garden area (cm) 0.43150968036564236
```

这代表了

```
Rooms 9.8%
Neighbourhood 2.1%
Median income 44.8%
Garden area (cm) 43.1%
```

这说明中值收入和花园面积是最重要的特征，而邻域是最弱的特征类。

## 我是怎么做到的？

让我展示给你看。

在你训练了你的随机森林分类器或回归器之后

```
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
```

你已经在这方面训练了你的数据

```
rnd_clf.fit(X_train,y_train)
```

您可以运行 feature_importances_ like

```
for score in rnd_clf.feature_importances_: 
     print score
```

对于 Iris 数据集，使用上述函数得到的特征重要性分数为

```
sepal length (cm) 0.09805484876235299
sepal width (cm) 0.021686162123673226
petal length (cm) 0.44874930874833113
petal width (cm) 0.43150968036564236
```

如果您想自己运行代码，只需将以下代码复制粘贴到您的笔记本中-

```
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifieriris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)X=iris[‘data’]
y=iris[‘target’]
rnd_clf.fit(X,y)Feature_score=zip(iris[“feature_names”], rnd_clf.feature_importances_)for name, score in Feature_score:
 print(name, score)
```

瞧，你有一个所有特征分数的列表。

> 诀窍在于，低百分比特性没有价值，不值得花时间进行清理、缩放和标准化，因此，您可以将它们组合在一起，形成一个新特性，然后再次找到特性重要性分数，或者您可以完全忽略它们。

一旦您有较少的功能需要处理，您就可以获得更好的模型分数，并且总体上构建更具可伸缩性的模型。

您可以对影像以及 MNIST 或类似数据集进行同样的操作。

如果你正在寻找一本特性工程指南，我强烈推荐 Aurelion Geron，因为与整个主题相比，这篇文章只是品酒。

希望这个技巧能帮助你更快地做出更高可信度的模型。

鳍。