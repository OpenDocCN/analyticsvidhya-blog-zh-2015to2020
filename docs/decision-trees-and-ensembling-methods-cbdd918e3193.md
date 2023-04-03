# 决策树和集成方法

> 原文：<https://medium.com/analytics-vidhya/decision-trees-and-ensembling-methods-cbdd918e3193?source=collection_archive---------14----------------------->

![](img/5d7f894d59c3023e8f2fef75d5b2cd8d.png)

学分: [www.forestryengland.uk](http://www.forestryengland.uk)

决策树还是传统模型？

如果预测值和目标变量之间的关系是近似线性的，那么传统方法可能会在数据上表现得更好，并且优于各种树模型。

另一方面，如果模型非常复杂并且遵循非线性模式，决策树可能会优于传统方法。此外，在项目需要更多的可解释性和可视化的情况下，决策树可能是首选，因为这些可以在“流程图”类型的图表中表示，并且比较容易解释。

**决策树的优势:**

*   比大多数模型更容易解释，包括线性回归
*   可以以“流程”或“组织”图的形式可视化，这使得它易于理解，即使对于外行人也是如此
*   决策树倾向于更好地模拟人类实际思考的方式
*   无需创建虚拟变量即可处理定性变量的能力

**决策树的缺点:**

*   不具有与其他回归和分类模型(如线性回归和逻辑回归)相同的预测准确性
*   不太容易概括，数据中的一个小变化就可能破坏最终的模型

考虑到上面提到的缺点，还有其他技术可以应用于决策树，从而大大提高它们的预测能力。这些是装袋，随机森林和助推。

**装袋:**

Bagging 或 Bootstrap Aggregation 正在生成数量为 ***n*** 的决策树，并用数据集的 ***n*** 个样本对它们进行训练(样本应该从单个数据集中随机抽取。带替换！).

虽然这种方法可以改进许多估计器的结果，但它对决策树特别有用。由于它们的高方差，对一组观察值进行平均已被证明可以减少方差而不影响偏倚！

就像所谓的“群体智慧”一样，虽然有些人低估了，有些人高估了，但当我们平均这些猜测时，误差的总和往往会减少。

虽然 Bagging 是一种提高决策树预测准确性的好方法，但它也带来了负面影响，即我们失去了模型的可解释性。使用单个树，很容易解释和可视化模型中使用的最佳特征，但是在对数百或数千个特征进行平均后，我们放弃了模型的可解释性…

**随机森林:**

与袋装决策树非常相似，随机森林由一组袋装决策树组成，但稍加调整，结果就会大不相同。

不是在同一个树上测试自举训练样本，而是在决策树的每个分裂处选择特征(预测器)的随机样本。这保证了被测试的模型彼此不同，因为它们不会总是使用最重要的特征来进行树的第一次分裂。

因此，不同的模型彼此之间的相关性较低，一旦取平均值，就会产生更有影响的变化，从而产生变化较小的模型。

袋装决策树和随机森林的主要区别在于从总共***【p】***个预测器 ***中选择一个预测器子集大小***【m】***。*** 通常对于随机森林，我们使用 **m** = **√p** ，这样可以减少预测误差。当***m***=***p***我们有一个常规的袋装决策树模型。

**增压:**

Boosting 也以类似的方式工作，但树是通过使用来自先前树的预测误差信息来顺序生长的。模型适合数据的修改版本，随着数据的迭代，每棵树被迫专注于那些错误或误分类。

根据树的性能，赋予它权重以指示其性能。权重将在最后用于平均结果并得出最佳模型。

在 Python 的 Scikit-Learn 上实现不同的技术:

```
# Import Models from Sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier# Set Features and Target Variable
X, y = make_classification(n_samples=750, n_features=20, random_state=42)# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    stratify=y)# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt.score(X_test, y_test)# OUTPUT
0.8670212765957447# Bagging Classifier
bag = BaggingClassifier(n_estimators=100, random_state=42)
bag.fit(X_train, y_train)
bag.score(X_test, y_test)# OUTPUT
0.898936170212766# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)# OUTPUT
0.898936170212766# Adaptative Boost Classifier
ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
print(ada.score(X_test, y_test))# OUTPUT
0.8457446808510638#Gradient Boost Classifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb.score(X_test, y_test)# OUTPUT
0.9095744680851063
```