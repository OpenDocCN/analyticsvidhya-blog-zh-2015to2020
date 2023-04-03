# 用最大似然法中的投票分类器增强性能度量

> 原文：<https://medium.com/analytics-vidhya/voting-classifier-in-machine-learning-9534504eba39?source=collection_archive---------5----------------------->

假设你是决策小组的一员，小组的每个成员都有一个关于某事的决定。专家组在他们之间进行投票，并做出最终决定。他们是怎么做到的？简单来说，他们采用全体成员投票的方式。

![](img/c56cdd850ac91d4c9b5b30e78123892c.png)

小组讨论和投票。

你可以用机器学习分类问题来做同样的事情。假设你已经训练了几个分类器，如逻辑回归分类器，SVC 分类器，决策树分类器，随机森林分类器，也许还有几个，每个分类器的准确率都在 85%左右。

类似地，机器学习分类我们也可以使用小组投票法。换句话说，创建一个均匀的分类器的一个非常简单的方法是聚合每个分类器的预测，并预测投票最多的类。这种多数投票分类器被称为硬投票分类器。

有点令人惊讶的是，这种投票分类器通常比集成中的最佳分类器获得更高的准确度。事实上，即使每个分类器都是弱学习器(意味着它只比随机猜测稍好)，集成仍然可以是强学习器(实现高精度)，只要有足够数量的弱学习器并且它们足够多样化。获得不同分类器的一种方法是使用非常不同的算法来训练它们。这增加了他们犯不同类型错误的机会，提高了整体的准确性。

下面的代码在 Scikit-Learn 中创建和训练了一个投票分类器。

```
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVClog_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()voting_clf = VotingClassifier( estimators=[(‘lr’, log_clf), (‘rf’, rnd_clf),
 (‘svc’, svm_clf)], voting=’hard’ )
voting_clf.fit(X_train, y_train)
```

让我们看看每个分类器在测试集上的准确性:

```
from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

***逻辑回归 0.864
随机森林分类器 0.872
SVC 0.888
投票分类器 0.896***

你有它！投票分类器略微优于所有单个分类器。如果所有分类器都能够估计类概率(即，它们有一个 pre dict_proba()方法)，那么您可以告诉 Scikit-Learn 预测具有最高类概率的类，在所有单个分类器上平均。这被称为软投票。它通常比硬投票获得更高的性能，因为它给予高度自信的投票更多的权重。你所需要做的就是用 voting="soft "替换 voting="hard "并确保所有的分类器都能估计类概率。SVC 类默认情况下不是这样，所以需要将其概率超参数设置为 True(这将使 SVC 类使用交叉验证来估计类概率，减慢训练，并且它将添加一个 predict_proba()方法)。如果您修改前面的代码以使用软投票，您会发现投票分类器达到了 91%以上的准确率！