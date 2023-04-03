# 模型选择和参数调整

> 原文：<https://medium.com/analytics-vidhya/model-selection-and-parameter-tuning-a78cdda122d?source=collection_archive---------12----------------------->

![](img/4f50d00be8acdbff31a0b733ab5127b5.png)

赫克托·j·里瓦斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

执行分类(或回归)任务时，数据科学工作流中最重要的步骤之一是为数据集选择最佳模型算法。假设数据集已经被充分清理，多重共线性已经减少(或通过主成分分析避免)，并且所有其他探索性数据分析(EDA)任务已经完成，可以开始建模过程。

挑选正确的模型没有什么神奇的诀窍，它完全取决于数据集本身。在确定要依赖的评分标准后，节省时间和避免无目的徘徊的第一步可能是尝试使用默认参数在循环的**中装配多个分类器。例如:**

```
# for acquiring and managing datasets
import pandas as pd
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')# for modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics# classifier modeling methods
import xgboost
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNBX_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=42)
classifiers = [KNeighborsClassifier, SVC, DecisionTreeClassifier,
               RandomForestClassifier, AdaBoostClassifier,
               XGBClassifier, LogisticRegression, GaussianNB]
classy_scores = []
for classifier in classifiers:
    clf = classifier()
    clf.fit(X_train, y_train.values.flatten())
    score = clf.score(X_test, y_test.values.flatten())
    classy_scores.append((str(classifier), score))
ranked_classifiers = sorted(classy_scores, key=lambda x: x[1], reverse=True)
ranked_classifiers   #outputs all classifiers, ranked by testing metric
```

虽然不精确，但这种方法至少会让您看到每种算法的相对效率，让您放弃那些分数明显较低的算法。(上面显示的只是一个例子，决不是算法可能性的详尽列表)。

一旦选择了一个(或两个，或三个)模型，就该开始参数调整过程了。这可能是一个漫长的努力，特别是在运行时间方面，因此必须了解如何处理这个过程。优化超参数的最好方法之一是使用 sklearn 的 GridSearchCV。使用网格搜索，只需输入要测试的参数值列表的字典，机器就会运行所有可能的参数组合。

注意:这是容易损失时间的地方。与其一次测试所有可能参数的大量参数值，不如分多个步骤小批量运行这些参数值更有效率。我建议一次最多测试两个或三个参数的 5 个可能值，最好每个步骤中的参数都是彼此最相关的。在第一轮之后，缩小相同参数值的分布范围(第二次时[1，2，3]变为[1.5，2.0，2.5])。或者，如果结果在测试列表的上端/下端，尝试在那个方向扩展列表(如果[1，2，3]返回' 3 '，下一次迭代测试[3，4，5])。

一旦你有了你的最佳值(不要被缩小到许多有效数字所迷惑)，继续下一批(希望是相关的)参数进行测试。这可能需要一段时间，可能会有点乏味。我最好的建议是*仔细阅读算法的文档，以确定哪些参数一起测试最有意义，哪些值测试最有意义。例如，如果您正在测试 sklearn 的 SVC 算法，并希望比较不同的内核(如“线性”、“rbf”、“poly”…)，则在没有嵌套参数字典的情况下，您无法同时测试其他内核特定的参数(如“degree”仅适用于“poly”内核)；这是可能的，但会大大增加运行时间。相反，只需首先比较内核本身，然后进一步调优看起来工作得最好的那个。*

*例如，我最近运行了一个分类任务，使用准确性作为期望的评分标准。为了节省时间(运行时间和编写重复代码的时间),我编写了几个函数来自动化网格搜索参数选项的过程，找到产生最佳测试精度的组合(与低过度拟合相平衡),然后为下一轮参数调整(不同参数)填充这些参数。重要的是，该过程使用支持测试数据进行验证，并且平衡过拟合(在调整超参数时通常会出现这种情况)。*

*为了实现这种平衡，我创建了一个度量标准，它是测试准确性的加权调和平均值，以及训练和测试准确性之间的差距(我对模型过度拟合的度量)。使用这个调和平均值而不仅仅是测试精度，会产生稍微低一点的测试精度分数，但是会带来显著减少模型过度拟合的好处。为了使每一次迭代可视化，我还在维持测试数据中加入了模型成功的注释混乱矩阵图。代码如下:*

```
*# define a function to generate a confusion matrix
def confu_matrix(y_pred, x_tst, y_tst):
    import warnings
    warnings.filterwarnings('ignore')
    y_pred = np.array(y_pred).flatten()
    y_tst = np.array(y_tst).flatten()
    cm = confusion_matrix(y_tst.flatten(), y_pred.flatten())
    sns.heatmap(cm, annot=True, fmt='0g', 
                annot_kws={'size':14, 'ha':'center', 'va':'top'})
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='0.01%', 
                annot_kws={'size':14, 'ha':'center', 'va':'bottom'})
    plt.title('Confusion Matrix', fontsize=14)
    plt.show();

def convert_params(best_params):
    params = {}
    for key, val in best_params.items():
        params[key] = [val]
    return params

def get_best_params(cv_results):
    """
    input:     model.cv_results_
    returns:   dictionary of parameters with the highest harmonic 
    mean balancing mean_test_score and (1 - test_train_diff)
    This reduces overfitting while maximizing test score.
    """
    dfp = pd.DataFrame(cv_results)
    dfp['test_train_diff'] = np.abs(dfp['mean_train_score'] - dfp['mean_test_score'])
    dfp['harmonic'] = 3 / ((2 / dfp['mean_test_score']) + (1 / (1-dfp['test_train_diff'])))
    dfp.sort_values(by='harmonic', ascending=False, inplace=True)
    dfp.reset_index(drop=True, inplace=True)
    return convert_params(dfp.iloc[0].params)

def gridsearch_params(estimator, params_test, old_params=None, 
                      update_params=True, scoring='accuracy'):
    """
    Inputs an instantiated estimator and a dictionary of parameters
    for tuning (optionally an old dictionary of established parameters)
    Returns a dictionary of the new best parameters.
    Requires X_train, X_test, y_train, y_test to exist as global variables.
    """
    import warnings
    warnings.filterwarnings('ignore')
    if update_params:
        old_params.update(params_test)
        params_test = old_params
    gsearch1 = GridSearchCV(estimator=estimator, refit=True,
                            param_grid=params_test, scoring=scoring,
                            n_jobs=4, iid=False, cv=5)
    gsearch1.fit(X_train, y_train.values.flatten())
    best_params = get_best_params(gsearch1.cv_results_)
    gsearch1a = GridSearchCV(estimator=estimator, refit=True,
                             param_grid=best_params, scoring=scoring,
                             n_jobs=4, iid=False, cv=5)
    gsearch1a.fit(X_train, y_train.values.flatten())
    confu_matrix(gsearch1a.predict(X_test), X_test, y_test)
    tr_acc = round(accuracy_score(y_train.values.flatten(),
                                  gsearch1a.predict(X_train)), 4)*100
    tst_acc = round(accuracy_score(y_test.values.flatten(),
                                   gsearch1a.predict(X_test)), 4)*100
    print(f"Train accuracy: {tr_acc}%\nTest accuracy: {tst_acc}%\n{best_params}")
    return best_params, gsearch1a

# First set of parameters 
param_test1 = {'max_depth':range(3,8),
               'min_child_weight':range(1,6)}
xgb1a = XGBClassifier(learning_rate=0.1, n_estimators=1000,
                      objective='binary:logistic', nthread=4, seed=42)
best_params, xgb_gs1 = gridsearch_params(xgb1a, param_test1,
                                         update_params=False)

# second set of parameters, including best params from first set
param_test2 = {'gamma': np.linspace(0.0, 0.2, 5),
               'subsample':np.linspace(.8, 1.0, 5),
               'colsample_bytree':np.linspace(.8, 1.0, 5)}
xgb1a = XGBClassifier(learning_rate=0.1, n_estimators=1000,
                      objective='binary:logistic', nthread=4, seed=42)
best_params, xgb_gs2 = gridsearch_params(xgb1a, param_test2, best_params,update_params=True)*
```