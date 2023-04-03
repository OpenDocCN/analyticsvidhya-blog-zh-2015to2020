# Scikit 简介-学习管道

> 原文：<https://medium.com/analytics-vidhya/scikit-learn-pipeline-d43c80559257?source=collection_archive---------4----------------------->

![](img/dbe3e8f43c280a78575ccda24c167cb3.png)

假设您是一名机器学习工程师，您被银行雇佣来创建一个机器学习算法，以确定欺诈性交易，从而避免客户损失金钱。我猜你会尝试不同的机器学习算法，比如支持向量机、梯度提升树和逻辑回归，然后你会想评估哪种算法最适合你的数据集。

在正常情况下，您会创建估计器并将其分别拟合到您的数据集，这既单调又令人厌倦，但如果存在一种解决方案，您只需将所有机器学习算法拟合到您的数据集一次，而无需逐一迭代拟合每个估计器的过程，会怎么样呢？这就是 Scikit-Learn 的管道类发挥作用的地方。

流水线类顺序地应用列表变换和最终估计器。管道的中间步骤必须是转换，也就是说，它们必须实现 fit 和 transform 方法。

管道的目的是组装可以在设置不同参数时一起交叉验证的步骤。

回到我们根据上面的管道定义为银行开发的机器学习应用程序。我们将使用的数据集在输入到我们的机器学习模型之前需要进行预处理，以便我们可以获得最佳结果。pipeline 类将允许我们应用转换方法，如用于缩放数据的标准 scaler 和其他 sklearn 类，如 gridsearch 和 k-fold。

让我们分解使用 python 实现 pipeline 类的代码。github 回购的链接在这里是[](https://github.com/isheunesutembo/Scikit-Learn-Pipelines/blob/master/SkLearn%20Pipelines.ipynb)**。**

*我在 youtube 上的视频[在这里。](http://youtube.com/watch?v=GmuUpLtMxSU)*

*我们将使用从 scikit-learn 导入的 iris 数据集。*

```
***from** **sklearn.datasets** **import** load_iris
**from** **sklearn.model_selection** **import** train_test_split
**from** **sklearn.preprocessing** **import** StandardScaler
**from** **sklearn.decomposition** **import** PCA
**from** **sklearn.pipeline** **import** Pipeline
**from** **sklearn.externals** **import** joblib
**from** **sklearn.linear_model** **import** LogisticRegression
**from** **sklearn.ensemble** **import** AdaBoostClassifier,GradientBoostingClassifier
**from** **sklearn.neighbors** **import** KNeighborsClassifier
**from** **sklearn** **import** svm
**from** **sklearn** **import** treeiris=load_iris()*
```

*然后，我们将变量 iris 设置为来自 sklearn 的 load_iris()的实例。*

```
*X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=42)*
```

*我们将数据集分为训练集和测试集，我们将 20%的数据用于测试集，我们将使用测试集来查看我们的机器学习模型的执行情况。我们将我们的特征传入 train_test-split()方法，我们的特征被解析为 iris.data，我们的目标值或标签被解析为 iris.target 传入该方法。*

```
*pipe_lr=Pipeline([('scl',StandardScaler()),
                 ('pca',PCA(n_components=2)),
                 ('clf',LogisticRegression(random_state=42))])

pipe_svm=Pipeline([('scl',StandardScaler()),
                  ('pca',PCA(n_components=2)),
                  ('clf',svm.SVC(random_state=42))])

pipe_dt=Pipeline([('scl',StandardScaler()),
                 ('pca',PCA(n_components=2)),
                 ('clf',tree.DecisionTreeClassifier(random_state=42))])

pipe_adaboost=Pipeline([('scl',StandardScaler()),
                       ('pca',PCA(n_components=2)),
                       ('clf',AdaBoostClassifier())])

pipe_gradientboosting=Pipeline([('scl',StandardScaler()),
                       ('pca',PCA(n_components=2)),
                       ('clf',GradientBoostingClassifier())])

pipe_knn=Pipeline([('scl',StandardScaler()),
                  ('pca',PCA(n_components=2)),
                  ('clf',KNeighborsClassifier(n_neighbors=3))])*
```

*现在是时候通过创建一个步骤列表来创建我们的管道了。首先，我们为逻辑回归创建管道步骤，它是变量 pipe_lr。然后，我们将 pipe_lr 变量设置为 pipeline 类的实例，即 Pipeline()。我们在一个元组中解析管道类的实例，该元组应该包含估计器或转换器的名称以及转换器或估计器的实例。例如，对于逻辑回归，我们将标准 Scaler 转换器的名称解析为“scl ”,将转换器的实例解析为 Standard Scaler()。我们对支持向量机和决策树等其他算法也做了同样的工作，如上面的代码所示。*

```
**#List of pipelines for ease of iteration*
pipelines=[pipe_lr,pipe_svm,pipe_dt,pipe_adaboost,pipe_gradientboosting,pipe_knn]*
```

*然后，我们创建一个管道实例的列表，这样我们可以更容易地遍历它们。*

```
**#Dictionery of pipelines and classifier types for ease of reference*
pipe_dict={0:'LogisticRegression',1:'Support Vector Machine',2:'Decision tree',3:'AdaBoostClassifier',4:'GradientBoosting',5:'KNearestNeighbors'}*
```

*然后，我们创建一个分类器或估计器的字典，以便我们更容易引用它们。*

```
**#fit the pipelines*
**for** pipe **in** pipelines:
    pipe.fit(X_train,y_train)*
```

*然后，我们遍历上面创建的管道实例的列表或数组，遍历它们，并对每个分类器拟合我们的训练集。正如你所看到的，管道使我们的工作变得更容易，只需将多个分类器拟合到训练集，不像如果我们没有管道，我们会将每个估计器分别拟合到我们的数据集，这是低效的。*

```
**#compare accuracies*
**for** idx,val **in** enumerate(pipelines):
    print('**%s** pipeline test accuracy: **%.3f**' %(pipe_dict[idx],val.score(X_test,y_test)))*
```

*然后，我们比较每个估计量的准确性。*

**#识别测试数据上最准确的模型*best _ ACC = 0.0 best _ clf = 0.0 best _ pipe = ' '**for**idx，val **in** enumerate(管道): **if** val.score(X_test，y _ test)>best _ ACC:best _ ACC = val . score(X _ test，y _ test)best _ pipe = val best _ clf = idx print('具有最佳精度的分类器: **%s***

*然后，我们在测试数据上确定最准确的模型，这些数据是我们的模型没有训练过的数据，这样我们就可以评估我们的模型是否概括得很好。*

*感谢阅读我的文章，如果你觉得有用，请鼓掌。*

*你也可以跟着我*

*推特:[https://twitter.com/IsheunesuTembo](https://twitter.com/IsheunesuTembo)*

*Instagram:[https://www.instagram.com/machine_learning_engineer/](https://www.instagram.com/machine_learning_engineer/)*