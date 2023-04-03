# Python 逻辑回归简介

> 原文：<https://medium.com/analytics-vidhya/introduction-to-logistic-regression-with-python-58cfca44bf48?source=collection_archive---------14----------------------->

## 预测信用卡申请的结果

![](img/1e63c7863dbcd359455af1bb13f5bbba.png)

信用卡申请

随着信用卡申请的大量增加，银行发现手动评估信用卡申请很麻烦。数据科学和机器学习的力量有助于银行预测申请是被接受还是被拒绝。写这篇文章是为了展示机器学习算法如何有助于自动评估任务，如信用卡申请。

对于这个项目，我们将使用来自 UCI ML 知识库的信用卡审批数据集，可以在这里找到。下载数据并保存到“数据集/”文件夹中

**先决条件:**

```
pip install pandas
pip install numpy
pip install sklearn
```

## **导入数据集**

```
**import pandas as pd
import matplotlib.pyplot as plt**credit_app = **pd.read_csv**("datasets/credit_application.csv", headers=None) 
```

## **了解我们的数据**

这个数据集是匿名的，但是我们可以从这篇[博客文章](http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html)中大致了解这些列的意思。典型的信用卡应用中的特征是`Gender`、`Age`、`Debt`、`Married`、`BankCustomer`、`EducationLevel`、`Ethnicity`、`YearsEmployed`、`PriorDefault`、`Employed`、`CreditScore`、`DriversLicense`、`Citizen`、`ZipCode`、`Income`以及最后的`ApprovalStatus`

我们可以计算我们的描述性统计数据，并观察到该数据集包含数字和分类特征，可以通过对数据进行一些处理来处理这些特征。

```
credit_app.**describe()**
credit_app.**info()**
```

## **数据清理**

我们的数据有三个关键问题需要解决

*   首先，我们有 float、int 和 object 类型的数值和分类数据。
*   其次，我们的数据中有多个数值范围，这使得我们无法从数据中做出准确的统计推断。
*   第三，我们必须估算所有缺失值，以确保我们的模型不会表现不佳。

我们可以看到缺失值被表示为“？”在我们的数据中，现在我们将使用均值插补来填充缺失的数值。

```
credit_app.**fillna**(credit_app.mean(), inplace=True)
# Printing the number of left over null values 
print(credit_app.**isnull()**)
```

我们观察到仍然有 120 个空值，并且这些都是非数字数据类型。我们可以通过选择每列最常见的循环值来处理这些值。

```
**for** val **in** credit_app:
    *# Check if the column is of object type*
    **if** credit_app[val].dtypes == 'object':
        *# Impute with the most frequent value*
        credit_app = credit_apps.fillna(credit_app[val].value_counts().index[0])
```

## **数据预处理**

在数据预处理阶段，我们将在拟合我们的模型之前完成某些任务

*   将所有非数字数据转换为数字数据。这个过程是使用一种称为标签编码的技术来完成的，这种技术有利于将标签转换成数字形式，从而提高机器的可读性。然后，机器学习算法可以以更好的方式决定这些标签必须如何操作。

```
**from** **sklearn.preprocessing** **import** LabelEncoder
*# Instantiate LabelEncoder*
le = LabelEncoder()

**for** val **in** credit_app:
    *# Compare if the dtype is object*
    **if** credit_app[val].dtypes=='object':
        credit_app[val]=le.fit_transform(credit_app[val])
```

*   接下来，我们将数据分成训练集和测试集。这个过程是为机器学习建模的两个不同阶段准备数据:训练和测试。理想情况下，来自测试数据的任何信息都不应用于调整训练数据，也不应用于指导机器学习模型的训练过程。我们将对该型号进行 70-30 分割。像`DriversLicense`和`ZipCode`这样的特征没有数据集中预测信用卡批准的其他特征重要。我们应该放弃它们，选择最好的功能集。这就是**特征选择**的过程。

```
**from** **sklearn.model_selection** **import** train_test_splitcredit_app = credit_app.drop([11, 13], axis=1)
credit_app = credit_app.values

*# Segregate features and labels into separate variables*
X,y = credit_app[:,0:13] , credit_app[:,13]

*# Split into train and test sets*
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.30,
                                random_state=42)
```

*   将我们的数据扩展到一个统一的范围。让我们用`CreditScore`作为缩放工作的真实例子。一个人的信用评分决定了他偿还信用卡账单的价值。这个数字越高，一个人被认为在财务上越值得信任。因此，1 的`CreditScore`是最高的，因为我们将把所有值缩放到 0-1 的范围。

```
**from** **sklearn.preprocessing** **import** MinMaxScaler
*# Instantiate MinMaxScaler and use it to rescale X_train and X_test*
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)
```

## **拟合模型并预测结果**

[根据 UCI](http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.names) ，数据集包含更多对应于“拒绝”状态的实例，而不是对应于“批准”状态的实例。具体来说，在 690 个案例中，有 383 个(55.5%)申请被拒绝，307 个(44.5%)申请获得批准。”为了建立一个好的模型，我们的结果应该在统计上与这些结果一致。

机器学习过程中最困难的部分是选择最适合我们业务问题的模型。我们必须就我们的数据提出问题，例如:*这些特征是否显示出彼此之间的线性关联？*在检查该数据的相关性时，我们可以观察到我们的特征是相关的，因此，我们可以选择广义线性模型来预测我们的结果变量。预测信用卡申请是否会被批准是一项分类任务，因此我们将选择使用逻辑回归模型。

```
**from** **sklearn.linear_model** **import** LogisticRegression
*# Instantiate a LogisticRegression classifier with default parameter values*
logreg = LogisticRegression()

*# Fit model to the train set*
logreg.fit(rescaledX_train, y_train)
```

Out []:

```
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='warn', n_jobs=None, penalty='l2',
random_state=None, solver='warn', tol=0.0001, verbose=0,warm_start=False)
```

现在，我们将在测试集上评估我们的模型的分类准确性。在预测信用卡申请的情况下，同样重要的是看看我们的机器学习模型是否能够预测那些最初被拒绝的申请的批准状态。如果我们的模型在这方面表现不好，那么它最终可能会批准本不应该批准的申请。使用混淆矩阵来计算我们的假阳性和假阴性是一个有用的任务，以找到我们的模型的错误分类率。

```
**from** **sklearn.metrics** **import** confusion_matrix
y_pred = logreg.predict(rescaledX_test)

print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))

*# Print the confusion matrix of the logreg model*
confusion_matrix(y_test, y_pred)
```

Out[]:

```
Accuracy of logistic regression classifier:  0.8377192982456141
[92, 11
 26, 99]
```

上面的输出显示了我们的准确度和混淆矩阵。我们可以看到我们的预测率相当不错，大约 84%！尽管如此，我们仍然可以使用其他技术来提高我们的模型精度。

## **通过超参数调整提高模型效率**

为了进一步增强我们的模型，我们可以使用一种众所周知的叫做网格搜索的技术来提高预测信用卡批准的能力。因此，超参数优化的目标是找到最小化模型验证误差函数的一组值。这一点非常重要，因为整个模型的性能取决于指定的超参数值。在本例中，我们将搜索参数

*   tol:停止标准的容差
*   max_iter:求解器收敛所需的最大迭代次数。

Scikit-Learn 为网格搜索的实现提供了很好的文档，如果你想了解更多，你可以在这里找到文档。

```
**from** **sklearn.model_selection** **import** GridSearchCV
*# Define the grid of values for tol and max_iter*
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

parameter_grid = dict(tol=tol, max_iter=max_iter)
```

现在，我们将开始网格搜索，看看哪些值表现最好。

我们将用我们拥有的所有数据用我们早期的模型实例化`GridSearchCV()`。我们将提供 x 和 y 的缩放版本。我们还将指示`GridSearchCV()`执行五重交叉验证。[在 K-Fold 交叉验证](https://www.cs.cmu.edu/~schneide/tut5/node42.html)中，数据集被分成 *k* 个子集，holdout 方法被重复 *k* 次。每一次， *k 个*子集中的一个作为测试集，其他 *k-1 个*子集放在一起形成一个训练集。然后计算所有 *k 次*试验的平均误差。

```
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

rescaledX = scaler.fit_transform(X)

grid_model_result = grid_model.fit(rescaledX, y)

*# Summarize results*
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: **%f** using **%s**" % (best_score, best_params))
```

Out[]:

```
Best: 0.853623 using {'max_iter': 100, 'tol': 0.01}
```

我们的模型成功地将其效率提高了 1%以上！网格搜索帮助我们完成了这项任务。我们完成了一些机器学习来预测一个人的信用卡申请是否会被批准，给定这个人的一些信息。

## 参考资料:

1.  [http://archive.ics.uci.edu/ml/datasets/credit+approval](http://archive.ics.uci.edu/ml/datasets/credit+approval)
2.  [https://sci kit-learn . org/stable/modules/generated/sk learn . linear _ model。LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
3.  【https://www.datacamp.com】T4/
4.  [https://www.cs.cmu.edu/~schneide/tut5/node42.html](https://www.cs.cmu.edu/~schneide/tut5/node42.html)

感谢你阅读我的文章，我希望你觉得有趣！

你可以在社交媒体上找到我！

> [领英](https://www.linkedin.com/in/aditya-nar/)
> 
> [Github](https://www.github.com/AdiNar1106)