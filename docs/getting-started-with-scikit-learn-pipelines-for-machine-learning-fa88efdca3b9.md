# scikit 入门-用于机器学习的学习管道

> 原文：<https://medium.com/analytics-vidhya/getting-started-with-scikit-learn-pipelines-for-machine-learning-fa88efdca3b9?source=collection_archive---------9----------------------->

## 从头开始建造管道

![](img/971ec97e4c5efd943eb74401652fdc36.png)

图片来源:[https://www.wocintechchat.com/blog/wocintechphotos](https://www.wocintechchat.com/blog/wocintechphotos)

(本帖中的所有代码也包含在[这个 GitHub 库](https://github.com/hoffm386/simple-sklearn-pipeline-example)中。)

# 为什么要使用管道？

使用 [scikit-learn](https://scikit-learn.org/stable/index.html) 的典型整体机器学习工作流程如下所示:

1.  将所有数据载入`X`和`y`
2.  使用`X`和`y`进行[试运行分割](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)，生成`X_train`、`X_test`、`y_train`和`y_test`
3.  在`X_train`上安装[标准定标器](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)和[简单估算器](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)等预处理程序
4.  使用安装的预处理程序转换`X_train`，并执行任何其他预处理步骤(如删除列)
5.  创建各种模型，调整超参数，并选择一个适合预处理`X_train`和`y_train`的最终模型
6.  使用安装的预处理程序转换`X_test`，并执行任何其他预处理步骤(如删除列)
7.  在预处理的`X_test`和`y_test`上评估最终模型

下面是遵循这些步骤的示例代码片段，使用了来自[统计教科书](https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html)的[羚羊数据集](https://raw.githubusercontent.com/hoffm386/simple-sklearn-pipeline-example/master/antelope.csv)(“羚羊. csv”)。目标是根据成年羚羊数量、年降雨量和冬季严寒程度来预测春季小鹿的数量。这是一个非常小的数据集，应该仅用于示例目的！此示例跳过任何超参数调整，并在对预处理的测试数据进行评估之前，简单地对预处理的训练数据拟合普通的线性回归模型。

没有管道的例子

训练测试分割是机器学习工作流程中最重要的组成部分之一。它有助于数据科学家理解模型性能，尤其是在过度拟合方面。适当的训练-测试分离意味着我们必须对训练数据和测试数据分别执行预处理步骤，因此不会有信息从测试集“泄漏”到训练集中。

但是作为一个软件开发人员来看这段代码，一个问题立刻凸显出来:第 4 步和第 6 步实际上是一样的。到底干嘛了(不要重复自己)？！解决方案:管道。管道的设计完全避免了这个问题。您只需声明一次预处理步骤，然后就可以根据需要将它们应用于`X_train`和`X_test`。

# 首先，编写没有管道的代码

是的，你没看错。在您真正成为使用管道的专家之前，最好先写出重复/冗余版本的代码，然后重构代码以使用管道。如果您希望编写功能性管道代码，那么先返回并创建类似上面代码片段的东西！

# 第二，反复添加预处理步骤

管道产生的错误消息可能非常难以破译！因此，如果您一次添加多个步骤，并且出现问题，您将很难找出问题所在。一个更好的计划是一次添加一个步骤，并在进行过程中仔细检查它是否仍然有效。

我的一般策略是从任何有依赖关系的步骤开始，例如一个简单的估算器(因为如果有丢失的数据，其他预处理步骤可能会失败)。在这个示例中，让我们从 OneHotEncoder 开始。

让我们放大一些细节。首先，**装配**(ML 过程中的#3)。旧版本是:

```
ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")ohe.fit(X_train[["winter_severity_index"]])
```

新版本是:

```
pipe = Pipeline(steps=[
    ("encode_winter", ColumnTransformer(transformers=[           
        ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore"), ["winter_severity_index"])
    ], remainder="passthrough"
))])
pipe.fit(X_train, y_train)
```

我们仍然有相同的编码器和相同的参数，但是现在它嵌套在 ColumnTransformer 中，column transformer 嵌套在管道中。我们没有用子集化`X_train`(用`[[]]`)来指定哪些列要进行一键编码，而是将它传递给 ColumnTransformer。(参见[这篇来自我以前的学生 Allison Honold 的文章](https://towardsdatascience.com/using-columntransformer-to-combine-data-processing-steps-af383f7d5260)，了解更多关于 ColumnTransformers 的细节。)然后，我们不直接使用编码器，而是将其添加为流水线的第一个“步骤”。

第二，**转换**(ML 过程中的#4 和#6)。旧版本是:

```
train_winter_array = ohe.transform(X_train[["winter_severity_index"]])train_winter_df = pd.DataFrame(train_winter_array, index=X_train.index)X_train = pd.concat([train_winter_df, X_train], axis=1)X_train.drop("winter_severity_index", axis=1, inplace=True)test_winter_array = ohe.transform(X_test[["winter_severity_index"]])test_winter_df = pd.DataFrame(test_winter_array, index=X_test.index)X_test = pd.concat([test_winter_df, X_test], axis=1)X_test.drop("winter_severity_index", axis=1, inplace=True)
```

新版本是:

```
columns_with_ohe = [0, 1, 2, 3, "adult_antelope_population", "annual_precipitation"]X_train_array = pipe.transform(X_train)X_train = pd.DataFrame(X_train_array, columns=columns_with_ohe)X_test_array = pipe.transform(X_test)X_test = pd.DataFrame(X_test_array, columns=columns_with_ohe)
```

如您所见，我们已经从使用管道中获得了一些好处。我们不再需要手动连接编码数据和原始数据，或者手动删除原始列。

然而，在这一点上，我们有一点“黑”，我们是硬编码的列名，以便后面的代码能够工作。为了创建“低降水量”列，我们需要“年降水量”列的名称，但是一次性编码已经删除了所有的列名称。让我们继续将预处理步骤添加到管道中，并确保在自定义转换之后进行一次性编码*，这样我们就不再需要这种“黑客”了。*

# 第三，根据需要创建自定义变压器

出于功能工程的目的，我们经常希望使用 Pandas 来做一些不太常见的任务，以便像 OneHotEncoder 一样作为 scikit-learn 预处理器来包含。要在管道中实现这一点，您需要创建一个定制的 transformer 类。

再看具体的，老版的**装**就是……什么都没有。我们没有使用任何有关训练数据的信息来执行转换。旧版本的**改造**为:

```
X_train["low_precipitation"] = [int(x < 12) for x in X_train["annual_precipitation"]]X_test["low_precipitation"] = [int(x < 12) for x in X_test["annual_precipitation"]]
```

**拟合和变换**的新版本是我们增加了一个新的类沉淀变换器:

```
class PrecipitationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self
    def transform(self, X, y=None):
        X_new = X.copy()
        X_new["low_precipitation"] = [int(x < 12) for x in X_new["annual_precipitation"]]
        return X_new
```

以及管道中的一个“步骤”:

```
...
("transform_precip", PrecipitationTransformer()),
...
```

不算短，但确实避免了重复！

# 第四，加入你的模型

我认为，作为最后一步添加模型是管道真正闪光的地方。添加它的方式与添加预处理步骤的方式相同:

```
...
("linreg_model", LinearRegression())
...
```

这是最终的工作流程。我们在某种程度上减少了代码行数，但更重要的是我们不再重复任何东西！

查看前面提到的关于 ColumnTransformers 的[博客文章，scikit-learn 的](https://towardsdatascience.com/using-columntransformer-to-combine-data-processing-steps-af383f7d5260)[这个例子](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)，或者[这个中型文章](https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65)以获得更高级的例子。

感谢您的阅读，如果您有任何问题，请在评论中告诉我！