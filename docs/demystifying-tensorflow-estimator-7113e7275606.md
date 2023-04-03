# 揭开张量流估计器的神秘面纱

> 原文：<https://medium.com/analytics-vidhya/demystifying-tensorflow-estimator-7113e7275606?source=collection_archive---------25----------------------->

![](img/df5f34ab645571a5972b6687b90d4943.png)

Tensorflow 是最受欢迎的机器学习库之一。它是少数几个让你在机器学习的最低和最高抽象层次工作的框架之一。您可以使用处理一切的高级 API 进行有用的机器学习，或者您可以从头开始构建一个自定义模型，控制从模型架构到训练循环的一切

> 您可以在这里找到并运行完整的代码片段示例

## 什么是张量流估计量？？

类似于 **keras 的模型 API** ，Tensorflow Estimator 是一个**的高层** API，**封装了训练、评估、预测和服务**。换句话说，您不必编写训练、评估或预测循环，esitmator 会为您完成所有工作。它还为您提供了一个可重复使用的模型，您可以在任何地方保存或加载该模型。

## 为什么要使用 Estimator API？

Estimator 是一个易于使用的 API，它为您处理大量的复杂性，但仍然提供足够的定制机会。您可以一次设计您的模型，它将在您的本地机器上无缝运行，无论有无 GPU，也可以在有 GPU 和 TPU 的多服务器分布式环境上运行！！！

它安全地运行一个分布式训练循环，负责加载数据、检查点并为 tensor board 编写摘要

## 它是如何工作的？

Estimator API 允许您使用预制的以及定制的估算器。在这里，我将介绍如何使用预制的估算器，而将定制的估算器留待下次讨论

通过以下步骤，您可以使任何预制的评估器工作

## 答:数据集输入函数

这些函数接收数据集并返回要素字典和标注。特征字典是数据集中特征名称到特征值的映射

```
def make_input_fn(df, epochs=500, shuffle=True, batch_size=32):
    df = df.copy()
    labels = df.pop('AveragePrice') # extract label from dataframe
    def input_function():
        *# create dataset from inmemory pandas dataframe* 
        dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df))

        dataset = dataset.batch(batch_size).repeat(epochs)
        return dataset
    return input_function

train_input_fn = make_input_fn(train_df)
eval_input_fn = make_input_fn(eval_df, epochs=1, shuffle=False)
```

在上面的示例中，make_input_fn 是输入函数创建者，因为它返回实际数据集输入函数 input_function

## b:定义特征列

在第二步中，我们创建了一个所有特征列的列表，我们希望在这些特征列上训练我们的模型。它将所有特性定义为 tf.feature_column，它需要特性名称和可选的数据类型以及预处理函数

```
DENSE_COLUMNS = ['Total_Bags','Small_Bags','Large_Bags','XLarge_Bags']

SPARSE_COLUMNS = ['type', 'year', 'region']

feature_columns = []
for feature **in** DENSE_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature))
for feature **in** SPARSE_COLUMNS:
    vocab = data[feature].unique()
    categorical_feature = tf.feature_column.categorical_column_with_vocabulary_list(feature, vocab)
    feature_columns.append(tf.feature_column.indicator_column(categorical_feature))
```

在上面的例子中，我有一个数据集，它有一些密集(连续)和一些稀疏(分类)的特征。对于每个特征类型，创建 tf.feature 并将其添加到 feature_columns 列表中。然后，我们的模型将使用这个列表来处理数据集中的实际数据

## c:实例化评估人员和培训

有许多[预制估算器](https://www.tensorflow.org/api_docs/python/tf/estimator/)可用。你可以在这里浏览完整的[列表。您只需将专题专栏传递给它，就万事大吉了。如果需要执行参数调整，也可以传递这些参数。要开始训练，只需在估计器上调用 train 方法，并将数据集输入函数作为参数传递](https://www.tensorflow.org/api_docs/python/tf/estimator/)

```
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    model_dir=output_dir
)
linear_regressor.train(train_input_fn)
```

仅此而已！！！

您的模型将开始训练，并在 output_dir 中保存模型检查点和摘要。你也可以将张量板指向这个目录来监控训练

*Tensorflow 附带了****tensor board****，这是一个用于可视化不同模型指标的强大工具。你可以在*[***Neptune . ai***](https://bit.ly/2STa5Ax)查看非常有用的 tensorboard 介绍