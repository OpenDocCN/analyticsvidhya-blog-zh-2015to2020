# ML 引擎中张量流估计器 API 预测结果中的目标类别标签

> 原文：<https://medium.com/analytics-vidhya/target-class-labels-in-prediction-result-of-tensorflow-estimator-api-in-ml-engine-439f23cc4047?source=collection_archive---------0----------------------->

![](img/e0e472fe2b327d1dc10e136f36da5976.png)

安妮·斯普拉特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

## 问题简介

这是我在云 ML 引擎中使用 tensorflow estimators API 配置服务模型时遇到的一个非常具体的问题。如果预测输入中有多行，并且您需要映射到每一行输出的目标类列表，这就相当棘手了。来自训练图的预测可以以配置的方式具有输出参数。例如，对于多类分类器，它可以包括预测的类或标签名称、类的概率列表、精确度等等。但是我发现有一件事很棘手，那就是将实际的类列表或标签名称添加到预测结果中。当一个已部署的模型被某个不知道类是什么的人使用时，这将是有用的，他可以获得关于它们的概率的信息。

似乎没有直接的方法可以在预测输出列表中添加类名。我们能够从 softmax 层获得与每个类相关联的概率，并且使用 **tf.gather** 我们可以获得预测的类名。

但是，如果您想在预测结果中获得类名列表，该怎么办呢？在像 StackOverflow 这样的地方进行了大量疯狂的搜索后，我得出结论，以前从来没有人遇到过这个特殊的问题。所以经过大量的试错，我想出了这个解决方案(可能有点工程过度)。

## 解决办法

使用逻辑上的 **tf.nn.softmax** 函数，可从神经网络的 softmax 层获得概率。

```
probabilities = tf.nn.softmax(logits)
```

现在，为了获得与最高概率相关联的目标标签或类名，我们可以对给定的目标名称列表使用 **tf.gather** 方法。使用 **tf.argmax** 方法获得预测指数，使用 **tf.gather** 和目标列表获得目标名称。

```
predicted_indices = tf.argmax(probabilities, 1)predicted_class = tf.gather(TARGET_LABELS, predicted_indices)
```

现在是获取姓名列表并将其添加到预测结果对象的棘手部分。首先，我使用 **tf.where** 来获取所有概率值的所有索引，条件是所有值都小于 2，这是所有的概率值。现在有了所有的指数，我找到了最后一个指数，形成了一个新的包含指数的列表，然后把它改造成概率的列表。现在，这个新的变形矩阵包含了所有的索引。

如果目标类别是:-

```
['Class A', Class B']
```

如果有两个输入，预测的概率将以 2x2 矩阵的形式出现

```
[[.5, .4], [.4, .5]]
```

从 CloudML 引擎获得输出的方式将是多行预测的对象列表。为了映射到这种情况，首先形成一个形状相似的索引矩阵，并使用 **tf.gather，**将类名也转换成如下形式的矩阵

```
[['Class A', Class B'], ['Class A', Class B']]
```

这将允许正确映射对象列表的结果。进行解释的代码是:-

```
condition = tf.less(probabilities, 2)*#all the indices are obtained* indices = tf.where(condition)

*# get the last index from the result* last_index = indices.get_shape().as_list()[1] - 1#the new list containing all the indices --> [0, 1, 0, 1]
last_indices_value = tf.slice(indices, [0, last_index], [-1, -1])

*# reshape the result to the correct format* # [0, 1, 0, 1] --> [[0, 1], [0, 1]]
classes_shape = tf.reshape(last_indices_value, tf.shape(probabilities))

*# form the classes list with the indices in the new shape* classe_names = tf.gather(TARGET_LABELS, classes_shape)
```

## 结论

因此，这是在预测中映射或包含类名列表的过度设计的方法，它将与预测输入中的任意数量的行一起工作。这主要是为了让我记住这种方法以及我是如何着手解决它的，也是为了让任何碰巧遇到类似问题的人记住。

请评论，如果你有一个替代或更简单的方式做同样的事情。谢谢你。