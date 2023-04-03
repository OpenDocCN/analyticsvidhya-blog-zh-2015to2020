# 评估随机森林模型

> 原文：<https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56?source=collection_archive---------0----------------------->

随机森林是分类问题的一个强大工具，但与许多机器学习算法一样，它可能需要一点努力才能准确理解预测的内容及其在上下文中的含义。幸运的是，Scikit-Learn 使得运行随机森林并解释结果变得非常容易。在这篇文章中，我将介绍训练一个简单的随机森林模型，并使用混淆矩阵和分类报告评估其性能的过程。我甚至将向您展示如何使用 Seaborn 和 Matplotlib 制作一个颜色编码的混淆矩阵。请继续阅读！

![](img/a3a795024b7376219054b525d453f3ae.png)

只是一些随机的森林。(笑话自己写的！)

本教程的数据集是由 J. A. Blackard 在 1998 年创建的，它包括 50 多万个观察值和 54 个特征。每一次观察都代表了科罗拉多州荒野地区一块 30 米乘 30 米的土地。这些要素记录了每个区域的制图数据:海拔、坡向、到水/道路/经过野火着火点的距离、一天中不同时间的遮荫量以及它包含 40 种土壤类型中的哪一种。你可以在 UCI KDD 档案馆找到完整的数据集和描述。

# 准备好

在我们跋涉到随机森林之前，让我们收集我们需要的包和数据。我们需要熊猫和小熊猫来帮助我们处理数据。我还将导入 Matplotlib 和 Seaborn，用于稍后创建的彩色编码可视化。

我们还需要从永远有用的 [Scikit-Learn](https://scikit-learn.org/stable/) 中得到一些东西。从 *sklearn.model_selection* 我们需要 *train-test-split* 以便我们可以在数据集的单独块上拟合和评估模型。当然，我们需要我们的 *RandomForestClassifier* ，并且从 *sklearn.metrics* 我们将需要 *accuracy_score* 、 *confusion_matrix* 和 *classification_report* 。装上子弹。

```
# Import needed packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# If you're working in Jupyter Notebook, include the following so that plots will display:
%matplotlib inline
```

现在我们准备好获取数据了。如果你正在下载一份数据到你的本地机器，请注意它的大小大约是 75MB。

```
# Import the dataset
df = pd.read_table("covtype.data", sep=',', header=None)
```

数据帧 *df* 包含 54 列代表特征，一列包含目标(标签)名为 *Cover_Type* 。让我们将数据分成特征和目标。

```
# Split dataset into features and target
y = df['Cover_Type']
X = df.drop('Cover_Type', axis=1)
```

很简单！我很好奇我们每个班有多少人。让我们检查价值计数:

```
# View count of each class
y.value_counts()
```

![](img/2a0ba90902f5e5fa5ecbc4966085611b.png)

这些班级非常不平衡——最小的班级只有最大班级的 1%!用这种类别不平衡的数据建模有点冒险，因为模型看不到全局。他们想找到一种方法来最大化你正在使用的任何评估指标，为此，他们可能会找到捷径。例如，如果你有两个类，其中一个有 99 个例子，另一个只有 1 个，一个模型可以*总是*预测第一个类，并且它在 99%的时候都是正确的！该模型在准确性上得分很高，但它实际上不会帮助您识别较小类的示例。

由于这是一个快速和肮脏的模型，为了查看一些评估工具的目的，我现在不会费心平衡类。只要知道你可以用来自[不平衡学习](https://imbalanced-learn.readthedocs.io/en/stable/)的一些工具很容易地做到这一点。

我们想要评估随机森林在它以前没有见过的数据上的表现，这意味着我们需要做一个训练测试分割。

```
# Split features and target into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
```

我想在上面的代码中指出两件重要的事情。首先是我设置了一个*random _ state*；这确保了如果我必须重新运行我的代码，我将得到完全相同的训练测试分割，所以我的结果不会改变。我不想花一整天的时间去评估一个每次我需要重启内核时输出都会改变的模型！

我要指出的第二点是*strategy = y*。这告诉 *train_test_split* 确保训练和测试数据集包含每个类的示例，其比例与原始数据集中的比例相同。这一点尤其重要，因为这些班级是多么的不平衡。随机拆分很容易导致测试集中最小类的所有示例，而训练集中没有任何示例，然后模型将无法识别该类。

# 走进树林

数据准备好了。是时候让随机森林开始运行了！如果您曾经使用过 Scikit-Learn，您会知道许多建模类都有完全相同的接口:您实例化一个模型，调用*。fit()* 来训练它，然后调用*。predict()* 获取预测。我将实例化一个*RandomForestClassifier()*并保留所有默认参数值。

```
# Instantiate and fit the RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
```

![](img/69f6aae409317e60e9e221e6004e3410.png)

当您符合模型时，您应该会看到如上所示的打印输出。这将告诉您模型中包含的所有参数值。[查看 Scikit-Learn 的随机森林分类器](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)的文档，了解每个参数的作用。

现在我们可以得到测试数据的预测标签:

```
# Make predictions for the test set
y_pred_test = forest.predict(X_test)
```

现在是我们对模型性能的第一次评估:准确性得分。该分数衡量模型在预测总数中获得了多少标签。你可以认为这是正确预测的百分比。使用 Scikit-Learn，使用测试集的真实标签和测试集的预测标签，这非常容易计算。

```
# View accuracy score
accuracy_score(y_test, y_pred_test)
```

该模型在测试数据上的准确率为 94%。这看起来很令人印象深刻，但是请记住，当类别不平衡时,**准确性不是分类器性能的一个很好的衡量标准。我们需要更多的信息来理解模型的实际表现。它在每个班级的表现都一样好吗？有没有哪几对类别特别难以区分？让我们用一个混淆矩阵来找出答案。**

# 混淆矩阵

一个[混淆矩阵](https://en.wikipedia.org/wiki/Confusion_matrix)是一种表达有多少分类器的预测是正确的，以及当不正确时，分类器在哪里被混淆(因此得名！).在下面的混淆矩阵中，行代表真实标签，列代表预测标签。对角线上的值表示预测标签与真实标签匹配的次数(或百分比，在归一化混淆矩阵中)。其他单元格中的值表示分类器错误标记观察值的情况；列告诉我们分类器预测了什么，行告诉我们正确的标签是什么。这是一种发现模型可能需要一些额外训练的区域的便捷方法。

[Scikit-Learn 的 *confusion_matrix()*](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) 获取真实标签和预测，并将混淆矩阵作为数组返回。

```
# View confusion matrix for test data and predictions
confusion_matrix(y_test, y_pred_test)
```

![](img/3c195f4298a7eee94045a5236d792906.png)

我们可以直接看到，在第一列和第二列的非对角线单元格中有一些大值，这意味着分类器预测了第 1 类和第 2 类很多次，而这是不应该的。这并不奇怪；这是两个最大的类别，分类器可以通过反复猜测这两个类别中的一个来获得很多正确的预测。

如果这个混淆矩阵有一些标签，甚至有一个色标来帮助我们找出最大值和最小值，它会更容易阅读。我将使用一个[Seaborn*heat map()*](https://seaborn.pydata.org/generated/seaborn.heatmap.html)来做到这一点。我还将标准化混淆矩阵中的值，因为我发现当类的大小如此不同时，百分比比绝对计数更容易理解。

```
# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 
               'Cottonwood/Willow', 'Aspen', 'Douglas-fir',    
               'Krummholz']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()
```

![](img/988dd455926d56fa9d675199d0c3170c.png)

不错！现在很容易看出，我们的分类器在预测白杨标签时遇到了困难。大约有四分之一的时间，白杨被误认为是黑松！

# 分类报告

为了更深入地了解模型性能，我们应该检查其他指标，如精确度、召回率和 F1 分数。

**Precision** 是一个类中被正确识别的成员数除以模型预测该类的所有时间。在白杨的情况下，精度分数将是正确识别的白杨的数量除以分类器正确或错误预测“白杨”的总次数。

**回忆**是分类器正确识别的一个类的成员数除以该类的成员总数。对于 aspen，这将是分类器正确识别的实际 aspen 的数量。

**F1 得分**有点不太直观，因为它将精确度和召回率结合在一个指标中。如果精度和召回率都很高，F1 也会很高。如果都低，F1 就低。如果一个为高，另一个为低，F1 将为低。F1 是一种快速判断分类器是否真的擅长识别一个类的成员，或者它是否在寻找捷径(例如，只是将所有东西都识别为一个大类的成员)。

让我们使用 [Scikit-Learn 的*classification _ report()*](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)来查看我们模型的这些指标。我建议用一个 *print()* 包装它，这样它的格式会很好。

```
# View the classification report for test data and predictions
print(classification_report(y_test, y_pred_test))
```

![](img/5133f764014e8968dbd5b4014770d22c.png)

查看第 5 类(白杨)的指标。精确度很高，这意味着该模型小心翼翼地避免将不是白杨的东西标为“白杨”。另一方面，召回率相对较低，这意味着分类器由于过于小心而遗漏了一堆白杨！F1 的分数反映了这种不平衡。

就其本身而言，分类报告一般会告诉我们该模型犯了什么样的错误，但它不会给出具体细节。混淆矩阵告诉我们错误发生的确切位置，但它没有给出精确度、召回率或 F1 分数等汇总指标。使用这两者可以让我们对模型的表现有更细致的了解，这远远超出了准确性分数所能告诉我们的，并避免了它的一些缺陷。

*交叉发布自我在*[*jrkreiger.net*](http://jrkreiger.net)*的博客。*