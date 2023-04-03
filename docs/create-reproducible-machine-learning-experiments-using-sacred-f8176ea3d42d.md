# 使用 Sacred 创建可重复的机器学习实验

> 原文：<https://medium.com/analytics-vidhya/create-reproducible-machine-learning-experiments-using-sacred-f8176ea3d42d?source=collection_archive---------15----------------------->

*每个实验都是神圣的*
*每个实验都是伟大的*
*如果一个实验被浪费了*
*上帝会非常生气*

![](img/42696ae90c6c96630863b493d66905d7.png)

[Katarzyna Pe](https://unsplash.com/@kasiape?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

[神圣](https://github.com/IDSIA/sacred)让你配置、组织、记录和复制实验。它是专门为 ML 实验设计的，但实际上可以用于任何种类的实验。

为了举例说明如何使用这个强大的框架，我将使用一个 Kaggle 竞赛的数据集，[真实与否？灾难推文 NLP](https://www.kaggle.com/c/nlp-getting-started)。这个竞赛是一个二元分类问题，你应该决定一条推文是否描述了一场真实的灾难。这里有两个例子:

**真实灾难推文:**

```
Forest fire near La Ronge Sask. Canada
```

**不是灾难推文:**

```
I love fruits
```

> 数据科学家迟早会注意到，模型的性能在很大程度上依赖于特定的配置和无数的数据修改。

假设我们想要运行一些实验，我们建立一个模型来对这些推文进行分类，并使用 [k 倍交叉验证](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)来测量分类器的 [F1 得分](https://en.wikipedia.org/wiki/F1_score)。大多数数据科学家可能会打开一个 Jupyter 笔记本，开始探索数据(顺便说一句，这确实总是正确的事情)，进行一些特别的实验，建立和评估模型。数据科学家迟早会注意到，模型的性能在很大程度上依赖于特定的配置和无数的数据修改。这就是再现性的力量开始得到回报的地方。

# 为什么神圣？

以下是使用**圣物**的主要特点和优势:

*   轻松地**定义**和**封装**每个实验的**配置**
*   自动**收集每次运行的元数据**
*   **记录**自定义指标
*   **使用**观察器**收集各地的**日志
*   确保**确定性**与**自动播种**一起运行

# 如何设置一个神圣的实验

我们从创建一个神圣的基础实验开始，如下所示:

```
logreg_experiment = **Experiment**(‘logreg’)
```

一个神圣的实验是由一个配置定义的，所以让我们创建一个:

```
@logreg_experiment.config
def **baseline_config**():
    max_features = None
    classifier = **Pipeline**([
        (‘tfidf’, **TfidfVectorizer**(max_features=max_features)),
        (‘clf’, **LogisticRegression**())
    ])
```

注意，实验对象的`config`属性被用作函数装饰器。这使得神圣的自动检测，该功能应被用来配置实验。

这个非常简单的配置定义了一个包含两个步骤的 **scikit-learn 管道**:计算所有 tweets 的 [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) 表示，然后使用[逻辑回归](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)对它们进行分类。我为其中一个超级参数`max_features`添加了一个变量，以展示如何通过修改配置来轻松创建新的实验。

现在，在运行这个实验之前，必须定义一个主函数:

```
@logreg_experiment.automain
def **main**(classifier):
    datadir = **Path**(‘../data’)
    train_df = pd.**read_csv**(datadir / ‘train.csv’)
    scores = **cross_val_score**(classifier, train_df[‘text’],
        train_df[‘target’], cv=5, scoring=’f1')
    mean_clf_score = scores.**mean**()
    logreg_experiment.**log_scalar**(‘f1_score’, mean_clf_score)
```

正如你所看到的，我们再次使用实验对象的属性作为装饰，在这个例子中是`automain`。这让 main 函数自动访问这个实验的配置中定义的任何变量。在这种情况下，我们只通过了`classifier`,将根据它在训练集上使用 5 重交叉验证对 Twitter 数据进行分类的能力进行评估。在最后一行代码中，我们想要测量的指标是使用`log_scalar`方法记录的。

# 进行实验

要运行实验，只需调用它的`run()`方法。为了用不同的参数值运行它，您可以方便地传递一个 dict `config_updates`来指定这个实验运行的确切配置。相当整洁！

```
# Run with default values
logreg_experiment.**run**()# Run with config updates
logreg_experiment.**run**(config_updates={‘max_features’: 1000})
```

我通常将实验本身放在不同的文件中，然后有一个单独的脚本来一次运行所有的实验。

# 记录您的结果

如果你运行上面的，你不会看到很多结果。你首先需要将一个**观察者**连接到实验上。然后**观察者**会将**日志**发送到某个**目的地**，通常是一个**数据库**。对于本地和非生产用途，您可以使用`FileStorageObserver`简单地写入磁盘。

```
logreg_experiment.observers.**append**(**FileStorageObserver**(‘logreg’))
```

如果您在上面的 runner 脚本中包含这一行并运行它，那么每次运行都会创建一个新文件夹`logreg`,其中包含一个子文件夹。一个用于默认运行，一个具有更新的`max_features`值。每个人都创建了四个单独的文件，内容如下:

*   **config.json** :配置**中每个对象的状态**，以及在所有**非确定性函数**中自动使用的`seed`参数，以保证**再现性**。
*   **cout.txt** :运行过程中产生的所有**标准输出**。
*   **metrics.json** : **在运行过程中记录的自定义指标**，例如我们案例中的 F1 分数。
*   **run.json** : **元数据**例如关于源代码(git repo、文件、依赖项等。)、运行主机、启动/停止时间等。

# 把所有的放在一起

为了完整起见，我将创建最后一个示例来展示如何从同一个 runner 脚本运行多个实验:

```
from pathlib import Path
import pandas as pd
from sacred import Experiment
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipelinerand_forest_experiment = **Experiment**(‘randforest’) @rand_forest_experiment.config
def **baseline_config**():
    n_estimators = 100
    classifier = **Pipeline**([
        (‘tfidf’, **TfidfVectorizer**()),
        (‘clf’, **RandomForestClassifier**(n_estimators=n_estimators))
    ])@rand_forest_experiment.automain
def **main**(classifier):
    datadir = **Path**(‘../data’)
    train_df = pd.**read_csv**(datadir / ‘train.csv’)
    scores = **cross_val_score**(classifier, train_df[‘text’],
        train_df[‘target’], cv=5, scoring=’f1')
    mean_clf_score = scores.**mean**()
    rand_forest_experiment.**log_scalar**(‘f1_score’, mean_clf_score)
```

现在，让我们通过一些配置更新来运行这两个实验…

```
from sacred.observers import FileStorageObserver
from experiments.logreg import logreg_experiment
from experiments.randforest import rand_forest_experimentlogreg_experiment.observers.**append**(**FileStorageObserver**(‘logreg’))
rand_forest_experiment.observers.**append**(**FileStorageObserver**(‘randforest’))# Run with default values
logreg_experiment.**run**()# Run with config updates
logreg_experiment.**run**(config_updates={‘max_features’: 1000})# Run different experiment
rand_forest_experiment.**run**()
rand_forest_experiment.**run**(config_updates={‘n_estimators’: 500})
```

通过查看每次运行的`metrics.json`文件，我们可以得出结论，默认的逻辑回归模型表现最好，F1 值约为 0.66，而有 100 个估计值的随机森林表现最差，F1 值约为 0.53。

当然，所有这些 json 格式的输出看起来并不是很吸引人，但是有几个**可视化工具**你可以使用。然而，这已经超出了本文的范围，但是可以看看这里:[https://github.com/IDSIA/sacred#Frontends](https://github.com/IDSIA/sacred#Frontends)

安全实验！

本文是构建和设计机器学习系统的最佳实践系列的一部分。在这里阅读第一部分:[https://medium . com/analytics-vid hya/how-to-get-data-science-to-really-work-in-production-bed 80 E6 bcfee](/analytics-vidhya/how-to-get-data-science-to-truly-work-in-production-bed80e6bcfee)