# 在 d6tflow DAG 中包装 Scikit-learn 工作流的实践教程

> 原文：<https://medium.com/analytics-vidhya/hands-on-tutorial-to-wrapping-your-scikit-learn-workflow-in-a-d6tflow-dag-3e4cca8d7479?source=collection_archive---------11----------------------->

![](img/d65b3014bcc81d1baa6daba56c8b8f7c.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Helloquence](https://unsplash.com/@helloquence?utm_source=medium&utm_medium=referral) 拍摄的照片

我正在阅读[诺姆的文章](https://towardsdatascience.com/4-reasons-why-your-machine-learning-code-is-probably-bad-c291752e4953)，并对学习如何使用 d6tflow 将数据科学工作流包装在由 d6tflow 管理的 DAG ( [有向无环图](https://en.wikipedia.org/wiki/Directed_acyclic_graph))中产生了兴趣。

我鼓励您阅读 Norm 的文章，以更好地了解使用 DAG 的动机和背景。

有向无环图是不含圈的有限有向图。简而言之，DAG 是一个图，在这个图中，你可以从任何节点开始，沿着有方向的箭头，永远不会回到你开始的地方。一个很好的类比是想象一条流动的小溪。数据科学工作流中的任务依赖链通常可以自然地表示为有向无环图。

在 jupyter 笔记本上进行一些设计表达性和易于遵循的机器学习示例/工作流的实验后，我开始觉得我可以在笔记本上进行更明确的组织和记账。在笔记本的不同位置定义函数，结合 jupyter 笔记本单元的状态性质，当我试图调试为什么机器学习管道没有给我预期的东西时，经常会让我感到悲伤。

我倾向于同意这样一种观点，即数据科学工作流最好编写为一组相互依赖的任务，而不是一组您必须保证线性执行的功能。您必须定义什么时候应该执行什么，以及哪些步骤需要前面步骤的输出。这也很有帮助，因为它让您从代码中后退一步，从整个管道的角度考虑您正在创建的工作流，而不仅仅是某个单元或一组单元的结果。

本文的目的是提供一个深入的例子，说明如何在 d6tflow 任务中包装一个完整的机器学习工作流。我认为将我从不同来源的文档和示例中获得的经验汇编成一篇博文会很有用。我试图创建一个指南来帮助人们快速开始在 d6tflow DAGs 中包装他们的工作流。我试图澄清和阐述最初使用 d6tflow 时可能不明显的一些关键点。在本文中，我将介绍一个纯粹的 scikit-learn 管道示例，并将其转换为一组 d6tflow 任务。

这个例子摘自[用 Scikit-Learn 和 TensorFlow 实践机器学习:构建智能系统的概念、工具和技术](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291/ref=sr_1_3?crid=2DEFL9N1UJICH&keywords=hands+on+machine+learning+with+scikit-learn+and+tensorflow&qid=1566963615&s=gateway&sprefix=hands+on+machine%2Caps%2C153&sr=8-3)，[第 2 章，机器学习项目的端到端例子](https://github.com/ageron/handson-ml/blob/master/02_end_to_end_machine_learning_project.ipynb)。我将首先重新陈述这个机器学习项目的最终结果的压缩版本，然后我将展示如何将其包装在一组 d6tflow 任务中。

目标是在给定一些地区信息的情况下，预测加利福尼亚地区的房价中值。我们得到每个区的数据，我们想预测房价中值，所以这是一个回归问题。

```
# general imports
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as npimport d6tflow# d6tflow uses luigi to intelligently pass parameters upstream and downstream
import luigi
from luigi.util import inherits, requires# scikit learn components for our workflow
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
```

首先，我们需要加载数据(你可以从[这里](https://github.com/ageron/handson-ml/tree/master/datasets/housing)下载)。出于几个原因，将分类变量声明为 category 是一个好主意。它使得原本属于对象类型的数据被更有效地存储，并允许我们在稍后的`ColumnTransformer`中基于数据类型选择列。

```
DATA_DIR = "../Aurelien_ML/data"
FILENAME = "housing.csv"# good practice to use os.path.join to generate full file paths
data = pd.read_csv(os.path.join(DATA_DIR, FILENAME))
# declare features you know are categorical as category 
data.ocean_proximity = data.ocean_proximity.astype("category")
data.info()
```

现在，我们将数据分成一组独立变量(我们的特征)和一个因变量(我们的目标变量——在本例中是中值房价)。

```
target_variable = "median_house_value"
features = [col for col in data.columns if col != target_variable]housing = data.loc[:,features]
labels = data.loc[:,target_variable]
print("Features to use: \n{}\n".format(list(housing.columns)))
print("Target variable: {}".format(labels.name))
```

Aurelion Geron 增加了一些额外的特性，展示了一个如何在 scikit-learn 中创建自定义转换器的示例。我包含了这些特性，以展示它们如何适应我们的管道。

至少有两种方法可以做到这一点，因为我们的特征化足够简单。我们所做的只是提取一些特征(房间、卧室、人口、家庭)并从它们的比例中创建新的特征。创建一个简单的转换器来将这些特性添加到我们的数据集中的第一种方法是从头创建一个定制的转换器，它继承自`BaseEstimator`和`TransformerMixin`。为了创建一个定制的转换器，我们只需要实现方法`fit`、`transform`和`fit_transform`(鸭子打字)。我们从`TransformerMixin`继承所以我们免费得到`fit_transform`，我们从`BaseEstimator`继承，免费得到`set_params`和`get_params`。这将允许我们通过访问转换器的参数(例如打开或关闭特征化)在管道内部执行超参数调整。

```
# here is how to do it from scratch with a custom transformer
class AddNewFeatures(BaseEstimator, TransformerMixin):
    """
    a class to add the features for rooms per household, population per household and berooms per room
    we inherit from BaseEstimator and TransformerMixin to get some stuff for free, such as fit_transform
    """
    def __init__(self, column_names):
        self.rooms_index, self.bedrooms_index, self.population_index, self.household_index = [ \
            list(column_names).index(col) for col in ("total_rooms","total_bedrooms","population",\
                                                         "households")]
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        """
        X - np array containing features
        y - np array containing labels
        """
        rooms_per_household = X[:, self.rooms_index] / X[:, self.household_index]
        populations_per_household = X[:, self.population_index] / X[:, self.household_index]
        bedrooms_per_room = X[:, self.bedrooms_index] / X[:, self.rooms_index]
        return np.c_[X, rooms_per_household, populations_per_household, bedrooms_per_room]Feature_Adder_Trans = AddNewFeatures(data.columns)
```

第二种方法是使用`FunctionTrasformer` API，它允许我们构造一个类，给定一个函数，该函数将整个数据集作为输入，并返回数据集的转换版本作为输出。下面是如何实现与上面相同的转换器:

```
# here is how to use the FunctionTransformer
# first, define the function that specifies the transformation
def add_extra_features(X, cols=None):
    rooms_index, bedrooms_index, population_index, household_index = [ \
            list(cols).index(col) for col in ("total_rooms","total_bedrooms","population",\
                                                         "households")]
    rooms_per_household = X[:, rooms_index] / X[:, household_index]
    populations_per_household = X[:, population_index] / X[:, household_index]
    bedrooms_per_room = X[:, bedrooms_index] / X[:, rooms_index]
    return np.c_[X, rooms_per_household, populations_per_household, bedrooms_per_room]# then instantiate a FunctionTransformer with the function specifying the transformation
Feature_Adder_FT = FunctionTransformer(add_extra_features, validate=False, kw_args={"cols":housing.columns})
```

现在，我们将组装最后的管道。首先，我们需要将分类变量转换成适合在 ML 管道中使用的独热编码。为此，我们将计算在我们的单个分类列中会遇到的不同类别，并将其传递给`OneHotEncoder`,这样我们就不会在尝试对尚未看到的类别进行编码时遇到问题。这只是一个问题，因为如果你不提前告诉它预期的类别，那么稍后当我们进行交叉验证时，可能会发生这种情况，我们的管道将尝试计算测试文件夹的性能得分，并遇到它在相应的培训文件夹中尚未看到的类别。(去掉 categories 参数就明白我的意思了。发生这种情况是因为`ocean_proximity`中的`ISLAND`类别出现的时间非常少，所以您很可能在生成的训练折叠之一中看不到它)。

```
categories = list(housing.ocean_proximity.unique())
```

因为我们有数字列和分类列，所以我们需要为它们创建不同的管道，并以正确的方式将它们合并成一个管道。对于我们的数字管道，转换是简单的:我们有一个估算器来估算缺失值(该列的中值)，特征器(添加新的特征列)和一个缩放器(标准缩放器缩放每一列，使平均值和单位方差为 0)。对于分类管道，我们简单地使用类别的一次性编码。我不会过多地探究一键编码的细节，但是一些背景知识可能会有所帮助。

one hot 编码意味着我们用一个在相应位置包含 1 的向量替换列中的每个元素(这在后面会很重要)。因为有 5 个可能的类别([‘靠近海湾’，‘< 1H 海洋’，‘内陆’，‘靠近海洋’，‘岛屿’])，所以我们的矢量将是 5 维矢量。所以我们的单列(Nx1)被一个 5 列的矩阵(Nx5)所取代。

然后，我们需要以一致的方式将这个数字管道与分类管道结合起来。我们通过将它们放在为此目的而创建的`ColumnTransformer`中来实现这一点。我们需要在每个元组中传递来自原始`DataFrame`的数字特性名称和分类特性名称的列表。我们基于原始`DataFrame`中的`dtypes`来做这件事(这一步取决于我们之前将分类列转换为“类别”)。

```
numerical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("featurization", Feature_Adder_FT),
                               ("strd_scaler", StandardScaler())], verbose=True)categorical_pipeline = Pipeline([("one_hot_encoding", OneHotEncoder(categories=[categories]))],verbose=True)numerical_features = list(housing.select_dtypes(include=[np.number]).columns)
categorical_features = list(
    housing.select_dtypes(include=["category"]).columns)full_pipeline = ColumnTransformer([("numerical_pipeline", numerical_pipeline, numerical_features),
                                   ("categorical_pipeline", categorical_pipeline, categorical_features)],
                                 verbose=True)
```

我们现在已经完成了预处理管道。唯一需要添加的是一个预测器(一个具有`predict`方法的 scikit-learn 估计器)。但首先，我们应该评估一下我们这里有什么，以确保我们了解正在发生什么。我们从 9 个特征开始(不包括目标变量)——8 个数字和 1 个分类。因为我们添加了自定义功能(添加了 3 个新功能)，这个数字上升到 12。然后，我们用 5 个独热列替换了分类列，这使得最终计数达到 12–1+5 = 16。我们可以将此作为一个健全性检查:

```
processed_data = full_pipeline.fit_transform(housing)
processed_data.shape
```

太好了！现在剩下唯一要做的事情是在我们的管道末端附加一些预测器，产生另一个管道(这可以是任何 sklearn 模型，对于本文的目的来说无关紧要):

```
full_pipeline_with_predictor = Pipeline(
    [("data_preprocessing", full_pipeline), ("linear_regressor", LinearRegression())])
```

现在，我们可以使用这最后的管道来做各种很酷的事情！例如，我们可以执行交叉折叠验证来调整超参数。注意，对于`param_grid`，语法是给你想要变化的参数的名字作为键，值是你想要尝试的值的集合。键的命名方案是跟随转换器的名称，直到您返回到您想要改变其参数的转换器(您用双下划线分隔不同转换器的名称，以便您在元组中为转换器指定的名称可以包含单下划线)。因此，作为一个例子，为了改变`imputer`转换器的`strategy`参数，我们跟随各种管道的名称，直到到达`imputer`。注意，你也可以用它来做一些事情，比如打开和关闭你的一个转换。例如，如果我们想关闭我们的自定义特征，我们可以向类定义添加一个`do_featurization`参数，并将转换代码包装在 if-else 块中，说明如果`do_featurization==False`返回原始数据。然后你可以添加到你的`param_grid` : `{"data_preprocessing__numerical_pipeline__featurization__do_featurization":[True, False]}`。网格搜索将根据您正在改变的参数的所有其他可能的组合来尝试这两个值。因此，如果我们将这个维度添加到参数网格中，我们将总共有 2x2=4 组不同的超参数可供尝试。并且对于每一组参数，我们都需要训练`cv=5`折叠。所以这很快就变得计算昂贵，但是它确实给了你一个更可靠的泛化误差估计。

```
param_grid = [
    {"data_preprocessing__numerical_pipeline__imputer__strategy": ["mean", "median"]}]grid_search = GridSearchCV(full_pipeline_with_predictor, param_grid, cv=5, scoring="neg_mean_squared_error",
                           verbose=2, n_jobs=8)
print(grid_search.fit(housing, labels))
```

我们还可以使用交叉折叠验证和我们在管道中设置的参数进行评估，并获得分数:

```
cross_val_scores = cross_val_score(full_pipeline_with_predictor, housing, labels, cv=10)
```

您可以获得最佳参数集，如下所示:

```
best_params = grid_search.best_params_
```

您可以训练最终的估计器(如果您离开`refit=True`，它也将是训练的最高得分的估计器——这是默认的，并导致您的`GridSearchCV`对象在最后使用找到的最佳参数集重新训练您的模型):

```
best_model = grid_search.best_estimator_
```

最后，您可以像这样保存并重新加载管道:

```
model = full_pipeline_with_predictor
joblib.dump(model, "full_pipeline_with_predictor.pkl")model_loaded = joblib.load("full_pipeline_with_predictor.pkl")
```

唷！这是一个很好的安排。现在，我们开始相对简单的工作，将所有这些都包装到 d6tflow DAG 中。

基本上，您为想要跟踪的每个任务创建一个新的类。通过继承`d6tflow.tasks.*`中的一个类，该类成为 d6tflow。您可以通过多种方式创建上游依赖关系。要么在你的类中定义一个`requires`方法，它返回你希望当前类依赖的任务类的实例，要么用`@requires`来修饰你的类。我更喜欢后者，因为它实际上自动为您定义了一个`requires`方法，并且它允许在上游类中定义的 luigi 参数向下游传播。

**当任务存在输出文件时，该任务被标记为“完成”。**

这可以防止已完成的任务再次运行。任务通常有一些输出，并保存到某个文件中。我们使用以下命令设置保存输出数据的根目录:

```
d6tflow.set_dir("d6tflow_output/")
```

这意味着将输出保存在当前工作目录下的一个名为`d6tflow_output`的文件夹中。这通常是我在 d6tflow 项目中的第一行。默认情况下，该路径设置为`data/`。

现在，开始我们的第一项任务！首先，我们编写一个任务来加载数据，并将其分为要素和标注:

```
class TaskGetData(d6tflow.tasks.TaskPqPandas):
    persist = ["training_data", "labels"]

    def run(self):
        """
        loading data from external files is done inside run
        so now, if you check os.path.join(d6tflow.settings.dirpath, TaskGetData().output().path),
        you should see a parquet file saved there, and this task is considered complete
        """
        data = pd.read_csv(os.path.join(DATA_DIR, FILENAME))target_variable = "median_house_value"
        features = [col for col in data.columns if col != "median_house_value"]housing = data.loc[:, features]
        labels = pd.DataFrame(data.loc[:, target_variable])
        # save as parquet
        self.save({"training_data": housing, "labels": labels})
```

现在，我们需要注意一些事情。首先，它没有`requires`函数，也没有`@requires`装饰器。这就是我们将它标记为没有依赖关系的方式。二是继承自`d6tflow.tasksTaskPqPandas`。您的类继承的 d6tflow 任务是根据您希望保存输出的文件格式来确定的。因为我们想要像 pandas `DataFrame`一样加载我们的数据，并且我们将数据保存到 parquet，这告诉我们这是我们继承的类。第三，任务的动作总是包含在它的`run`方法中，通常以`self.save`结束，以保存任务的输出。最后，有必要声明您的任务作为类级成员将保存什么，这意味着就在`persist`中的`class TaskGetData():`行下面。另外，`persist`里面的名字需要和`self.save`里面的键匹配。现在，您可以通过运行以下命令来预览您的 DAG:

```
d6tflow.preview(TaskGetData())
# or
get_data_task = TaskGetData()
d6tflow.preview(get_data_task)
```

你运行它:

```
d6tflow.run(get_data_task)
```

现在，当你检查`d6tflow_output/`时，你应该看到另一个文件夹`TaskGetData`，包含两个 pickle 文件，一个是`training_data`，另一个是`labels`。这两个文件的存在标志着任务完成。如果你运行`d6tflow.preview(get_data_task)`，它会说完成，现在而不是待定。如果您删除这些拼花文件，将导致任务被注册为未完成。试试看！(删除文件并重新运行`d6tflow.preview`)

现在，进入我们的下一个任务-预处理。这段代码的绝大部分是上面我们创建管道的部分的精确副本。有几件事需要注意。首先，我选择使用`@requires`装饰器，以表明这个类依赖于`TaskGetData`类。我也可以定义一个返回`TaskGetData()`的`requires`方法。第二，这个类继承自`d6tflow.tasks.TaskPickle`。这表明我们将把这个任务的结果保存为 pickle 文件。继承导致`save`方法被覆盖，以将结果对象保存为 pickle 文件。第三，我们已经定义了`persist`来显示我们将要保存转换后的数据，以及 pickle 文件的管道(预处理后的数据应该保存在 parquet 中，我可以重构它，我只是认为它对于本文的目的来说不是太重要)。第四，这里有一点很重要，我们将`do_preprocess`和`categorical_column_name`定义为`luigi`参数。这允许将参数智能地传递给需要该任务的下游任务(类)，以及它们返回给上游的值。所以说我们有一个依赖于`TaskPreprocess`的类，叫它`TaskTrain`。通过用`@requires(TaskPreprocess)`来修饰`TaskTrain`，我们将能够编写`training_task = TaskTrain(do_preprocess=True)`，并且参数`do_preprocess`的值将被上行传递给`TaskPreprocess`。在这一点上，您可以看到这个功能减轻了在 DAG 中传递参数的相当大的麻烦。只要您在每个连续的下游任务中正确地标记了依赖关系，您就能够通过在最下游的任务中初始化它来在 DAG 中的任何位置设置任何参数，而无需任何额外的工作。Luigi 是另一个用于轻松管理任务和参数的流行库。我推荐你去[看一看。另一件要注意的事情是，如果你想访问参数，你需要把它当作一个成员变量。所以我们需要说`self.categorical_column_name`。最后，我们有熟悉的`save`方法，它使用一个字典，键匹配`persist`中的条目，字典中的值作为 pickle 文件保存在文件夹`d6tflow_output/TaskPreprocess`中。](https://github.com/spotify/luigi)

```
[@requires](http://twitter.com/requires)(TaskGetData)
class TaskPreprocess(d6tflow.tasks.TaskPickle):
    persist = ["processed_data", "pipeline"]
    categorical_column_name = luigi.Parameter(default="ocean_proximity")
    do_preprocess = luigi.BoolParameter(default=True)
    def run(self):
        # multiple dependencies, multiple outputs
        X = self.input()["training_data"].load()
        # when is loaded back from parquet, we need to convert this one to categorical again
        X[self.categorical_column_name] = X[self.categorical_column_name].astype("category")
        # get unique values in the categorical column
        categories = list(X[self.categorical_column_name].unique())
        # get column names for categorical and numerical columns separately
        numerical_features = list(
                X.select_dtypes(include=[np.number]).columns)
        categorical_features = list(
            X.select_dtypes(include=["category"]).columns)

        # we could split these transformations into different tasks if we wanted
        # really fine-grain control over keeping track of the progression of data
        numerical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                       ("featurization", Feature_Adder_FT),
                                       ("strd_scaler", StandardScaler())], verbose=True)
        categorical_pipeline = Pipeline(
            [("one_hot_encoding", OneHotEncoder(categories=[categories]))], verbose=True)full_pipeline = ColumnTransformer([("numerical_pipeline", numerical_pipeline, numerical_features),
                                    ("categorical_pipeline", categorical_pipeline, categorical_features)],
                                          verbose=True)
        if self.do_preprocess:
            number_categories = len(categories)
            featurized_feature_names = [
                "rooms_per_household", "population_per_household", "bedrooms_per_room"]
            categorical_feature_names = ["one_hot_{:d}".format(
                i) for i in range(number_categories)]
            # because our transformations add columns to our original matrix, need to manually
            # construct column names
            # our featurizer adds 3 new feature columns to the right side of original feature matrix
            # and the categorical columns are added to the right side because it comes after numerical
            # in the full pipeline
            column_names = list(X.drop(self.categorical_column_name,axis=1).columns) + \
                featurized_feature_names + categorical_feature_names
            preprocessed_data = pd.DataFrame(data=full_pipeline.fit_transform(X), \
                                             columns=column_names)

        self.save({"processed_data":preprocessed_data,"pipeline":full_pipeline})
```

另一件要提到的事情是，您传递的参数的具体值是检查任务是否完成的因素。如果你有参数，那么任务就必须用那些特定的参数来完成。因此，您可以使用不同的参数值运行任务的多个版本，并将它们都标记为已完成。

到目前为止，我们遇到的第一个障碍是，因为我们将预测器放在一个单独的任务中，破坏了原始的 scikit 学习管道，所以我们需要手动计算列名的正确顺序，因为我们需要将处理后的数据保存到文件中，以便我们可以将它加载到任务中来训练我们的模型。这导致了一些额外的代码，但是如果您理解您的转换在做什么，应该不难发现。我在这里是这样做的:因为数值管道在分类管道之前，所以我们需要先处理它。我们获取所有原始的数字列名(没有分类列名)，然后添加三个特征列名，因为这是管道中接下来要添加的内容。然后，我们在末尾添加分类列的名称，因为分类管道连接在数字列的末尾。

现在，我们将创建一个任务来训练我们的模型。注意这里我们使用带有两个参数的`@requires`装饰器。这仅仅意味着我们想要从它们两个中继承 luigi 参数，并且当我们创建一个`TaskTrain`的实例时，我们将能够通过`TaskTrain`的构造函数为这些任务中的任何一个提供 luigi 参数。像往常一样，我们从`TaskPickle`继承，因为我们将把我们得到的训练模型保存为 pickle 文件。另外，我们需要在`persist`列表中标出这个名字。现在，由于我们向`@requires`装饰器传递了两个类，我们需要一种方法来访问它们的输出。这是通过以列表形式访问`self.input()`来完成的。例如，为了获得`TaskPreprocess`的`processed_data`输出，我们使用`self.input()[0]["processed_data"].load()`。从左到右，该调用访问输入列表，选择第一个依赖项，然后获取该依赖项的输出(这是一个字典)，然后在该字典中选择`processed_data`键，最后我们用`load()`将 pickle 文件加载回内存。

```
[@requires](http://twitter.com/requires)(TaskPreprocess, TaskGetData)
class TaskTrain(d6tflow.tasks.TaskPickle):
    """
    you can pass do_preprocess=False and model="svm" to TaskTrain __init__
    """
    persist=["model"]
    model = luigi.Parameter(default='ols')def run(self):
        # even tho this was a smultiple dependency, single output, self.input()["processed_data"]
        # is still a dictionary, whose key-name is from persist inside TaskPreprocess
        training_data = self.input()[0]["processed_data"].load()
        labels = self.input()[1]["labels"].load()
        if self.model == "ols":
            model = LinearRegression()
        elif self.model == "svm":
            model = SVR()
        else:
            raise ValueError("invalid model selection")
        model.fit(training_data, labels)
        self.save({"model":model})
```

既然我们的模型已经保存到 pickle 文件中，我们可以加载它并运行一个交叉文件夹验证任务:

```
[@requires](http://twitter.com/requires)(TaskGetData, TaskPreprocess, TaskTrain) # allows me to pass model='svm' to this
class TaskCrossFoldValidation(d6tflow.tasks.TaskCache):
    persist=["cross_val_scores"]
    def requires(self):
        # use self.clone_parent so that when I pass model='svm' to this, it gets passed to TraskTrain
        return {"processed_data":TaskPreprocess(),"model":self.clone_parent()}def run(self):
        labels = self.input()[0]["labels"].load()
        training_data = self.input()[1]["processed_data"].load()
        model = self.input()[1]["model"].load()

        cross_val_scores = cross_val_score(model, training_data, labels, cv=10)

        self.save({"cross_val_scores":cross_val_scores})
```

最后，我将在这里描述的最后一种任务是`d6tflow.tasks.TaskAggregator`。这个任务只是产生许多其他任务对象。这是为了让您的任务保持模块化，并且在聚合器内部有某种循环，产生您想要运行的所有任务。因为循环的每一次迭代都会产生具有不同参数的不同任务实例，所以所有任务都将被单独标记为完成。如果生成的所有任务都已完成，则聚合器任务将被标记为完成。下面是一个使用`d6tflow.tasks.TaskAggregator`创建独立任务来管理计算不同模型的交叉验证分数的例子。当聚合器任务的两个组成任务对象被标记为完成时，该任务本身也被标记为完成。

```
class TaskAggregator(d6tflow.tasks.TaskAggregator):
    def run(self):
        yield TaskPrintCrossValScore(model="svm")
        yield TaskPrintCrossValScore(model="ols")
```

使用我们已经讨论过的不同任务类型，您只需像前面讨论的那样将它们链接在一起，然后运行`d6tflow.run(most_downstream_task)`来运行您的 DAG，或者运行`d6tflow.preview(most_downstream_task)`来预览不同部分的依赖关系和状态。

好了，这结束了我对使用 d6tflow 将 sklearn 管道包装在受管 DAG 中的端到端介绍。你可以在这里找到我这个项目[的笔记本。如果有任何不清楚的地方，或者你有任何问题，请随时告诉我！](https://nbviewer.jupyter.org/github/jcorrado76/sklearn_d6tflow_pipeline/blob/master/D6tflow%20or%20Sklearn%20Pipeline.ipynb)