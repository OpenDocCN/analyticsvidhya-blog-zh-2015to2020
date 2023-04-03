# 自定义估算器—为什么和如何

> 原文：<https://medium.com/analytics-vidhya/a-custom-imputer-why-and-how-8076773a23ba?source=collection_archive---------22----------------------->

![](img/a475b888cc1700ca85490c78c1b35009.png)

今年早些时候，我们的三人数据科学团队聚集在一个小会议室，查看描述新模型特征行为的图表。我们在考虑生产一个新的模型，对我们来说，这个过程中的一个关键步骤就是戳戳这个模型，看看它是如何工作的。在这种情况下，这意味着为每个特性绘制部分依赖图，并检查每个特性的行为有多明智。

盯着这些图，有一件事很快变得很明显:某些分类特征表现得很奇怪。一些标签编码的分类特征在其估算值处具有很强的模型响应，导致其 PDP 图中出现奇怪的起伏。在聊了一下这些意味着什么以及该怎么做之后，我们决定通过修改我们的插补策略来研究这些特征到底是怎么回事。我们不是对每个特征进行均值或中值估算，而是对所有分类特征进行-1 值估算，看看新的部分相关图是什么样的。我们认为，如果强模型响应随着估算值的变化而变化，这将是一个明确的信号，表明模型依赖于缺失值的存在，而不是直觉地与特征值有关系。

就这样，我们结束了会议。现在是时候回到笔记本电脑上，开始编写一个自定义的估算器来处理这些分类变量了！

我想做的第一件事是看看 Sci-kit Learn 的`SimpleImputer`背后的代码。我对两件事很好奇:他们如何存储计算出的值，以备将来插补？他们在输入时如何应用这些存储的值？

是时候深入一些 SimpleImputer 的源代码了。通读[的源代码](https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/impute/_base.py#L84)，我能够回答我脑海中的两个问题。估算值作为 numpy 数组存储在`statistics`属性中，并通过布尔掩码应用于空值。

考虑到这一点，我们想到了一个快速实现估算器的设计。我们需要一种方法来识别分类特征，并将这些特征的存储估算值更改为-1。

首先，让我们看看我们能做些什么来识别分类特征。这对于自动化来说有点困难，但是分类特征背后的主要思想是存在一定数量的可能值来填充该特征。如果我们假设有一个大型数据集，我们可以做一些事情，比如检查由唯一值组成的要素数据的百分比。编程到 Python 中，该函数看起来会像这样( [source here](https://github.com/StatesTitle/ds-blog/blob/master/custom_imputer/dtype_imputer.py) ):

```
def is_categorical(array, percent_unique_cutoff = 0.1):
    test_array = array[~np.isnan(array)]
    not_int = (test_array.astype(int) != test_array).sum()
    if not_int:
        return False
    percent_unique = len(np.unique(test_array)) / len(array)
    return percent_unique < percent_unique_cutoff
```

既然我们有了一种识别分类特征的方法，我们需要一种将它集成到估算器中的方法。正如我们之前提到的，在拟合估算值时，估算值作为 numpy 数组存储在`statistics`属性中。最容易实现的是修改`statistics`数组，并将识别的分类特征的估算值更改为我们想要的值-1。具体来说，我们将创建一个从`SimpleImputer`继承的新类，并修改`__init__`和`fit`方法来包含这个新功能。

这在代码中是什么样子的:

```
import numpy as np
from sklearn.impute._base import _get_mask
from sklearn.impute import SimpleImputerclass DTypeImputer(SimpleImputer):def __init__(self, missing_values=np.nan, strategy="mean",
                 fill_value=None, verbose=0, copy=True, add_indicator=False,
                categorical_fill_value = -1):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose
        self.copy = copy
        self.add_indicator = add_indicator
        self.categorical_fill_value = categorical_fill_value

    def fit(self, X, y=None):
        """Fit the imputer on X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        Returns
        -------
        self : SimpleImputer
        """
        X = self._validate_input(X)# default fill_value is 0 for numerical input and "missing_value"
        # otherwise
        if self.fill_value is None:
            if X.dtype.kind in ("i", "u", "f"):
                fill_value = 0
            else:
                fill_value = "missing_value"
        else:
            fill_value = self.fill_value# fill_value should be numerical in case of numerical input
        if (self.strategy == "constant" and
                X.dtype.kind in ("i", "u", "f") and
                not isinstance(fill_value, numbers.Real)):
            raise ValueError("'fill_value'={0} is invalid. Expected a "
                             "numerical value when imputing numerical "
                             "data".format(fill_value))if sparse.issparse(X):
            # missing_values = 0 not allowed with sparse data as it would
            # force densification
            if self.missing_values == 0:
                raise ValueError("Imputation not possible when missing_values "
                                 "== 0 and input is sparse. Provide a dense "
                                 "array instead.")
            else:
                self.statistics_ = self._sparse_fit(X,
                                                    self.strategy,
                                                    self.missing_values,
                                                    fill_value)
        else:
            self.statistics_ = self._dense_fit(X,
                                               self.strategy,
                                               self.missing_values,
                                               fill_value)if self.add_indicator:
            self.indicator_ = MissingIndicator(
                missing_values=self.missing_values)
            self.indicator_.fit(X)
        else:
            self.indicator_ = None

        mask = _get_mask(X, np.nan)
        masked_X = np.ma.masked_array(X, mask=mask)
        categorical_mask = np.apply_along_axis(is_categorical, 0, masked_X)
        self.statistics_[categorical_mask] = self.categorical_fill_value

        return self
```

代码实现后，我们使用这个新的估算器来生成另一组部分依赖图。通过观察这些数据，我们发现，模型的强烈响应是随着估算值的变化而变化的，这告诉我们，无论该特征是否有缺失数据，模型都具有预测能力。在某些情况下，最强有力的信号是我们是否有数据！

# 附言

回顾我一年前写的代码，我惊讶于我从中学到了多少，以及它是多么的不必要。如果我更聪明，我会手动识别分类特征，并使用 Sci-kit Learn 的 ColumnTransformer 对连续和分类特征进行单独插补。

然而，与此同时，我不会像今天这样熟悉归罪。总而言之，这是我很高兴经历的一次锻炼。

*最初发表于*[*http://github.com*](https://gist.github.com/b9ab868d8d378ad64f5261c836fef5e6)*。*