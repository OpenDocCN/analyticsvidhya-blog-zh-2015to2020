# 用于分类数据回归的 Lightgbm。

> 原文：<https://medium.com/analytics-vidhya/lightgbm-for-regression-with-categorical-data-b08eaff501d1?source=collection_archive---------1----------------------->

在这方面，我们将

*   了解 LGBM
*   在 kaggle 数据集上实现
*   利弊

# 了解 LGBM

我们都知道，梯度推进技术是用数据进行实验所必须的，它们也能给出很好的结果。他们如此受欢迎，我们不能忽视他们。但是梯度增强树在准确性和效率之间进行了权衡。因为它们需要扫描整个数据来计算所有可能分裂点的信息增益。因此计算时间随着特征数量的增加而增加。所以用这种方法处理大数据非常困难。一种方法是丢弃低梯度变化的特征，但我们会冒准确性的风险，因此在 LGBM 中，我们进行**基于梯度的单侧采样(GOSS)。**

**基于梯度的单侧采样(GOSS)**

在这种情况下，我们将保留具有大梯度特征，并将随机选择具有小梯度变化的特征。

**寻找最佳分割点(基于直方图)**。

提升树中的另一个非常耗时的部分是找到最佳分裂以减少这种情况，基于直方图的算法将连续的特征值存储到离散的箱中，并在训练期间使用这些箱来构建特征直方图，而不是在排序的特征值上找到分裂点。这在内存和速度上都是有效的。

所有这些使得 LGBM 速度很快，因此被命名为 LightGBM，因为它可以在大数据上操作，并且效率高。

**垂直生长**

它垂直生长，因为它会检测到具有较高数据丢失的叶子，然后在其上生长，而不是以其他方式水平生长，即在每片叶子上生长。更多信息请访问[这里](/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)。垂直生长也使它变得很快。

# 在 kaggle 数据集上实现

我们已经从 kaggle 下载了数据，并在本教程的[中清理了数据。就这样，直到删除空值。我会告诉你 LGBM 有多酷，它如何处理分类特征。](/@rajanlagah/data-cleaning-for-machine-learning-algorithms-to-eat-328858114fbc)

```
import pandas as pd 
import numpy as np
import lightgbm as lgb
```

如果您没有安装 lightgbm

```
pip install lightgbm
```

我已经保存了该教程中的数据，并将测试和培训结合在一起

```
data = pd.read_csv('./train_test_nullFill_break_1460.csv')
```

让我们看看所有不连续的特征

```
obj_feat = list(data.loc[:, data.dtypes == 'object'].columns.values)
obj_feat
```

它将输出

```
['MSZoning','Street',....]
```

但是花了我一天时间解决的恼人的部分是 lgbm 不接受对象格式或字符串格式的分类数据，你必须将它们转换成**分类**类型。因此

```
for feature in obj_feat:
    data[feature] = pd.Series(data[feature], dtype="category")
```

现在

```
train_df = pd.read_csv('./train.csv') # just forgot to save y values
X_train = data[:1260]
y_train = train_df['SalePrice'][:1260]
X_valid = data[1260:1460]
y_valid = train_df['SalePrice'][1260:]
test_df = data[1460:]
```

现在开始建模

```
hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l2', 'auc'],
    'learning_rate': 0.005,
    "num_leaves": 128,  
    "max_bin": 512,
}lgb_train = lgb.Dataset(X_train, y_train)
gbm = lgb.train(hyper_params, lgb_train, num_boost_round=10, verbose_eval=False)
```

这就是使用它是多么简单。不需要处理分类变量。现在让我们保存预测并提交给 kaggle

```
preds_3 = gbm0.predict(test_df)
pred = pd.DataFrame(preds_3)
sub_df=pd.read_csv('sample_submission.csv')datasets = pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns = ["Id",'SalePrice']datasets.to_csv('lgbm_v1.csv',index=False)
```

我们会把它提交给卡格尔。你可以调整得更精确，我刚刚改变了我使用的参数，你必须继续努力。

# 利弊

**优点**

*   快的
*   你不需要担心分类变量(如果你愿意，你可以)
*   它给出了很好结果

**缺点**

*   它不适合小数据集，因为它会过拟合
*   它有很多可以玩的参数

如果你读到这里，我希望你能得到一些好的东西。如果您有任何建议，我们将乐于在 rajanlagah@gmail.com**的邮件中听到**