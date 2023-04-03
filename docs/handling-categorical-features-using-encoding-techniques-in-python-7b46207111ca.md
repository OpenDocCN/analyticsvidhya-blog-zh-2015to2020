# 使用 Python 中的编码技术处理分类要素

> 原文：<https://medium.com/analytics-vidhya/handling-categorical-features-using-encoding-techniques-in-python-7b46207111ca?source=collection_archive---------4----------------------->

![](img/a7b3dbef2c043b3210d96e0a86188459.png)

马库斯·斯皮斯克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

在这篇文章中，我们将讨论机器学习中的分类特征，以及使用两种最有效的方法来处理这些特征的方法。

# **分类特征**

在机器学习中，特征可以大致分为两大类:

*   数字特征(年龄、价格、面积等。)
*   分类特征(性别、婚姻状况、职业等。)

由一定数量的类别组成的所有这些特征被称为分类特征。分类特征可以分为两种主要类型:

1.  名义上的
2.  序数

**名义**特征是指有两个或两个以上类别的特征，没有特定的顺序。比如性别有两个值，男性和女性，可以认为是一个名义上的特征。

**序数**另一方面，特征有特定顺序的类别。例如，如果我们有一个名为 Level 的特性，其值为 high、medium 和 low，那么它将被视为一个序数特性，因为这里的顺序很重要。

# 处理分类特征

因此，出现的第一个问题是，为什么我们需要分别处理分类特征？为什么我们不像数字特征一样简单地将它们作为输入传递给我们的模型呢？答案是，与人类不同，机器，尤其是机器学习模型，不理解文本数据。在将文本值输入模型之前，我们需要将它们转换成相关的数字。

将类别转换成数字的过程称为编码。最有效和最广泛使用的两种编码方法是:

1.  标签编码
2.  一个热编码

**标签编码**

标签编码是将数字标签分配给要素中每个类别的过程。如果 N 是类别数，所有类别值将被分配一个从 0 到 N-1 的唯一数字。

如果我们有一个名为“颜色”的特征，其值为红色、蓝色、绿色和黄色，它可以转换为数字映射，如下所示

```
**Category** : **Label**
"red"    : 0
"blue"   : 1
"green"  : 2
"yellow" : 3
```

**注意:**正如我们在这里看到的，为类别生成的标签不是标准化的，即不在 0 和 1 之间。由于这一限制，标注编码不应用于要素量起着重要作用的线性模型。由于基于树的算法不需要特征归一化，标签编码可以容易地用于这些模型，例如:

*   决策树
*   随机森林
*   XGBoost
*   LighGBM

我们可以使用 scikit-learn 的 LabelEncoder 类实现标签编码。我们将在下一节看到实现。

**一个热编码**

标签编码的限制可以通过二进制化类别来克服，即仅使用 0 和 1 来表示类别。这里，我们通过大小为 N 的向量来表示每个类别，其中 N 是该特征中类别的数量。每个向量有一个 1，其余所有值都是 0。因此，它被称为一热编码。

假设我们有一个名为 temperature 的列。它有四个值:冷、冷、温、热。每个类别将表示如下:

```
**Category**        E**ncoded vector** Freezing        0  0  0  1Cold            0  0  1  0Warm            0  1  0  0Hot             1  0  0  0
```

正如您在这里看到的，每个类别由一个长度为 4 的向量表示，因为 4 是特性中唯一类别的数量。每个向量只有一个 1，其余所有值都是 0。

由于一键编码可生成归一化要素，因此可用于线性模型，例如:

*   线性回归
*   逻辑回归

现在我们已经对这两种编码技术有了基本的了解，为了更好的理解，让我们看看这两种编码技术的 python 实现。

# 用 Python 实现

在对分类特征应用编码之前，处理 NaN 值很重要。一种简单有效的方法是将 NaN 值作为一个单独的类别来处理。通过这样做，我们可以确保不会丢失任何重要信息。

因此，我们在处理分类特征时遵循的步骤是:

1.  用新的类别填充 NaN 值(例如 NONE)
2.  对基于树的模型使用标签编码，对线性模型使用 hot 编码，将类别转换为数值。
3.  使用数字和编码要素构建模型。

我们将在 kaggle 的 Dat 中使用一个名为**猫的公共数据集。链接[这里](https://www.kaggle.com/c/cat-in-the-dat-ii)。这是一个包含大量分类特征的二元分类问题。**

首先，我们将使用 scikit-learn 中的 StratifiedKFold 类创建 5 个折叠进行验证。KFold 的这种变体用于确保每个折叠中目标变量的比率相同。

```
import pandas as pd
from sklearn import model_selection#read training data
df = pd.read_csv('../input/train.csv')#create column for kfolds and fill it with -1
df['kfold'] = -1#randomize the rows
df = df.sample(frac=1).reset_index(drop=True)#fetch the targets
y = df['target'].values#initiatre StratifiedKFold class from model_selection
kf = model_selection.StratifiedKFold(n_splits=5)#fill the new kfold column 
for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
    df.loc[v_,'kfold'] = f#save the new csv with kfold column
df.to_csv('../input/train_folds.csv',index=False)
```

**标签编码**

接下来，让我们定义在每个文件夹上运行训练和验证函数。在这个例子中，我们将对随机森林使用 LabelEncoder。

```
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessingdef run(fold):
    #read training data with folds
    df = pd.read_csv('../input/train_folds.csv') #get all relevant features excluding id, target and kfold columns
    features = [feature for feature in df.columns if feature not in ['id','target','kfold']] #fill all nan values with NONE
    for feature in features:
        df.loc[:,feature] = df[feature].astype(str).fillna('NONE') #Label encoding the features
    for feature in features:
        #initiate LabelEncoder for each feature
        lbl = preprocessing.LabelEncoder() #fit the label encoder
        lbl.fit(df[feature]) #transform data
        df.loc[:,feature] = lbl.transform(df[feature]) #get training data using folds
    df_train = df[df['kfold']!=fold].reset_index(drop=True)

    #get validation data using folds
    df_valid = df[df['kfold']==fold].reset_index(drop=True) #get training features
    X_train = df_train[features].values

    #get validation features
    X_valid = df_valid[features].values #initiate Random forest model
    model = ensemble.RandomForestClassifier(n_jobs=-1) #fit the model on train data
    model.fit(X_train,df_train['target'].values) #predict the probabilities on validation data
    valid_preds = model.predict_proba(X_valid)[:,1] #get auc-roc score
    auc = metrics.roc_auc_score(df_valid['target'].values,valid_preds) #print AUC score for each fold
    print(f'Fold ={fold}, AUC = {auc}')
```

最后，让我们调用这个方法来为每个文件夹执行 run 方法。

```
if __name__=='__main__':
    for fold_ in range(5):
        run(fold_)
```

执行这段代码将产生如下所示的输出。

```
Fold =0, AUC = 0.7163772816343564
Fold =1, AUC = 0.7136206487083182
Fold =2, AUC = 0.7171801474337066
Fold =3, AUC = 0.7158938474390842
Fold =4, AUC = 0.7186004462481813
```

这里需要注意的一点是，我们没有对随机森林模型进行任何超参数调整。您可以调整参数来提高验证的准确性。在上面的代码中要提到的另一件事是，我们使用 AUC ROC 分数作为验证的度量。这是因为目标值是有偏差的，准确性等指标不会给我们正确的结果。

**一个热编码**

现在让我们来看一个带有逻辑回归的热编码的实现。

下面是这种方法的 run 方法的修改版本。

```
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessingdef run(fold):
    #read training data with folds
    df = pd.read_csv('../input/train_folds.csv') #get all relevant features excluding id, target and folds columns
    features = [feature for feature in df.columns if feature not in ['id','target','kfold']] #fill all nan values with NONE
    for feature in features:
        df.loc[:,feature] = df[feature].astype(str).fillna('NONE') #get training data using folds
    df_train = df[df['kfold']!=fold].reset_index(drop=True)

    #get validation data using folds
    df_valid = df[df['kfold']==fold].reset_index(drop=True) #initiate OneHotEncoder from sklearn
    ohe = preprocessing.OneHotEncoder() #fit ohe on training+validation features
    full_data = pd.concat([df_train[features],df_valid[features]],axis=0)
    ohe.fit(full_data[features]) #transform training data
    X_train = ohe.transform(df_train[features])

    #transform validation data
    X_valid = ohe.transform(df_valid[features]) #initiate logistic regression
    model = linear_model.LogisticRegression() #fit the model on train data
    model.fit(X_train,df_train['target'].values) #predict the probabilities on validation data
    valid_preds = model.predict_proba(X_valid)[:,1] #get auc-roc score
    auc = metrics.roc_auc_score(df_valid['target'].values,valid_preds) #print AUC score for each fold
    print(f'Fold ={fold}, AUC = {auc}')
```

循环所有折叠的方法保持不变。

```
if __name__=='__main__':
    for fold_ in range(5):
        run(fold_)
```

这段代码的输出如下所示:

```
Fold =0, AUC = 0.7872262099199782
Fold =1, AUC = 0.7856877416085041
Fold =2, AUC = 0.7850910855093067
Fold =3, AUC = 0.7842966593706009
Fold =4, AUC = 0.7887711592194284
```

正如我们在这里看到的，一个简单的逻辑回归通过对分类特征应用特征编码给了我们相当好的准确性。

在这两种方法的实现中需要注意的一个区别是，LabelEncoder 必须分别适用于每个分类特征，而 OneHotEncoder 可以适用于所有特征。

# 结论

在这篇博客中，我讨论了机器学习中的分类特征，为什么处理这些特征很重要。我们还讨论了将分类特征编码成数字的两种最重要的方法，以及实现。

我希望我已经帮助您更好地理解了这里涉及的主题。请让我知道你在评论中的反馈，如果你喜欢就给它一个掌声。[这里的](https://www.linkedin.com/in/sawan-saxena-640a4475/)是我的 Linkedin 个人资料的链接，如果你想连接的话。

感谢阅读。:)