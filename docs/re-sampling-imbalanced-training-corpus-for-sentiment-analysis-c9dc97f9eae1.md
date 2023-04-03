# 面向情感分析的不平衡训练语料重采样

> 原文：<https://medium.com/analytics-vidhya/re-sampling-imbalanced-training-corpus-for-sentiment-analysis-c9dc97f9eae1?source=collection_archive---------0----------------------->

在本文中，我将讨论过采样、欠采样和两者结合的技术，以平衡不平衡的训练数据集，解决在线 Twitter 情感分析的挑战。该挑战旨在检测人工判断的测试语料库中的仇恨和辱骂言论。包括培训和测试数据在内的挑战在此处[可用。](https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/)

![](img/9fc7517c6a0c793c6ddeae8fa335a947.png)

对不平衡数据集进行重新采样肯定会提高分类效果

训练语料库包含人工判断的推文，那些被认为是滥用的推文被标记为 1，其他正常推文被标记为 0。

让我们首先导入必要的库来继续:

```
#import the necessary libraries for dataset preparation, feature engineering, model training
from sklearn import model_selection, preprocessing, metrics, linear_model, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN, SMOTENC, RandomOverSampler
from imblearn.under_sampling import (RandomUnderSampler, 
                                    NearMiss, 
                                    InstanceHardnessThreshold,
                                    CondensedNearestNeighbour,
                                    EditedNearestNeighbours,
                                    RepeatedEditedNearestNeighbours,
                                    AllKNN,
                                    NeighbourhoodCleaningRule,
                                    OneSidedSelection,
                                    TomekLinks)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline
import pandas as pd, numpy, string
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
#Remove Special Charactors
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
```

现在，我们需要导入培训和测试 csv 文件，并查看部分文本:

```
#Import Training and Testing Data
train = pd.read_csv('train.csv')
print("Training Set:"% train.columns, train.shape, len(train))
test = pd.read_csv('test_tweets.csv')
print("Test Set:"% test.columns, test.shape, len(test))Training Set: (31962, 3) 31962
Test Set: (17197, 2) 17197
```

训练集由 32K tweets 组成，测试集由 17K tweets 组成。让我们来探索一下这两组的最高记录:

```
train.head()
id      label   tweet
1       0       @user when a father is dysfunctional and is                2       0       @user @user thanks for #lyft credit i can't                 3       0       bihday your majesty                 
4       0       #model   i love u take with u all the time i               5       0       factsguide: society now    #motivation\test.head()
id          tweet
31963       #studiolife #aislife #requires #passion #dedic                 31964       @user #white #supremacists want everyone to s...                 31965       safe ways to heal your #acne!! #altwaystohe...                 31966       is the hp and the cursed child book up for res.              31967       3rd #bihday to my amazing, hilarious #nephew...
```

如果您注意到语料库取代了对@ user 用户名的提及，文本还包括#标签、特殊字符和数字。让我们看看在训练语料库中辱骂推文占正常推文的百分比:

```
#Percentage of Positive/Negative
print("Positive: ", train.label.value_counts()[0]/len(train)*100,"%")
print("Negative: ", train.label.value_counts()[1]/len(train)*100,"%")
```

负面推文占数据集的 93%，负面占 7%:

```
Positive:  92.98542018647143 %
Negative:  7.014579813528565 %
```

这意味着我们有不平衡的训练数据集，这肯定会影响预测的准确性。为了使这个论点有效，我们将首先通过分割判断的训练数据来预测负面和正面的推文，然后将预测的标签与人类判断的标签进行比较。但在我们开始之前，让我们清除 tweets 中的数字、html/xml 标签、特殊字符、空格以及词干，这些都在下面的代码块中完成:

```
porter=PorterStemmer()
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    #Stemming
    stem_sentence=[]
    for word in words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    words="".join(stem_sentence).strip()
    return words
nums = [0,len(train)]
clean_tweet_texts = []
for i in range(nums[0],nums[1]):
    clean_tweet_texts.append(tweet_cleaner(train['tweet'][i]))
nums = [0,len(test)]
test_tweet_texts = []
for i in range(nums[0],nums[1]):
    test_tweet_texts.append(tweet_cleaner(test['tweet'][i])) 
train_clean = pd.DataFrame(clean_tweet_texts,columns=['tweet'])
train_clean['label'] = train.label
train_clean['id'] = train.id
test_clean = pd.DataFrame(test_tweet_texts,columns=['tweet'])
test_clean['id'] = test.id
```

让我们将训练数据集分为训练和测试:

```
#split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_clean['tweet'],train_clean['label'])
#label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
```

在前面的代码中，请注意，我必须将标签列转换为一个数组，因为我现在将使用术语频率 TF 和逆文档频率 IDF 将推文转换为术语的加权向量，在这里阅读更多关于它们的信息。

我们有不止一个选项来评估术语的重要性，最好的方法之一是使用 TFIDF，为了便于讨论，您可以尝试使用 bag of words BOW。

```
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100000)
tfidf_vect.fit(train_clean['tweet'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
```

# 评估指标

准确性是一个很好的衡量标准，但只有当你有平衡的数据集，其中标签 0 和 1 被几乎相等地分解。因此，我们需要看看其他参数来评估模型的性能。在我们的模型中，我们可以得到很高的准确率，因为 0 约占语料库的 93%。

对于这个模型，我使用了 **F1 得分**——这是精确度和召回率的加权平均值。

*F1 得分= 2*(召回率*精确度)/(召回率+精确度)*

有关评估技术的更多详情，请点击查看[。](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)

```
#Return the f1 Score
def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)return metrics.f1_score(valid_y,predictions)
```

# 分类器

我将使用线性分类器(逻辑回归 LR)和支持向量机(SVM)分类器来预测负面和正面的推文。LR 通过使用 logistic/sigmoid 函数估计概率来衡量分类因变量与一个或多个自变量之间的关系。

SVM 算法寻找在 N 维空间(N-特征的数量)中具有最大余量的超平面，该超平面清楚地分类数据点。更多关于这两种分类器的信息可以在[这里](https://towardsdatascience.com/support-vector-machine-vs-logistic-regression-94cc2975433f)找到。

让我们看看基线结果(没有重新采样):

```
accuracyORIGINAL = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),xtrain_tfidf, train_y, xvalid_tfidf)
print ("LR Baseline, WordLevel TFIDF: ", accuracyORIGINAL)
accuracyORIGINAL = train_model(svm.LinearSVC(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("SVM Baseline, WordLevel TFIDF: ", accuracyORIGINAL)#LR Baseline, WordLevel TFIDF:  0.534131736526946
#SVM Baseline, WordLevel TFIDF:  0.6991701244813279
```

# 重新采样训练语料

平衡训练数据集有许多方法，我们将在本文中讨论:欠采样、过采样以及过采样和欠采样相结合。

**欠采样**通过减少多数类的大小来平衡数据集。当数据量足够时，使用这种方法。通过将所有样本保留在少数类中并在多数类中随机选择相等数量的样本。在我们的模型中，正面和负面的推文之间有巨大的差距，只有 7%的推文是负面的，所以欠采样可能不会产生很大的结果。我们将在本文中看到一些场景。

**过采样**在数据量不足时使用。它试图通过增加少数样本的大小来平衡数据集。不是去除多数样本，而是通过使用:重复、自举、 [SMOTE](https://arxiv.org/pdf/1106.1813.pdf) (合成少数过采样技术)或 [ADASYN](https://scinapse.io/papers/2104933073) (自适应合成采样)来生成新的少数样本。

将这两种方法结合起来是对数据集进行重采样的另一种方法。选择重采样技术取决于模型本身，对于我们的模型，我们将评估这三种方法。

# 随机过采样

下面的代码块使用 Scikit learn 过采样方法随机重复一些少数样本，并在数据集之间平衡样本数量。

```
#Random Over Sampling
ros = RandomOverSampler(random_state=777)
ros_xtrain_tfidf, ros_train_y = ros.fit_sample(xtrain_tfidf, train_y)accuracyROS = train_model(linear_model.LogisticRegression(random_state=0, solver=’lbfgs’,multi_class=’multinomial’),ros_xtrain_tfidf, ros_train_y, xvalid_tfidf)
print (“LR ORIGINAL, WordLevel TFIDF: “, accuracyROS)
accuracyROS = train_model(svm.LinearSVC(),ros_xtrain_tfidf, ros_train_y, xvalid_tfidf)
print ("SVM ROS, WordLevel TFIDF: ", accuracyROS)#LR ROS, WordLevel TFIDF:  0.6822066822066821
#SVM ROS, WordLevel TFIDF:  0.6995744680851064
```

从 f1 的分数结果可以清楚地看出，用这种简单的方法已经不太偏向于多数阶级了。但是如果我们把它应用到真实的测试数据中，我们会得到这个结果吗？你可以自己试试。

# SMOTE 过采样

SMOTE 是一种合成少数民族过采样方法，其中通过创建“合成”样本而不是从现有少数民族类创建新的随机少数民族样本来对少数民族类进行过采样。

```
#SMOTE
sm = SMOTE(random_state=777, ratio = 1.0)
sm_xtrain_tfidf, sm_train_y = sm.fit_sample(xtrain_tfidf, train_y)accuracySMOTE = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),sm_xtrain_tfidf, sm_train_y, xvalid_tfidf)
print ("LR SMOTE, WordLevel TFIDF: ", accuracySMOTE)
accuracySMOTE = train_model(svm.LinearSVC(),sm_xtrain_tfidf, sm_train_y, xvalid_tfidf)
print ("SVC SMOTE, WordLevel TFIDF: ", accuracySMOTE)#LR SMOTE, WordLevel TFIDF:  0.6848436246992782
#SVC SMOTE, WordLevel TFIDF:  0.693288020390824
```

我们可以看到，SMOTE 的得分略高于 ROS，但这是针对训练/测试数据的，但对于真实的测试数据，这可能会发生，请使用 test_clean 数据集尝试一下。

# ADASYN:自适应合成采样(过采样)

ADASYN 相对于 SMOTE 的优势在于根据不同少数类样本的学习难度对它们使用加权分布，其中与那些较容易学习的少数类样本相比，为较难学习的少数类样本生成更多的合成数据。更多详情请点击[这里](https://scinapse.io/papers/2104933073)。

```
#ADASYN
ad = ADASYN(random_state=777, ratio = 1.0)
ad_xtrain_tfidf, ad_train_y = ad.fit_sample(xtrain_tfidf, train_y)
accuracyADASYN = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),ad_xtrain_tfidf, ad_train_y, xvalid_tfidf)
print ("LR ADASYN, WordLevel TFIDF: ", accuracyADASYN)
accuracyADASYN = train_model(svm.LinearSVC(),ad_xtrain_tfidf, ad_train_y, xvalid_tfidf)
print ("SVM ADASYN, WordLevel TFIDF: ", accuracyADASYN)#LR ADASYN, WordLevel TFIDF:  0.6666666666666666
#SVM ADASYN, WordLevel TFIDF:  0.6818181818181819
```

f1 得分结果再次低于之前的两种方法，但这仍然是在训练/测试语料库上。

在之前的 3 个过采样示例中，`[RandomOverSampler](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.RandomOverSampler.html#imblearn.over_sampling.RandomOverSampler)`通过复制少数类的一些原始样本来过采样，而`[SMOTE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html#imblearn.over_sampling.SMOTE)`和`[ADASYN](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.ADASYN.html#imblearn.over_sampling.ADASYN)`通过插值在语料库中生成新样本。然而，用于生成新合成样本的样本是不同的。`[ADASYN](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.ADASYN.html#imblearn.over_sampling.ADASYN)`方法着重于生成与使用 k-最近邻分类器(难以学习)错误分类的原始样本相邻的样本，而`[SMOTE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html#imblearn.over_sampling.SMOTE)`的基本实现不会对使用最近邻规则分类的简单和困难学习样本进行任何区分。

值得一提的是，SMOTE 可能会连接内点和外点，因此这种方法提供了三个额外的选项来生成样本。这些方法侧重于最佳决策函数边界附近的样本，并将生成与最近邻类方向相反的样本。我实验过`BorderlineSMOTE`和`SMOTENC.`

**SMOTE-NC** 通过合成新的少数样本来使用 SMOTE 方法，但通过执行针对分类特征的特定操作来略微改变新样本的生成方式。事实上，新生成样本的类别是通过挑选在生成期间出现的最频繁类别的最近邻来决定的:

```
#SMOTENC
smnc = SMOTENC(categorical_features=[0, 2], random_state=0)
smnc_xtrain_tfidf, smnc_train_y = smnc.fit_sample(xtrain_tfidf, train_y)
accuracySMOTENC = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),smnc_xtrain_tfidf, smnc_train_y, xvalid_tfidf)
print ("LR SMOTENC, WordLevel TFIDF: ", accuracySMOTENC)
accuracySMOTENC = train_model(svm.LinearSVC(),smnc_xtrain_tfidf, smnc_train_y, xvalid_tfidf)
print ("SVM SMOTENC, WordLevel TFIDF: ", accuracySMOTENC)#LR SMOTENC, WordLevel TFIDF:  0.534131736526946
#SVM SMOTENC, WordLevel TFIDF:  0.6978193146417445
```

**边界线** SMOTE 将使用处于危险(弱判断)的样本生成新样本；

```
#Borderline SMOTE
bsm = BorderlineSMOTE()
bsm_xtrain_tfidf, bsm_train_y = bsm.fit_sample(xtrain_tfidf, train_y)
accuracyBSMOTE = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),bsm_xtrain_tfidf, bsm_train_y, xvalid_tfidf)
print ("LR Borderline SMOTE, WordLevel TFIDF: ", accuracyBSMOTE)
accuracyBSMOTE = train_model(svm.LinearSVC(),bsm_xtrain_tfidf, bsm_train_y, xvalid_tfidf)
print ("SVM Borderline SMOTE, WordLevel TFIDF: ", accuracyBSMOTE)#LR Borderline SMOTE, WordLevel TFIDF:  0.6799999999999999
#SVM Borderline SMOTE, WordLevel TFIDF:  0.6947368421052631
```

SMOTE-NC 和 Borderline SMOTE 似乎提高了两种分类器的性能，但大大提高了 LR。

# 随机欠采样

随机欠采样是一种受控欠采样方法，其中可以定义要选择的样本数量。这种方法随机减少了多数类以取得平衡，让我们看看评价:

```
# Random Under Sampling
rus = RandomUnderSampler(random_state=0, replacement=True)
rus_xtrain_tfidf, rus_train_y = rus.fit_sample(xtrain_tfidf, train_y)
accuracyrus = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),rus_xtrain_tfidf, rus_train_y, xvalid_tfidf)
print ("LR RUS, WordLevel TFIDF: ", accuracyrus)
accuracyrus = train_model(svm.LinearSVC(),rus_xtrain_tfidf, rus_train_y, xvalid_tfidf)
print ("SVC RUS, WordLevel TFIDF: ", accuracyrus)#LR RUS, WordLevel TFIDF:  0.5045045045045046
#SVC RUS, WordLevel TFIDF:  0.5091093117408907
```

所以 f1 分数低于所有使用的过采样方法。让我们试试别的东西。

# 差点错过

NearMiss 方法是[不平衡学习](https://imbalanced-learn.readthedocs.io/en/stable/index.html)库的一部分，NearMiss 执行启发式规则以选择样本。它根据多数类中样本与同一类中其他样本的距离，对这些样本执行欠采样。这种方法使用三种不同的变体:

**NearMiss-1** 保留来自多数类的样本，对于这些样本，少数类的 *k* ( [可调超参数](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)))最近样本的平均距离是最小的。

**NearMiss-2** 保留来自多数类的样本，其到少数类中 *k* 最远样本的平均距离最低。

**NearMiss-3** 为少数类中的每个样本选择 *k* 多数类中的最近邻居。所以欠采样率由 *k* 直接控制，不单独调整。

```
#NearMiss
for sampler in (NearMiss(version=1),NearMiss(version=2),NearMiss(version=3)):
    nm_xtrain_tfidf, nm_train_y = sampler.fit_sample(xtrain_tfidf, train_y)
    accuracysm = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),nm_xtrain_tfidf, nm_train_y, xvalid_tfidf)
    print ("LR NearMiss(version= {0}), WordLevel TFIDF: ".format(sampler.version), accuracysm)

for sampler in (NearMiss(version=1),NearMiss(version=2),NearMiss(version=3)):
    nm_xtrain_tfidf, nm_train_y = sampler.fit_sample(xtrain_tfidf, train_y)
    accuracysm = train_model(svm.LinearSVC(),nm_xtrain_tfidf, nm_train_y, xvalid_tfidf)
    print ("SVC NearMiss(version= {0}), WordLevel TFIDF: ".format(sampler.version), accuracysm)#LR NearMiss(version= 1), WordLevel TFIDF:  0.2734271303424476
#LR NearMiss(version= 2), WordLevel TFIDF:  0.5236625514403291
#LR NearMiss(version= 3), WordLevel TFIDF:  0.5541591861160982
#SVC NearMiss(version= 1), WordLevel TFIDF:  0.3597020219936148
#SVC NearMiss(version= 2), WordLevel TFIDF:  0.5224948875255624
#SVC NearMiss(version= 3), WordLevel TFIDF:  0.5594149908592323
```

正如你所看到的，NearMiss-3 的性能超过了其他两个版本。但是仍然低于过采样结果。

# Tomek 链接删除

如果一对样本属于不同的类并且是彼此最近的邻居，则称它们为 Tomek 链接。欠采样可以通过从数据集中移除所有 tomek 链接来完成。另一种方法是仅移除作为 Tomek 链接一部分的多数类样本。 [*参考 1*](https://imbalanced-learn.readthedocs.io/en/stable/under_sampling.html) *，* [*参考 2*](https://arxiv.org/pdf/1608.06048.pdf)

```
# Under-Sampling TomekLinks
tl = TomekLinks()
tl_xtrain_tfidf, tl_train_y = tl.fit_sample(xtrain_tfidf, train_y)
accuracy = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),tl_xtrain_tfidf, tl_train_y, xvalid_tfidf)
print ("LR TomekLinks, WordLevel TFIDF: ", accuracy)
accuracy = train_model(svm.LinearSVC(),tl_xtrain_tfidf, tl_train_y, xvalid_tfidf)
print ("SVC TomekLinks, WordLevel TFIDF: ", accuracy)#LR TomekLinks, WordLevel TFIDF:  0.5358851674641147
#SVC TomekLinks, WordLevel TFIDF:  0.6998961578400832
```

看起来，从不同的类中移除最近邻确实改进了 SVC 分类器，使其优于所有以前的方法。

# 编辑过的最近邻(ENN)

ENN 通过移除其类别与其最近邻类别不同的样本来对多数类别进行欠采样。如果重复这个步骤，那么我们驱动一个新的方法`RepeatedEditedNearestNeighbours`

另一种驱动方法:`AllKNN`与`RepeatedEditedNearestNeighbours`略有不同，它通过改变内部最近邻算法的 *k* 参数，在每次迭代时增加它。后两种算法需要更长的处理时间。

```
#ENN - EditedNearestNeighbours
for sampler in (EditedNearestNeighbours(),
         RepeatedEditedNearestNeighbours(),
        AllKNN(allow_minority=True)):
    enn_xtrain_tfidf, enn_train_y = sampler.fit_sample(xtrain_tfidf, train_y)
    accuracy = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),enn_xtrain_tfidf, enn_train_y, xvalid_tfidf)
    print ("LR {0}, WordLevel TFIDF: ".format(sampler), accuracy)
for sampler in (EditedNearestNeighbours(),
         RepeatedEditedNearestNeighbours(),
        AllKNN(allow_minority=True)):
    enn_xtrain_tfidf, enn_train_y = sampler.fit_sample(xtrain_tfidf, train_y)
    accuracy = train_model(svm.LinearSVC(),enn_xtrain_tfidf, enn_train_y, xvalid_tfidf)
    print ("SVM {0}, WordLevel TFIDF: ".format(sampler), accuracy)#LR EditedNearestNeighbours:  0.5480093676814988
#LR RepeatedEditedNearestNeighbours:  0.5727170236753101
#LR AllKNN: 0.5547785547785548
#SVM EditedNearestNeighbours:  0.7101303911735205
#SVM RepeatedEditedNearestNeighbours:  0.7046332046332046
#SVM AllKNN:  0.7131474103585658
```

K-最近邻 [KNN](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/#what-is-knn) 算法是一种健壮且通用的分类器，经常被用作更复杂分类器的基准，如人工神经网络(ANN)和支持向量机(SVM)。尽管简单，KNN 可以胜过更强大的分类器

AllKNN 应用 KNN 算法进行欠采样，使用 SVM 分类器时，其性能超过了所有重采样方法。KNN 具有内存开销，并且需要处理时间。

# 浓缩最近邻(CNN)

CNN 利用第一最近邻(1-NN)来迭代地决定样本是否应该保留在数据集中。与其他方法相比，通过 CNN 的欠采样可能较慢，因为它需要多次通过训练数据。CNN 被认为是噪声敏感的，并且保留有噪声的样本。另一种驱动方法`OneSidedSelection`也使用 1-NN，并使用上面讨论的`TomekLinks`来去除被认为有噪声的样本。同样由 CNN 驱动的`NeighbourhoodCleaningRule`使用`EditedNearestNeighbours`移除一些样本。此外，他们使用 3 个最近邻来移除不符合此规则的样本:

```
#CNN - CondensedNearestNeighbor
for sampler in (CondensedNearestNeighbour(random_state=0),
        OneSidedSelection(random_state=0),
        NeighbourhoodCleaningRule()):
    nm_xtrain_tfidf, nm_train_y = sampler.fit_sample(xtrain_tfidf, train_y)
    accuracysm = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),nm_xtrain_tfidf, nm_train_y, xvalid_tfidf)
    print ("LR {0}, WordLevel TFIDF: ".format(sampler), accuracysm)for sampler in (CondensedNearestNeighbour(random_state=0),
        OneSidedSelection(random_state=0),
        NeighbourhoodCleaningRule()):
    nm_xtrain_tfidf, nm_train_y = sampler.fit_sample(xtrain_tfidf, train_y)
    accuracysm = train_model(svm.LinearSVC(),nm_xtrain_tfidf, nm_train_y, xvalid_tfidf)
    print ("SVM {0}, WordLevel TFIDF: ".format(sampler), accuracysm)#LR CondensedNearestNeighbour:  0.4289156626506024
#LR OneSidedSelection:  0.5358851674641147
#LR NeighbourhoodCleaningRule:  0.5541327124563447#SVM CondensedNearestNeighbour:  0.48195030473511485
#SVM OneSidedSelection:  0.6998961578400832
#SVM NeighbourhoodCleaningRule:  0.7075376884422111
```

# 结合欠采样和过采样

过采样和欠采样还有其他方法，但如何将两者结合起来呢？Ajinkya More 在本文中声称，SMOTE 和 ENN 的结合为他的重新采样实验产生了最好的结果，让我们看看我们是否会达到同样的结果:

```
# Re-Sampling SMOTEENN
se = SMOTEENN(random_state=42)
se_xtrain_tfidf, se_train_y = se.fit_sample(xtrain_tfidf, train_y)
accuracy = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),se_xtrain_tfidf, se_train_y, xvalid_tfidf)
print ("LR SMOTEENN: ", accuracy)
accuracy = train_model(svm.LinearSVC(),se_xtrain_tfidf, se_train_y, xvalid_tfidf)
print ("SVC SMOTEENN: ", accuracy)#LR SMOTEENN:  0.39669421487603307
#SVC SMOTEENN:  0.47154471544715443
```

但是另一种方法 **SMOTE + Tomek 链接移除**在我们的实验中运行良好:

```
# Re-Sampling SMOTETomek
se = SMOTETomek(random_state=42)
se_xtrain_tfidf, se_train_y = se.fit_sample(xtrain_tfidf, train_y)accuracy = train_model(linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'),se_xtrain_tfidf, se_train_y, xvalid_tfidf)
print ("LR SMOTETomek: ", accuracy)
accuracy = train_model(svm.LinearSVC(),se_xtrain_tfidf, se_train_y, xvalid_tfidf)
print ("SVC SMOTETomek: ", accuracy)#LR SMOTETomek:  0.6756756756756755
#SVC SMOTETomek:  0.6876061120543293
```

# 结论

为不平衡语料库选择正确的重采样算法对于向分类器提供最佳训练数据集是重要的。在这个长示例中，我讨论了各种过采样、欠采样以及过采样和欠采样相结合的方法，并得出结论，最适合我们模型的方法是应用 AllKNN 欠采样算法。当在竞赛测试数据上测试代码时，f1 的分数是:0 . 46866 . 38686868661

我已经用[集合极度随机化的树](/@muabusalah/twitter-hate-speech-sentiment-analysis-6060b45b6d2c)做了同样的练习，这超过了这个方法。

完整的运行代码可以在 [github](https://github.com/mabusalah/Resampling) 上获得。