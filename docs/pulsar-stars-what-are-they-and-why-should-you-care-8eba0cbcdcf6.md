# 使用机器学习来预测我们银河系中的垂死恒星…甚至更远！

> 原文：<https://medium.com/analytics-vidhya/pulsar-stars-what-are-they-and-why-should-you-care-8eba0cbcdcf6?source=collection_archive---------17----------------------->

## 在这篇文章中，我展示了机器学习模型如何用于正确分类垂死的恒星，或者更具体地说，脉冲星，因为它们似乎对空间探索最有潜在用途。

这将是一次预测地球上的高倍望远镜和未来潜在的深空探测器观测的结果是否是脉冲星的旅程。在我们进入我开发的帮助识别脉冲星的机器学习模型之前，我们先来谈谈脉冲星或“脉冲星星”实际上是什么，因为它们没有脉动，实际上也不是严格意义上的恒星(不再是了)。

![](img/e8bacaddbd1c6b1effb355018c4fae60.png)

图片来源:PITRIS/GETTY IMAGES

# 什么是脉冲星？

为了便于解释，假设恒星有生命。(无意冒犯地球上的任何恒星)恒星最终会在成为超新星时死亡，如果它们质量极大，会坍缩成黑洞。如果它们的质量较小，在 7 到 25 个太阳质量之间(太阳质量的 7-25 倍)，或者如果它们特别富含金属，可能会稍大一些，它们就会成为中子星，这是一种半径只有大约 10 公里的超高密度质量，但密度如此之大，以至于如果放在地球上，一茶匙的质量相当于珠穆朗玛峰的重量。数百万年来，中子星不断从其坍塌的核心发出辐射，直到它们最终完全冷却，成为它们曾经辉煌的天体的冰冷残余。

这些被称为中子的恒星残余物的一个较小子集被称为“脉冲星”或“脉冲星星”。虽然术语“脉冲星”是由“脉冲”和“恒星”两个词组合而成，但脉冲星没有脉动，也不再是活着的恒星。如上图所示，它们的电磁辐射是连续的，但却是从磁极发出的。磁轴通常与旋转轴不同，所以它们被称为脉冲星，因为从任何单一的角度来看，每次光束扫过观察者的视角时，都会观察到辐射脉冲。这些扫描的间隔非常有规律，因为它们随着脉冲星的每一次旋转而发生。应该注意的是，脉冲星永远不会“死亡”,因为它们实际上已经是死亡的恒星，但最终，在大约 100 多万年的时间里，它们的自转会变慢，它们的发射也会停止。在这一点上，只有通过它们对其他天体施加的引力或者通过检测一种叫做[黑体辐射](https://en.wikipedia.org/wiki/Black-body_radiation)的东西，它们才能被探测到。

# 地球人关心脉冲星什么？

## 地球人以我们有限的五官无法观测到大部分脉冲星天体，甚至无法观测到它们的光束。它们对我们没有影响，所以你可能会想:“这与我或整个人类有什么关系？”

发现脉冲星候选体，然后正确识别这些潜在的脉冲星是否是真正的脉冲星，需要大量的时间、金钱和努力。这已经持续了半个多世纪，无疑涉及到许多科学家和大量资金来完成迄今为止对 2000 多颗已知脉冲星的识别。

我们已经有了一个利用地球上的无线电天线网络导航太阳系的系统，称为[深空网络](https://en.wikipedia.org/wiki/NASA_Deep_Space_Network)。这很好地确定了飞行器已经飞行了多远，以及要到达目的地还需要多远。DSN 的唯一问题是，航天器离地球越远，DSN 就越不精确，它只能确定航天器离地球的距离，而不是其在空间的横向位置。利用脉冲星作为一组灯塔信标，飞行器可以精确确定自己在三维空间中的位置。这将变得更加重要，因为更多的飞船被发射出去探索我们太阳系以外的地方。

![](img/de29707666f583e70c2df6b5d009c5d1.png)

图片来源:多元宇宙中心

# 现在是数据科学！

## 那些在这里寻找诙谐戏谑和图片的人可以随意浏览这一切…这将变得非常技术性。

如果你留下来，你将会看到所谓的“监督机器学习”。对于数据集中的每个观测，我都有所有经过验证的标签，(脉冲星或非脉冲星)。在将数据分成训练/验证/测试集之后，我将取出验证和测试集的答案，并使用大部分数据(训练数据)的答案来训练我的模型，然后使用该模型来预测其余数据的答案。

*本项目使用的数据集可以在 Kaggle* [*这里*](https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star) *和 UCI* [*这里*](https://archive.ics.uci.edu/ml/datasets/HTRU2) *找到。*

由于这些数据是天文学家精心收集和记录的，非常干净和完整，我的第一个任务是将数据分成一组，用于训练我开发的每个模型，一组用于验证每个模型的性能并检查调整我的超参数的指标，最后一组用于测试模型从未见过的模型。

我选择将 10%的数据分离到测试集中，剩下的 90%中，20%用于我的验证集。这意味着该模型根据 72%的可用数据进行训练，并根据剩余的 28%进行验证和测试。

```
# Setting aside a sample of the data for testing.
# this portion will not be “known” during the model adjustment process.
 train, test = train_test_split(ps,test_size=0.10, 
                                stratify=ps[‘target_class’],
                                random_state= 17)# Separating the remaining data into training and validation sets.
#this allows me to tweak my model to perform at it's best before testing 
#it on unknown data. 
train, val = train_test_split(train,test_size=0.20, 
                              stratify=train['target_class'],
                               random_state= 17)
train.shape,val.shape,test.shapeOut[7]:((12886, 9), (3222, 9), (1790, 9))
```

接下来，我快速检查了目标类在我的分割中的分布。这种均匀分布是使用“目标类”功能上的“分层”参数实现的。(在这一栏中，脉冲星标记为 1，非脉冲星标记为 0)

```
train['target_class'].value_counts(normalize=True),
val['target_class'].value_counts(normalize=True),
test['target_class'].value_counts(normalize=True)Out[8]:(0    0.908428
 1    0.091572
 Name: target_class, dtype: float64,
 0    0.908442
 1    0.091558
 Name: target_class, dtype: float64,
 0    0.90838
 1    0.09162
 Name: target_class, dtype: float64)
```

## 固体

如果我只是建立一个总是报告‘0’(非脉冲星)的模型，它在大约 90.8%的时间里是正确的(我们的基线精度)。这可能很难用机器学习来改进，但我喜欢挑战！

![](img/4be3b72a122e7a7eedd462d686f1c9ab.png)

图片 via: [NRAO 脉动](https://public.nrao.edu/gallery/parts-of-a-pulsar/) r

**现在把我们的数据分成 X 个特征矩阵和 y 个目标向量。**

```
target = 'target_class'
features = ps.columns.drop(target)X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]
y_test = test[target]
```

## 转向模型生产…

我不会张贴其余的代码，因为它可能会有点冗长，但如果代码示例是你的难题，请随意访问我的 GitHub 页面，在那里有[项目笔记本](https://github.com/DoctorDroid/Build2/blob/master/Bulid%202%20%20Pulsar%20Stars.ipynb)。

# 第一个模型:随机森林回归量

```
Validation Accuracy for Random Forest Regrssor : 0.8039317754042418
```

该模型没有显示出基线精度的提高。它只在 80.4%的时候预测到正确的类别。虽然这是我所期望的，并且对许多任务来说似乎仍然很好，但简单地预测没有观测到脉冲星会给我 90.8%的准确性，所以对我来说，这个模型对这个用例没有用。

# 模型 2:逻辑回归

```
Validation Accuracy for Logistic Regression 0.978895096213532
```

在第一次尝试中，没有对 SciKitLearn 的 linear_model 模块中的股票逻辑回归函数进行调整，该模型在给定数据的情况下预测脉冲星方面做得很好。这对我来说很棒，因为我可以使用线性模型，通过几行简单的代码来确定数据集中每个特征的系数。

![](img/f1d30133c7d77f22c51a64aea1f9fe2b.png)

线性回归模型的系数图

# 模型 V3.0:随机森林分类器

```
Validation Accuracy for Random Forest Classifier 0.9798261949099938
```

经过多次迭代，调整超参数越来越低，我发现了这个模型的性能能力的甜蜜点。

现在我们说话了！RFC 模型能够在验证集上 98%的时间内预测正确的分类。没有太多的改进可以做，但我仍然会尝试更好的准确性分数…为科学！

## 虽然我很想看到《迷失太空》的另一季，但我不想负责真人秀版本的发布。

所以为了我自己的安心或者为了证明，在决定这是否是一个有用的模型之前，让我们看一下另一个度量。

# 其他指标

**我的随机森林分类器模型对验证集做出的预测的混淆矩阵:**

![](img/9ddf63df7d84882a5db26469256ea7e0.png)

作者创作的情节图像

*此混淆矩阵显示了我的模型预测正确的频率，以及它何时被特征“混淆”并错误地标记了观察结果。*

如果我在撞击或空间异常导致的不可预见的路线改变后，在太空中漫无目的地漂浮，我会希望我的分析更倾向于谨慎，(将“可能”归类为“不是脉冲星”)，而不是相反。在这个假设中，我的团队没有使用机器标记为非脉冲星的实际脉冲星，只是根据它作为脉冲星报告的观测结果来重新定向和修正航向。

# ROC/AUC 曲线

![](img/d15fe2c2a853c92789089bfe5f52c0b5.png)

作者创建的绘图图像

```
ROC/AUC score =  0.9754350205277573
F1 score =  0.8969258589511754
Recall score =  0.8406779661016949
```

我的 RFC 模型更倾向于假阴性，而不是在分类错误时的假阳性。

因此，我认为这个模型是有用的。

## 尽管如此，对“更好”的追求仍在继续…

# 决赛:XGBoost 分类器:

```
Validation Accuracy using XGBoost 0.9823091247672253This is a  0.0024829298572315306  improvement on the RFC model.
```

我的 RFC 模型无法实现的另一项预测准确性改进。这同样需要对超参数进行微调才能实现。

**我的 XGBoost 分类器模型在验证集上做出的预测的混淆矩阵:**

![](img/7d3e5739a7b163a348eb50980a7e6a8c.png)

作者创作的情节图像

# 尤里卡。…

# 没那么快。我需要检查其他指标。

![](img/f21bca9d29ed4e5099e9cbcb12d41aad.png)

作者创建的绘图图像

```
ROC/AUC score =  0.9828035878698036
F1 score =  0.8969258589511754
Recall score =  0.8406779661016949
```

XGBoost 的准确性和 ROC/AUC 分数有所提高，但召回率或 F1 分数没有下降。

*警告:下图中的物体可能看起来比实际更近。*

![](img/6887f202d62028a9eb07bce00934c648.png)

图片 via:【PHYS.org 

# 测试精度:

没错，我终于得到了我的模型从来不知道的测试数据，直到在编码、验证、调整、验证、调整、验证中获得了所有的乐趣……好了，你明白了。

```
Testing accuracy with Random Forrest Regressor =  0.7627017333383153
Testing accuracy with Linear Regression =  0.976536312849162
Testing accuracy with Random Forrest Classifier =  0.976536312849162
Testing accuracy with XGBoost =  0.9793296089385475
```

# 我们有一个赢家:XGBoost！

机器学习算法中发生的许多事情对用户来说是隐藏的。多亏了伦德伯格和李，我们数据科学家现在有了一个叫做 SHAP 的人工智能图书馆来帮助解释这种数字魔法是如何发生的。有趣的事实:Shapley values 最初是为博弈论开发的，并以提出它的劳埃德·沙普利的名字命名。

![](img/62f118b1ea07356ac3e1a92516ef6bfc.png)

图片来源:SHAP Github。(致谢中的链接)

# 模型解释

![](img/664db3a98d84ca8df8298f0df243461a.png)

模型的特点是撞击分为 1 级脉冲星和 0 级非脉冲星

![](img/36b7d755b655a812b57e72e207f8aadd.png)

来自 RFC 模型的非脉冲星观测的 Shapley 力图

![](img/2e33616c00d899d0a49c5650afe0b2e8.png)

来自 RFC 模型的脉冲星观测的 Shapley 力图

为那些不满足的人准备了更多的情节。同样，这些图和其他图的所有代码都可以在[这里](https://github.com/DoctorDroid/Build2/blob/master/Bulid%202%20%20Pulsar%20Stars.ipynb)找到。

![](img/3d6aa453898ef2949a70dd9d87e013a0.png)

使用 [eli5](https://eli5.readthedocs.io/en/latest/) 排列所有特征的重要性列表

![](img/c63f87934a70a5d56626ee4dddf22506.png)

最具影响力特征的部分相关性图

![](img/403f005a1d9300b98ac1fac30b72814b.png)

由作者使用 [pdpbox](https://github.com/SauceCat/PDPbox) 创建

# Outro:

我可以想象，在我们以比光速慢得多的速度在宇宙中航行后的某个时候，来自另一个时间、地点或维度的一些其他生物可能会给我们一块发光的石头，它可以在眨眼之间带着我们去任何地方，只需一个想法……但在那一天到来之前，我们有机器学习。

感谢你的阅读！所有的掌声都将献给这个让我着迷的宇宙。干杯！

引用:

R.J. Lyon，B. W. Stappers，S. Cooper，J. M. Brooke，J. D. Knowles,《脉冲星候选选择的五十年:从简单的滤波器到一种新的原则性实时分类方法》,皇家天文学会月报 459 (1)，1104–1123，DOI: 10.1093/mnras/stw656

数据集的 DOI:

R.j .里昂，HTRU2，DOI:10.6084/M9 . fig share . 3080389 . v1。

感谢

该数据是在英国工程和物理科学研究委员会(EPSRC)对曼彻斯特大学计算机科学博士培训中心的授权 EP/I028099/1 的支持下获得的。原始观测数据由高时间分辨率宇宙合作组织利用巴夏礼天文台收集，该天文台由澳大利亚联邦资助，由澳大利亚联邦科学与工业研究组织管理。

其他来源:

[【SPINN:脉冲星候选选择问题的直接机器学习解决方案】](https://academic.oup.com/mnras/article/443/2/1651/1058949)

[【利用人工神经网络选择射电脉冲星候选】](http://old.inspirehep.net/record/856422/plots?ln=en)

[【NRAO 公众形象许可条款】](https://creativecommons.org/licenses/by/3.0/)

[【脉冲星介绍、脉冲星计时和脉冲到达时间测量】](http://ipta.phys.wvu.edu/files/student-week-2017/IPTA2017_KuoLiu_pulsartiming.pdf)

[【s . Lundberg，S. Lee《解释模型预测的统一方法》，NIPS 2017】](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)

[【SHAP Github】](https://github.com/slundberg/shap)