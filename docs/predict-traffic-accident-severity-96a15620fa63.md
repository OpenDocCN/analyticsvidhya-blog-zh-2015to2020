# 预测交通事故严重程度

> 原文：<https://medium.com/analytics-vidhya/predict-traffic-accident-severity-96a15620fa63?source=collection_archive---------20----------------------->

分类模型

![](img/8240a2320ff6209eb71d541876e4218f.png)

预测交通事故严重程度

道路安全应该是政府、地方当局和私营公司投资于有助于减少事故和提高整体驾驶安全的技术的优先考虑。

这里我们将分析历史碰撞数据并准备分类模型来预测未来事件。

# **数据:**

从[https://S3 . us . cloud-object-storage . appdomain . cloud/cf-courses-Data/cognitive class/DP 0701 en/version-2/Data-collisions . CSV](https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv)下载数据集

数据集形状: **194673 个样本，38 个特征**

![](img/ccbc3fc8628813cca964718c8b6e7ccf.png)

功能列表

这里 **SEVERITYCODE** 是目标变量。

**1:财产损失，**

**2:受伤**

# **数据清理和数据分析:**

下图显示了目标变量中的类别分布不平衡。

![](img/3812f9f246629f2dc4e7c11f083811cd.png)

我们需要减少大多数班级的人数以使分布均衡。

![](img/09ce2674d510bc7c2288142eec74eea6.png)

执行向下采样后目标变量中的分布。现在是平衡数据集。

![](img/fe4742267ae04ba8c447e91e0ccb75c8.png)

识别并删除缺失值超过 30%的列。并移除缺失值少于 3%的样本。因为如果我们使用任意方法填充缺失的数据，生成坏数据的可能性会更高。

![](img/d3009df7d3052eebcce668a3f3e42637.png)![](img/be130699407060d50937df5fffd07094.png)

缺失值> 30%和缺失值< 3%

Check category distribution of other features with respect to target variable along with data distribution in continuous features and visualize results. It will help to find duplicate and ambiguous categories within data set.

It will also help to check patterns in data before performing down sampling and after performing down sampling. If data distribution gets changed after applying any up sampling or down sampling technique, that means either we may have generated bad data or have lost some important information from original data set. Here pattern remains intact after applying down sampling technique that means we have not lost any important information.

![](img/35a447e6dc940471675fc4bf78bd2adc.png)![](img/aa29b40665d6c86406ca46a665330a47.png)

**关于目标变量的数据分布:下采样前**

![](img/6ab276ab0bcc57a0effc00d30e40429b.png)![](img/e48cdad21ffc2db9345e2e6b22a86e81.png)

**目标变量的数据分布:降采样后**

**观察:**

上图显示，即使在下采样后，数据模式仍保持不变。这意味着我们没有丢失任何重要数据。

具有模糊类别的功能“UNDERINFL”。我们应该把 Y 和 1 合并，把 N 和 0 合并。

具有模糊类别的特征天气。我们应该把雨夹雪/冰雹/冻雨和下雪混在一起，其他的和未知的，有严重侧风的风沙/尘土混在一起。

具有不明确类别的特征道路状况。我们应该把其他的和未知的融合在一起。

具有不明确类别的特征光导管。我们应该把其他未知的、黑暗的——路灯关掉——没有路灯。

它还表明，最大的碰撞发生在晴朗的天气干燥的道路上和日光下。

检查异常值并删除。

![](img/5d89e9fdb63399caab3fba2d9075a002.png)

然后将分类变量转换为数值，因为机器学习模型不支持字符串或文本数据。

![](img/a6b5001d60fc30bf8b713cfcb234022e.png)

丢弃重复样本并标准化输入数据。

![](img/7724b4277f01bae7e4cbcca13689f127.png)

# **建模:**

![](img/05bd9077c8199e91da8d0287b0f7d62d.png)![](img/623fe6aed46bc0073351383d24d3e238.png)![](img/8e6eb809ca6d07d572f970212cc6eaeb.png)

> 在这里，您也可以使用 scikit-learn 的 **GridSearchCV** 方法尝试多个超参数，这将产生给定模型的最佳超参数。稍后，您可以使用这些参数来训练模型。

使用**准确度、F1-得分、Jaccrd 相似性得分、精确度、召回率进行评估。**

![](img/a2cb8d83b21a05a9a58e70a1c6e4e583.png)

# **结果:**

![](img/b612aa7cd85d9a6c666311dc75f61de4.png)

使用不同算法和评估指标的预测结果

![](img/6ee9672d81996642bd506b7b3cf66037.png)

# **结论:**

SVM 在其他车型中表现最佳，同时培训时间也比其他车型高 100 倍。

如果我们追求精确，那么决策树表现良好，如果我们追求召回，那么 KNN 表现良好。

这里“精确”指的是预测的碰撞伤害百分比是真实的伤害。取而代之的是，90%的碰撞确实涉及到了被正确预测的伤害.对于这个特定的问题，召回率比精确度更重要，因为高召回率将有利于所有需要的资源都准备好来预测碰撞是否涉及伤害。