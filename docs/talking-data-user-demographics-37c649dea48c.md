# 对话数据用户人口统计

> 原文：<https://medium.com/analytics-vidhya/talking-data-user-demographics-37c649dea48c?source=collection_archive---------19----------------------->

## 田径比赛

![](img/a791b9f5363e8a8b683141327e205a4a.png)

ML/AI 在数字营销中的作用

**商业问题**

来源:[比赛链接](https://www.kaggle.com/c/talkingdata-mobile-user-demographics/overview/description)

当你走进街角咖啡馆的门时，没有什么比被你最喜欢的饮料问候更令人欣慰的了。虽然一个体贴的咖啡师知道你每周三早上 8:15 会喝一杯卡布奇诺，但在数字空间里，你喜欢的品牌要个性化你的体验要困难得多。

中国最大的第三方移动数据平台 Talking Data 明白，日常的选择和行为描绘了我们是谁，我们重视什么。目前，Talking Data 正在寻求利用中国每天活跃的 5 亿移动设备中超过 70%的行为数据，帮助其客户更好地了解他们的受众并与之互动。

在 kaggle 竞赛中，参赛者被要求建立一个模型，根据他们的应用程序使用情况、地理位置和移动设备属性来预测用户的人口统计特征。这样做将有助于全球数百万开发者和品牌广告商追求数据驱动的营销努力，这些努力与他们的用户相关并迎合他们的偏好。

**现实世界/业务目标和约束**

1.没有低延迟要求。

2.需要数据点属于每个类别的概率。

**现有解决问题的方法:**

。[https://github . com/gau tam-v-ml/talking data-Mobile-User-Demographics](https://github.com/Gautam-v-ml/TalkingData-Mobile-User-Demographics)

这里完成了 EDA 和数据清理。

使用逻辑回归等基本 ML 模型，他可以获得 2.38 的最高分

**对现有方法的改进:**

我跟踪了不同 DL 和 ML 型号的组装。

**现在我将开始我的方法:**

**数据概述**

数据可从 [**竞赛页面下载 _ 数据**](https://www.kaggle.com/c/talkingdata-mobile-user-demographics/data) 其中包括:

1. **gender_age_train.csv，gender_age_test.csv** —这些文件分别包含训练和测试模型的设备的详细信息。

2. **events.csv，app_events.csv** —当用户使用 TalkingData SDK 时，事件会连同其时间戳一起记录在该数据中。每个事件都有一个事件 id，位置(纬度/经度)，事件对应 app_events 中的一个应用列表。

3. **app_labels.csv** —包含用户使用的应用及其对应的 label_id。

4. **label_categories.csv** —由 app_labels.csv 中的应用标签 id 及其类别组成。例如，标签 ID: 4 属于游戏艺术风格类别。

5.**phone _ brand _ device _ model . CSV**—包含用户使用的设备的设备 id，以及相应的电话品牌和设备型号。

**绩效指标**

[**多类日志丢失**](https://www.kaggle.com/c/talkingdata-mobile-user-demographics/overview/evaluation) :每个设备都被标注了一个真实的类。对于每个设备，我们必须预测一组预测概率(每个类别一个)。公式是

其中，N 是测试集中设备的数量，M 是类别标签的数量，log 是自然对数，如果设备 I 属于类别 j，yij 是 1，否则是 0，pij 是观察值 ii 属于类别 j 的预测概率。

**探索性数据分析:**

导入所需的库

![](img/97b33d18d96fc1957bffb9475a658dc6.png)

开始对作为数据一部分的每个 csv 进行分析:

**性别 _ 年龄 _ 训练** **数据**

![](img/38694e08167c4010ab1b9f07d351926b.png)

因此，gender_age_train 包含 4 列:

**设备 id:** 注册了通话数据的用户设备 id

**性别:**用户的性别，用 M 或 F 分别表示男性或女性

**年龄:**用户的年龄

group: 这是我们问题的目标类，包含我们需要预测的类。第一个字母表示用户的性别，随后是用户所属的年龄组

请看下面的图表，它向我们展示了每组的数量

![](img/6183a7351eea39bd23bd8d7350005875.png)

从上面的图中我们可以推断出男性用户比女性用户多。这一点在饼状图中表现得更为明显，其中近 65%是男性，35%是女性。

![](img/660fa78e5e3fb131c1db8d213d613ebd.png)

**性别 _ 年龄 _ 测试数据**

这是我们需要预测组(目标变量)的测试数据

![](img/8d40455740f65398f928b5d10cb0771f.png)

所以测试数据只有一列 device_id。

![](img/3b93e2e097efed438610b49845d706cd.png)

**手机 _ 品牌 _ 设备 _ 型号数据**

![](img/7b76bb302f68b0a2b6ac034b6f2c4491.png)

我们这里有 device_id、phone_brand、device_model，并且注意到一些 phone_brand 和 device_model 值是中文的。

首先，我们将检查重复项。

![](img/6d904f7c3c6ebafe72f819f396477927.png)

所以有一些重复的，他们需要被删除。

**事件数据**

![](img/96afcd03ff17a5a3d537e93b30588a4e.png)

首先，我们将检查一个设备 id 是否有多个事件 id。

为此，我们来看看 id 为 **29182687948017175** 的设备的事件数据

事件[事件['设备 id']==29182687948017175]。头(10)

![](img/58ed92707368c2f72bd20d5091d513d4.png)

从上面我们可以说，设备 id 可以有多个事件。

我们有所有事件的时间戳，我们可以查看所有事件的总体开始和结束时间，以了解所有事件的记录时间。

![](img/61613912d4b2e41e59139018151404e8.png)

因此，我们可用的事件数据记录时间为 8 天，从 2016 年 4 月 30 日午夜到 2016 年 5 月 8 日凌晨 12 点。

现在最重要的是我们需要检查是否所有设备都有事件？

下面是讲述事件信息的图

![](img/639b20505b17643ef510e9605b22bb17.png)![](img/04ba0151facfbbcc5cd68d428f14a6fe.png)

因此，几乎 69%的数据没有事件。

现在我们看到数据是如何在全球传播的？

![](img/fb7035cae8bc6fbb599b0fb05485beab.png)

上表显示大多数事件发生在位于大西洋中部的(0，0)附近。可以有把握地假设，这些位置上的日志是用户不想共享他们的位置的产品，因此是无用的。其他坐标大部分位于中国，只有少数 pin 用户在世界其他地方。虽然有三分之一的数据没有信息，但我仍然认为使用位置信息是值得的，因为这两个数字显示了女性和男性之间的分布存在一定的差异

**app_labels 数据**

![](img/683b9dde91b2ca943f79f731ac21fef3.png)

现在我们会发现我们有多少独特的应用程序标签？

![](img/ce4ddcb25401b18e088c0476ce681182.png)

现在我们有 507 个唯一的应用程序标签，我们也可以有多个标签 id 与特定的应用程序 id 相关联。让我们考虑一个 id 为 **7324884708820027918** 的应用，它有多个标签。

应用标签[应用标签['应用标识']==7324884708820027918]

![](img/1b9855f4ea4c5d21e7c750b56b131149.png)

**app _ 事件数据**

![](img/487dff1c70b77c55613132bee77080ae.png)

我们将通过下面的代码了解独特的应用程序和事件的数量:

![](img/8d5dc40b162e27cf4fc6a9107b4d5e28.png)

is_active 列分析

![](img/e14dffcc098c0eb1eb6d75ca9e22fc98.png)![](img/e2e9088dea7fbe5fd468e5b238d7a209.png)

因此，当有活动正在进行时，大多数应用程序都是不活动的。几乎 61%的应用程序是不活跃的，39%的应用程序是活跃的。

**探索性数据分析结论**

1.只有 31%的训练和测试数据包含事件和应用相关功能
2。我们需要为没有事件的设备使用手机品牌和手机型号数据
3。对于包含事件信息的设备，我们可以使用事件相关功能以及电话品牌和型号功能

**数据准备**

以下是我在准备模型中使用的数据时遵循的步骤:

1.因为我们的数据中有两种类型的设备，一种有事件详细信息，另一种没有任何事件详细信息。我分离了设备，并为有事件的设备和无事件的设备创建了数据。

2.对于没有事件数据的设备，我只使用手机品牌和设备型号作为特征。

3.对于具有事件数据的设备，我使用了手机品牌、设备型号以及事件数据功能，如中纬度、中经度、事件发生的时间、事件发生的星期几、应用程序事件数据中的 is_active 功能、设备中使用/安装的所有应用程序列表、按设备 id 分组的所有应用程序标签列表。

要查看准备数据的更多细节和代码，请查看我的库:[我的 Github 库](https://github.com/ytataji/TalkingDataUserDemographics)

现在是时候使用不同的模型来获得更好的预测了！

**以下是实现以下目标的不同方法:**

1.由于我们数据中的所有设备都有手机品牌和设备型号的详细信息，所以我使用这两个特征来训练一个逻辑回归和两个不同的神经网络模型。我使用这些模型来预测测试数据中设备的类别概率，这些测试数据不包含我们在前面的步骤中分离的事件数据。

2.对于包含事件细节的设备，我使用我们为这些设备提取的事件相关特征，并且仅使用包含事件细节的设备来训练两个不同的神经网络模型。然后，我使用这些模型来预测包含事件数据的测试数据中设备的分类概率。

3.最后，我将有事件的设备和无事件的设备的测试数据预测连接起来，并创建了整个测试数据预测文件。

**使用两个不同数据集的特征工程**

因为我们有两个独立的数据，一个是有事件的设备，另一个是没有事件的设备，所以我为这些数据分别准备了特性。

1.**无事件设备:**创建手机品牌、设备型号特征的一次性编码。

2.**带事件的设备:**创建手机品牌、设备型号、属于设备的应用程序、属于设备的应用程序标签、TFIDF 小时编码功能、星期几、应用程序激活、标准化纬度和经度功能的一键编码。我们称之为**事件特征矩阵。**

**建模**

正如我的方法中提到的，在这一节中，我将带您浏览我的解决方案中使用的模型。这包括两个部分，即，在没有事件细节的设备数据上建模和在包含事件细节的设备数据上建模。让我们先从没有事件细节的设备建模开始。

**无事件详情的设备**

在开始建模之前，我将为我的模型创建训练、验证和测试数据。

![](img/9eff871a350485b838b1feebf7782519.png)![](img/51bd44bf2b3335e69e0c5b2e2b8f6802.png)

**X_train_one_hot** 是所有设备(有无事件)的列车数据的手机品牌、设备型号特征的一键编码。**X _ test _ no _ events _ one _ hot**是列车数据的手机品牌、设备型号特征的一键编码，仅用于不包含事件详情的设备。原因是，正如我在方法中提到的，手机品牌、设备型号功能适用于所有设备。因此，我针对训练数据中所有可用设备的这些特征来训练模型，并使用模型来预测测试数据中没有事件细节的设备。

**逻辑回归**

逻辑回归模型在 train_1 上训练，cv_1 是验证数据，test_1 是我们预测分类概率的测试数据。

我已经做了超参数调整，我得到了最好的 C 为 0.1(最低的日志损失)

所以我们使用最佳 C 值为 0.1

![](img/f1ee474d800c8eb5de248b5b8e8c0b8c.png)

仅使用电话品牌和设备型号 one-hot 编码作为特征，逻辑回归模型具有 2.38 的 CV 对数损失。

不，我们去找神经网络

**神经网络 1**

从[竞赛讨论页](https://www.kaggle.com/c/talkingdata-mobile-user-demographics/discussion/23424)引用的神经网络架构

![](img/b861d51534abfa1e3c2894e6e287acd5.png)

**input_shape** 这里是 X_train_one_hot 中的特征数。

![](img/968c01832fa32447ae35d23b52034894.png)

我在 X_train_one_hot 数据上使用不同的随机分割 train，CV 训练神经网络 1 模型 5 次，如下所示

**model_list_1** 包含使用不同随机种子在不同版本的 X_train_one_hot 数据分割上训练的 5 个模型。我使用了 model_list_1 中的每个模型，并对测试数据(test_1)进行了概率预测，然后取所有预测的平均值。

下面是以上 5 个模型的张量板标量

![](img/1723223b13a0da29afbf63dc911e9671.png)

**神经网络 2**

参考自[竞赛讨论页面](https://www.kaggle.com/c/talkingdata-mobile-user-demographics/discussion/23424)的神经网络架构

![](img/f132e798a47f9c7830667d2d01d8fb02.png)

这里， **input_dim** 是 X_train_one_hot 中的特征数。

![](img/ef49499f8ffec2ca9f9d9ab053dcc643.png)

我在 train_1 上对神经网络 2 进行了一次 30 个时期的训练，并使用该模型对测试数据(test_1)进行预测。

请参考[我的 Github 库](https://github.com/ytataji/TalkingDataUserDemographics)获取张量板标量

**带有事件详情的设备**

让我们为我的模型创建训练、验证和测试数据，就像我们以前为没有事件数据的设备所做的那样。

![](img/822dd4e361cda2d7ca736d5e6e54a98b.png)

这里 X_train_events_one_hot_1 是我们在特征准备步骤中创建的列车数据的**事件特征矩阵**。X_test_events_one_hot_1 是一个类似于测试数据的**事件特征矩阵**。

**神经网络 3**

参考自[竞赛讨论页面](https://www.kaggle.com/c/talkingdata-mobile-user-demographics/discussion/23424)的神经网络架构

输入图层中的差值在此处增加了价值，因为它为预测提供了可变性，因为当我们在输入图层中使用差值时，在每次运行期间，只有一组随机要素被用作模型的输入。

![](img/10d108aaa95d55e4480c8d98ebb838d2.png)

这里， **input_dim** 是 X_train_events_one_hot_1 中的特征数。

![](img/e3c515a820bc9412ca6b8af05215d60b.png)

我训练神经网络 3 模型 20 次。

然后，我使用 model_list_2 中的 20 个模型中的每一个对测试数据(test_2)进行预测，并取预测概率的平均值。

请参考[我的 Github 库](https://github.com/ytataji/TalkingDataUserDemographics)获取张量板标量

**神经网络 4**

该神经网络是神经网络 3 的变体，但是具有 2 个密集层和不同数量的隐藏单元。

![](img/2a26acca6bd99c6c03c69c7559cf5a78.png)

这里， **input_dim** 是 X_train_events_one_hot_1 中的特征数。

![](img/d9e02bd66432ac62b393a66a91e96717.png)

类似于我训练神经网络 3 的方法，我训练神经网络 4 模型 20 次。

然后，我使用 model_list_3 中的 20 个模型中的每一个对测试数据(test_2)进行预测，并取预测概率的平均值。

请参考[我的 Github 库](https://github.com/ytataji/TalkingDataUserDemographics)获取张量板标量

**模特合奏**

1.**无事件数据的设备:**

![](img/8165867b068933ec1016acbe6542b7dc.png)

**2。具有事件数据的设备:**

![](img/82173a3aaf699a61eab94b7b4bbf4b75.png)

最终预测:

![](img/3270401937554b46fd50101341aa4d2a.png)![](img/a537a2fe2d0fae3633feaccef448dbdc.png)

最后，我将有事件和无事件设备的这些测试数据预测连接起来，创建了包含 112071 行的整个测试数据预测文件，其中每行包含设备属于 12 个类别中每一个类别的预测概率。

![](img/5869d8cd0be3f7de8dc255692c301119.png)

**结果**

1.**无事件数据:**对所有设备的手机品牌、设备型号的 One_Hot 编码进行模型训练。

2.**事件数据:**仅针对包含事件细节的设备，在事件特征矩阵上训练模型。

3.模型中的平均值表示模型有多次运行，并且预测是平均的。

![](img/fbef89feb371727c1adc5af2326f8469.png)

提交分数的串联测试预测:

![](img/08a2b80a6b1b0c4ac61b27291863e35c.png)

**进一步改进**

1.我们可以在不同的模型组合上使用不同的权重来改善对数损失

2.我们可以在不同的 ML 模型、随机森林或任何模型上使用集成

我试图用最简单的方式解释，希望你能理解我的博客。如果您有任何疑问，请随时通过[我的 LinkedIn 个人资料](http://www.linkedin.com/in/TatajiYerubandhi)联系我

**感谢您的阅读！！**

**参考文献**

1 . https:[/](https://www.kaggle.com/c/talkingdata-mobile-user-demographics/)/www . ka ggle . com/c/talking data-mobile-user-demographics/

2.[https://www . ka ggle . com/c/talking data-mobile-user-demographics/discussion/23424](https://www.kaggle.com/c/talkingdata-mobile-user-demographics/discussion/23424)

3.[https://machine learning mastery . com/model-averaging-ensemble-for-deep-learning-neural-networks/](https://machinelearningmastery.com/model-averaging-ensemble-for-deep-learning-neural-networks/)

[4 . https://www . applied ai course . com/lecture/11/applied-machine-learning-online-course/3081/what-are-ensembles/4/module-4-machine-learning-ii-supervised-learning-models](https://www.appliedaicourse.com/lecture/11/applied-machine-learning-online-course/3081/what-are-ensembles/4/module-4-machine-learning-ii-supervised-learning-models)