# 微软研究院 EconML 的因果推理世界

> 原文：<https://medium.com/analytics-vidhya/a-world-of-causal-inference-with-econml-by-microsoft-research-7dd43e97ce09?source=collection_archive---------2----------------------->

# 介绍

这是令人惊讶的头条新闻之一，即研究*获得了 2019 年 Abhijit Banerjee、Esther Duflo 和 Michael Kremer 的诺贝尔经济学奖。据说，他们已经引入了新的方法和标准，以获得可靠的方法来从发展经济学的微观数据中确定因果关系，即使在微观数据中观察到的关系有时会受到混淆效应的影响。[他们的研究中加入了 RCT(随机对照试验)](https://en.wikipedia.org/wiki/Randomized_controlled_trial)技术来确定因果关系，这是过去在科学和医学领域诞生的一种众所周知的方法，近年来在 EBPM(循证决策)领域频繁出现。[【1】](https://www.nobelprize.org/prizes/economic-sciences/2019/press-release/)*

*![](img/9f183acbe7e25a8ff6503477771f26b2.png)*

*图片来自 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=3501528) 的 [Ahmed Gad](https://pixabay.com/users/ahmedgad-9403351/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=3501528)*

*[](https://www.nobelprize.org/prizes/economic-sciences/2019/press-release/) [## 2019 年纪念阿尔弗雷德·诺贝尔的瑞典中央银行经济学奖

### 2019 年 10 月 14 日瑞典皇家科学院决定授予瑞典央行经济学奖

www.nobelprize.org](https://www.nobelprize.org/prizes/economic-sciences/2019/press-release/) 

> 获奖者的研究发现——以及跟随他们脚步的研究人员的研究发现——极大地提高了我们在实践中战胜贫困的能力。作为其中一项研究的直接结果，500 多万印度儿童从有效的学校补习项目中受益。另一个例子是许多国家对预防性保健的大量补贴。* 

*在这篇文章中，我将在统计和机器学习中的因果推理的上下文中覆盖一个稍微广泛的领域，另外介绍一个由 Mirosoft Research 开发的 Python EconML 包，它包含一些用于计量经济学的机器学习技术，可以将特定领域中复杂的因果推理问题自动化。这是一个概述性的调查，目的是让自己对因果推理的最新研究进行研究，并探索一个案例使用 EconML 软件包进行实际因果研究的可能性。*

**免责声明:本文包含第三方网站或其他内容的链接，仅供参考(“第三方网站”)。第三方网站不受我及其附属机构的控制，我不对任何第三方网站的内容负责，包括但不限于第三方网站中包含的任何链接，或第三方网站的任何更改或更新。我向您提供这些链接只是为了方便，包含任何链接并不意味着我对该网站或其运营商的任何关联的认可、批准或推荐。**

# *RCT(随机对照试验)*

*让我们从 RCT(随机对照试验)开始吧，但我不会在这里就因果推理而言过多地讨论 RCT 本身的细节。因为从历史的起点到现在，这条路太长了。从 1747 年[詹姆斯·林德](https://en.wikipedia.org/wiki/James_Lind)的第一份报告确定了[坏血病](https://en.wikipedia.org/wiki/Scurvy)的治疗方法，到后来在农业领域，由于[杰吉·内曼](https://en.wikipedia.org/wiki/Jerzy_Neyman)和[罗纳德·a·费希尔](https://en.wikipedia.org/wiki/Ronald_A._Fisher)，有很多故事可供你查阅这项技术是如何被发现的，以及这项实验是如何被完善的。费希尔的实验研究和他的著作特别在公众中普及和熟悉了随机实验。[【2】](https://en.wikipedia.org/wiki/Randomized_controlled_trial)*

*[](https://en.wikipedia.org/wiki/Randomized_controlled_trial) [## 随机对照试验

### 随机对照试验(或随机对照试验；RCT)是一种科学(通常是医学)实验…

en.wikipedia.org](https://en.wikipedia.org/wiki/Randomized_controlled_trial) 

让我们先用 RCT 的基本知识来训练自己。RCT 是如何工作的在名字中解释得很清楚。与**随机**、**对照**和**试验**相关。我们拿一个人群来做某个假设的实验。我们可以将它分成多个组，假设我们有两个组，标记为*和 ***【对照】*** ，从群体中随机选取。前一组 ***【治疗】*** 进行干预评估，而后一组 ***【对照】*** 进行替代性治疗(多数为安慰剂治疗)，或根本不进行干预。RCT 是这样的。在排除选择偏差和分配偏差的随机**组中，对**控制的**组的实验干预可以显示**试验**的效果。可以通过与对照组比较来评估治疗效果。这是《RCT 医学案例》的一篇好文章，附有浅显易懂的解释。[【3】](/@berationable/what-is-evidence-part-1-randomised-controlled-trials-dd17328f0c32)***

*[](/@berationable/what-is-evidence-part-1-randomised-controlled-trials-dd17328f0c32) [## 什么是证据？第 1 部分:随机对照试验

### 随机对照试验是测试药物、饮食等的重要工具。以下是你如何使用它作为…

medium.com](/@berationable/what-is-evidence-part-1-randomised-controlled-trials-dd17328f0c32)* 

*在 RCT，随机化对于避免混杂效应至关重要。当通过避免选择偏倚和分配偏倚以获得满意的前提条件来实现随机化分组的公平性时，RCT 的治疗效果可以最大化。一旦试验开始，采用*“盲法试验”*来避免参与者、实验者和评估者对试验结果产生偏见是很重要的。RCT 得了五分，这是马里兰科学方法量表上最可靠的方法。它评估一项评估是否提供了关于政策影响的令人信服的证据。RCT 可以说是一种可信的统计因果推断方法，因为它是为了临床试验而发明的，通过减少系统评价中的虚假因果关系和偏见来进行决策。[【4】](https://whatworksgrowth.org/public/files/Scoring-Guide.pdf)*

*RCT 的缺点经常被提及，如昂贵的成本和时间、伦理问题、利益冲突等。首先，组与组之间可能会有一些差异，因为有时无法实现公平的随机化，而且不能强迫治疗组的人接受治疗，所以他们可以在表面上假装接受治疗，这将极大地改变实验结果。它也不关心分配的组中的子组之间以及个体之间的不同治疗效果(可以考虑个体的一些属性)。因此，接下来我将在介绍 EconML 库之前，介绍一些其他的因果推理方法，以解决 RCT 的缺点或那些可以绕过问题的方法，以及如何为您的目的选择一种技术。*

# *因果推理*

*进行 RCT 实验需要昂贵的成本和时间，那么这个实验可以产生牺牲的实验数据。你可能已经注意到，理想的 RCT 方法需要实验数据，而一些深奥的方法需要观察数据([鲁宾因果模型](https://en.wikipedia.org/wiki/Rubin_causal_model)是假设潜在结果的观察研究的统计方法之一)。由于某些原因，有时无法获得合法的实验数据，当不可能进行随机对照实验时，估计治疗效果可能是可行的。因为？由于实验数据的噪声性质，例如数据中的高方差，或者当治疗组中的一些人没有接受治疗时，或者当组的真正随机化根本不可能时。此外，我们可能希望了解指定组或个人的子组之间的不同有效性，以便制定更个性化的策略或服务。有多种因果推断方法，不仅是统计 RCT 方法，还有其他方法来解决 RCT 的缺点，或解决实验数据和观察数据中的额外复杂性，从而使我们更深入地了解治疗效果。*

*优步工程发表了一篇非常有用的文章，关于如何为实验数据和观察数据案例选择因果推断方法。如果你通读这篇文章，你会对优步工程中已经采用的各种方法有一个大概的了解，以及这些技术如何改进了对在[的](https://eng.uber.com/analyzing-experiment-outcomes/)实验准实验和观察数据的分析。因果推理正在成为统计方法和机器学习技术与各种方法的交叉，我们可以在流程图中找到这些方法，以掌握这里的概述。据说该流程图并不详尽，但看起来很容易理解实验数据的假设情况。*

*[](https://eng.uber.com/causal-inference-at-uber/) [## 使用因果推理改善优步用户体验

### 本文是我们致力于强调因果推断方法及其行业的系列文章的第二篇…

eng.uber.com](https://eng.uber.com/causal-inference-at-uber/)* 

# *因果推理方法*

*![](img/33ff3461dd4e2d4f52db4cb092ac8c7e.png)*

*[*图一。因果推断方法适用于非常具体的实验数据。*](https://eng.uber.com/causal-inference-at-uber/)*

*看左上方的方框。如前所述，RCT 的第一个重大缺陷似乎是要求用适当的方法进行足够公平的随机化。在 RCT，人口被随机分组是意料之中的事。由于实验数据的噪声性质，在指定的组之间可能总是存在预先存在的差异。有一些方法来处理预先存在的问题，如通过减少数据中的方差来调整预先存在的差异，或将实验视为观察性研究，以了解这种情况下的因果关系。我介绍一下倾向评分匹配，供大家参考，还有其他三个的链接。*

*让我们考虑一下治疗组和对照组的样本数量的差异。图片我们从总共 1000 人(一个人群)中选出 100 人为治疗组，900 人为对照组。**治疗组中的这 100 人可能存在选择偏倚，且治疗效果将与偏倚的协变量的影响混淆。**倾向评分匹配可以减少这种因混杂因素导致的偏差，混杂因素可能在通过简单比较接受治疗和未接受治疗的单位之间的结果而获得的治疗效果估计中发现。*

*但是怎么做呢？倾向评分基本上是治疗组成员相对于对照组成员的预测概率(通过逻辑回归)——基于观察到的预测因素。换句话说，这是每个参与者相对于混杂因素在治疗组中的概率。现在，我们可以使用倾向分数进行匹配，即根据倾向分数从比较单元中选择治疗单元中的一个参与者与一个或多个非参与者。一对一分配的卡尺匹配是通用的(在处理单元的倾向分数的特定宽度内的比较单元得到匹配)。*

*   *[CUPED(使用预实验数据的受控实验)](https://booking.ai/how-booking-com-increases-the-power-of-online-experiments-with-cuped-995d186fff1d)*
*   *[差异中的差异](https://en.wikipedia.org/wiki/Difference_in_differences)*
*   *[倾向评分匹配](https://en.wikipedia.org/wiki/Propensity_score_matching)*
*   *[IPTW(治疗加权的逆概率)](https://en.wikipedia.org/wiki/Inverse_probability_weighting)*

*如果在指定的组中没有预先存在的差异，我们将需要考虑治疗效果是否可信，因为治疗组中的一些人可能有意识或无意识地没有接受治疗。例如，**一些被分配到治疗组的人接受了药物治疗，但由于某些原因，他们实际上没有服用该药物。他们不是实际接受治疗的，而是被算作治疗组的成员。所以估计的效果会被稀释，因为治疗组的一些人实际上没有接受治疗。情况就是这样。这里我们想知道的是，当且仅当接受治疗的人被分配到治疗组时，他们的治疗效果。这些人在这项研究中被称为**编者**([约书亚·d·安格里斯特、圭多·w·因本斯和唐纳德·b·鲁宾 1996](https://www.jstor.org/stable/2291629?seq=1) )。另一方面，**非顺从者**由四个小组中的其他三个(从不接受者、总是接受者、违抗者)组成，如果我们可以假设排他性限制的话。***

*   *编辑者——通过分配给处理方法来诱导接受处理方法*
*   *从不接受者——不改变分配的状态，回避者*
*   *永远接受治疗——不管任务是什么，自愿接受治疗*
*   *defier——做与他们的任务相反的事情，避免对控制组进行任务分配处理或被诱导进行任务分配处理*

*[CACE(编译者平均因果效应)](https://en.wikipedia.org/wiki/Local_average_treatment_effect)或[晚期(局部平均治疗效应)](https://en.wikipedia.org/wiki/Local_average_treatment_effect)是编译者的 [ATE(平均治疗效应)](https://en.wikipedia.org/wiki/Average_treatment_effect)当且仅当他们被分配到治疗组时，编译者是被诱导采取治疗的人群的子网。在不完全依从的情况下，我们不可能直接确定 [ATE(平均治疗效果)](https://en.wikipedia.org/wiki/Average_treatment_effect)。相反，在 RCM 中，将 CACE/LATE 作为一个观察变量进行估计变得更加可行([鲁宾因果模型](https://en.wikipedia.org/wiki/Rubin_causal_model))。它可以通过估计的 ITT(意图治疗)效应与估计的从犯比例的比率来估计，或者通过[工具变量](https://en.wikipedia.org/wiki/Instrumental_variables_estimation)估计器来估计。[【6】](https://en.wikipedia.org/wiki/Local_average_treatment_effect)*

*   *[CACE(编者平均因果效应)](https://en.wikipedia.org/wiki/Local_average_treatment_effect)*

 *[## 局部平均治疗效果

### 局部平均治疗效应(晚期)，也被称为编者平均因果效应(CACE)，是第一…

en.wikipedia.org](https://en.wikipedia.org/wiki/Local_average_treatment_effect)* 

*在左起第三个方框中，如果分配的组或个人中的子组之间的有效性不同，或者可能在治疗组和对照组的子组中都存在一些片段，则是这种情况。例如，在 RCT，对于那些治疗组和对照组，没有考虑诸如性别、年龄或某些部分的属性的概念。因为随机化被期望通过最大化统计效果、最小化选择偏差和与协变量的影响混淆来将群体分配到亚组中。例如，假设我们想知道公司给客户的电子邮件的处理效果，以避免取消服务(如降低电话合同的流失率)。了解什么样的人(属性)可能会降低他们流失的可能性，这一点很重要。我们在机器学习模型中看到的似乎不是预测性分析，而是更规范的分析，即治疗可能如何改变特定小组或个人的结果。异质性治疗效果是指干预对具有特定特征集的样本的感兴趣结果的影响，通常涉及 [CATE(条件平均治疗效果)](https://egap.org/methods-guides/10-things-heterogeneous-treatment-effects)计算。 [CATE(条件平均治疗效果)](https://egap.org/methods-guides/10-things-heterogeneous-treatment-effects)是特定于受试者亚组的平均治疗效果，其中亚组由一些属性(例如，女性个体中的 ATE)或实验发生的环境的属性(例如，多位点现场实验中特定位点的个体中的 ATE)来定义。[本文还介绍了隆起建模](https://www.predictiveanalyticsworld.com/machinelearningtimes/uplift-modeling-making-predictive-models-actionable/8578/)和[分位数回归](https://eng.uber.com/analyzing-experiment-outcomes/)来估计非均质处理效果。*

*   *[HTE(异质治疗效果)](https://egap.org/methods-guides/10-things-heterogeneous-treatment-effects)*
*   *[隆起建模](https://www.predictiveanalyticsworld.com/machinelearningtimes/uplift-modeling-making-predictive-models-actionable/8578/)*
*   *[分位数回归](https://eng.uber.com/analyzing-experiment-outcomes/)*

*正如我们所看到的，有从实验到观察研究的各种方法来了解或估计特定情况下的因果关系和因果效应。预先从数据中准确预见你想要调查的问题和治疗效果是至关重要的。异质治疗效果可用于不预测结果，但可用于估计治疗可能如何改变特定亚组的结果，该亚组具有一组特殊的特征和属性，在分析中计算 [CATE(条件平均治疗效果)](https://egap.org/methods-guides/10-things-heterogeneous-treatment-effects)。[另一方面，隆起建模](https://www.predictiveanalyticsworld.com/machinelearningtimes/uplift-modeling-making-predictive-models-actionable/8578/)需要 A/B 实验测试数据来训练模型，然后使用该数据根据结果确定不同区段的最佳处理。*

# *经济学导论*

*ALICE(代表因果关系和经济学的自动学习和智能)是由微软研究院领导的一个项目，也是 EconML 软件包的实施者，该软件包是从观察数据中估计[CATE](https://egap.org/methods-guides/10-things-heterogeneous-treatment-effects)的一个有用的便捷工具。[【7】](https://www.microsoft.com/en-us/research/project/alice/)*

*[](https://www.microsoft.com/en-us/research/project/alice/) [## 爱丽丝-微软研究院

### 爱丽丝自动学习和智能因果关系和经济学爱丽丝是一个项目，直接人工…

www.microsoft.com](https://www.microsoft.com/en-us/research/project/alice/) 

> EconML 是一个 Python 包，用于通过机器学习从观察数据中估计异质治疗效果。

有多种实现估计器的方法，这些估计器分为两个主要类别，一种以各种方式利用机器学习技术来估计异质治疗效果，例如在双机器学习、双鲁棒学习和正交随机森林(基于森林的估计器)中，一种使用元算法，该元算法包括分别处理控制组和治疗组的基本学习器(随机森林、线性回归等)和元学习器中的元级，该元级可以被视为基本学习器的函数。

安装 EconML 很简单，只需运行 pip 命令，如下所示。有一个容器映像，其中包含基于 Anaconda3 的 econml 包，或者包含 github 存储库中的笔记本的 Dockerfile。您可以通过 pip 安装软件包，也可以将代码克隆到本地并构建它以供测试。

```
# install econml package
$ pip install econml# use docker image
$ git clone [git@github.com](mailto:git@github.com):yuyasugano/econml-test.git
$ docker build -t econml .
$ docker run -p 3000:8888 -v ${PWD}/notebooks:/opt/notebooks econml
```

![](img/d429c6d090d24efbf99c4984a22b4726.png)*

*容器中的`conda list`显示了截至编写时的库版本如下。econml 0.7.0 就是在这个容器中构建的。*

```
***econml                    0.7.0**                    pypi_0    pypi
numpy                     1.16.0                   pypi_0    pypi
scikit-learn              0.21.2           py37hd81dba3_0
scikit-image              0.15.0           py37he6710b0_0
pandas                    0.24.2           py37he6710b0_0
h5py                      2.9.0            py37h7918eee_0
tensorboard               2.1.1                    pypi_0    pypi
tensorflow                2.1.0                    pypi_0    pypi
tensorflow-estimator      2.1.0                    pypi_0    pypi*
```

*如果您想为所需的库使用这些固定版本，请不要自己创建映像，而是从 DockerHub 中提取构建的映像。[【9】](https://hub.docker.com/r/suganoyuya/econml)*

 *[## 码头枢纽

### 编辑描述

hub.docker.com](https://hub.docker.com/r/suganoyuya/econml)* 

*如果您使用`-v ${PWD}/notebooks:/opt/notebooks`进行了批量安装，那么您已经有了示例笔记本。CustomerScenarios 目录下有两个信息丰富的案例研究，一个是估计随多个客户特征变化的**异质价格敏感度**，以了解哪类用户对媒体公司案例的折扣反应最强烈，一个是通过处理旅游公司案例特征的一些缺点，从直接 A/B 测试中理解**异质处理效果**。*

*![](img/7644d979a237d906d5b80502558d2223.png)*

*客户场景*

*如果你迷失在方法选择的迷宫中，给定的流程图有助于确定库中的哪个类将满足用户指南页中的要求。[【10】](https://econml.azurewebsites.net/spec/flowchart.html)*

*![](img/8dc78b8e08802bef4cf9da3e0ab74305.png)*

*[库流程图](https://econml.azurewebsites.net/spec/flowchart.html)*

*正如我们所看到的，这是统计方法和机器学习技术在各个领域和行业的交叉，以帮助当今的政策和业务决策。EconML 是一个丰富而有用的工具集，可以根据特定亚组或具有特定属性或特征的人的观察数据来估计 CATE(异质性治疗效果),这很好。然而，从 RCT 等经典方法到相对较新的 EconML 库，使用这些方法需要统计和机器学习方面的广泛专业知识以及更深入的知识，以便我们理解在正确的地方正确使用。在这里完全涵盖因果推论是不现实的。因此，它可能是肤浅的，但我希望这篇小文章为你打开一扇通往因果推理世界的大门。*

# *参考*

*   *[1] [新闻稿:2019 年经济学奖](https://www.nobelprize.org/prizes/economic-sciences/2019/press-release/)*
*   *[2] [维基百科—随机对照试验](https://en.wikipedia.org/wiki/Randomized_controlled_trial)*
*   *【3】[什么是证据？随机对照试验](/@berationable/what-is-evidence-part-1-randomised-controlled-trials-dd17328f0c32)*
*   *[4] [使用马里兰科学方法量表的评分方法指南](https://whatworksgrowth.org/public/files/Scoring-Guide.pdf)*
*   *[5] [利用因果推理改善优步用户体验](https://eng.uber.com/causal-inference-at-uber/)*
*   *[6] [百科—当地平均治疗效果](https://en.wikipedia.org/wiki/Local_average_treatment_effect)*
*   *[7] [爱丽丝(因果关系和经济学的自动学习和智能)——微软研究院](https://www.microsoft.com/en-us/research/project/alice/)*
*   *[8] [微软/EconML](https://github.com/Microsoft/EconML)*
*   *[9] [菅野鸭/econml](https://hub.docker.com/r/suganoyuya/econml)*
*   *[10] [EconML 用户指南—库流程图](https://econml.azurewebsites.net/spec/flowchart.html)*
*   *[11] [微软研究院 EconML 的因果推理世界](https://techflare.blog/a-world-of-causal-inference-with-econml-by-microsoft-research/)**