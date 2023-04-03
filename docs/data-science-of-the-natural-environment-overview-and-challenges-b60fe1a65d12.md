# 自然环境的数据科学:概述与挑战。

> 原文：<https://medium.com/analytics-vidhya/data-science-of-the-natural-environment-overview-and-challenges-b60fe1a65d12?source=collection_archive---------5----------------------->

本文旨在

*   给你一个新兴的环境数据科学领域的概念
*   让您深入了解环境科学的数据、建模和复杂性带来的一些技术挑战。

![](img/e224be79cec6512dacfd1b059ad11402.png)

图片来源:马塞尔·杜维·德克尔

首先，简单介绍一下数据科学和环境科学。然后，我们将开始探索科学背后的数据挑战、环境建模及其背后的复杂性。

> 海洋不会拒绝河流，同样的道理也适用于我们科学数据的透明度和公开性。系统科学探索复杂的动态系统是如何由多个相互关联的维度组成的。生物学家用还原主义者的方法研究系统已经几十年了。如果我们不整合一个用于数据和模型互操作性的互联和集成的系统，我们怎么能希望理解一个这样的系统呢？

D 尽管数据科学在现代商业中具有重要意义，但它尚未在环境科学中得到广泛应用。尽管“大数据”很大，而且环境科学越来越由数据驱动(Hey 等人，2009 年)，但这是怎么回事呢？

首先，我还不习惯“大数据”这个术语…

![](img/c91e81a3514679e0b086c9279f40538b.png)

归功于 SiliconANGLE

我认为我们不应该把注意力放在数据的大小上，因为还有很多其他的事情需要我们去关注；据说真正的挑战在于这种大数据来源的复杂性和异构性。

> -这不全是关于大小，事实上，在环境科学领域更重要的是数据的多样性和准确性(精确或准确程度)。
> 
> 你听说过大数据的四个 V 吗？

> 多样性、准确性、速度和容量。

让我们看看自然环境。数据科学和环境科学都是交叉学科领域。看看下面的雨林。这是一个使用许多相互关联的变量从多个层面得出结论的领域。现在，环境科学领域越来越多地由数据驱动，并且正在朝着开源数据的方向发展。

![](img/94cd0caf989bec5435547b8a01ccea27.png)

Markus Mauthe /绿色和平组织

这是一件好事，因为更高的透明度将创造更多的机会和联系。也许是因为环境科学有压力去影响政策和形成科学支持的生态系统管理战略。

正如 [Blair G. S .等人在 2019 年](https://www.frontiersin.org/articles/10.3389/fenvs.2019.00121/full)指出的那样:( 1)高度复杂数据集的整合，(2)将这些数据转化为新的知识，例如围绕生态系统服务的知识，(3)为政策提供信息，例如面对气候变化的适当缓解和适应战略。

![](img/7b772f5aec630370c84cb3326c621c09.png)

我认为数据科学将是弥合上述差距的一个好工具。让我们仔细看看为什么做得这么少…

虽然不断有大量新数据进入，但有时数据可能很难处理。Blair G. S .等人的上述论文详细介绍了处理环境数据的挑战:[自然环境的数据科学:研究路线图](https://www.frontiersin.org/articles/10.3389/fenvs.2019.00121/full)。下面我将总结三个主要挑战领域:数据、建模和复杂性挑战:

![](img/e4eab3f7d9fd8c7a4b5c91d4ae27a40e.png)

来源:【Flydata.com】大数据集成的 6 大挑战

## 数据挑战概述:

*   处理数据源固有的异构性，并实现整个数据集的互操作性。
*   为什么不建立步入式数据中心？请把数据递给我！
*   将数据从封闭的科学家牢笼中解放出来！使用链接数据模式(一组发布和创建语义网的最佳实践),语义网是一个数据网，您可以轻松地与系统进行交互，以获取这些数据的含义。
*   建立系统和机制来验证数据及其测量过程的准确性和精确性。
*   事情可能更像万维网——整合世界各地的数据集来帮助科学。

> I 互操作性——计算机系统或软件交换和利用信息的能力

![](img/f22acaa4e1ab4636e7f5654671bac6a9.png)

[图片来源:bejo / Shutterstock](https://www.shutterstock.com/g/bejo)

## 建模挑战总结:

理解和预测环境变化的主要工具。模型**大致分为流程模型或数据驱动模型。以下摘自布莱尔. G. S. et。艾尔。, 2019.**

*   通过云上传支持增加模型的可共享性和开放访问；
*   更强的互操作性，甚至在流程驱动和数据驱动模型之间；
*   支持构建一系列可能的集合模型；
*   支持集成建模，包括潜在的高度复杂和多方面的自然资本评估模型；
*   推理和**管理模型运行中的不确定性**,包括整体和跨集成建模框架。

> **自然资本**可以定义为**自然**资产的世界存量，包括地质、土壤、空气、水和所有生物。

> 好了，抓紧你们的帽子！我们将要深入混沌理论、自我实现的回音室和复杂性科学中的复杂性！准备好了吗？

![](img/b0c8abd57e5babc8681c7d097b8df66b.png)

## 复杂系统:

直到一周前，我才知道复杂性科学是一个东西。复杂性科学是对复杂系统的研究。Kastens 等人(2009 年)将这些系统明确定义为具有以下特征的系统:

*   反馈回路，其中变量的变化导致该变化的放大(正反馈)或衰减(负反馈)；
*   许多密切相关的变量，多个输入对观察到的输出有贡献；
*   混沌行为，即对初始条件、分形几何和自组织临界性的极端敏感性；
*   多重(亚)稳定状态，其中条件的微小变化可能导致系统的重大变化；
*   输出的非高斯分布，通常情况下，远离平均值的结果比您想象的更有可能出现。

这种复杂性存在于整个自然界的生物群落和生态系统中，可以肯定地说，数据科学家有他们的工作要做。下面是一些真实世界的例子，从数据科学的角度说明了如何处理复杂系统的这些属性:

反馈回路

来自 stack exchange 的这篇[帖子询问反馈循环是否会导致机器学习算法变得不那么精确。答案是肯定的。是的，他们有。理论是给定模型的预测变得更加偏向特定类型的数据。对该数据给予更多关注，并且当模型被重新训练时，预测被进一步驱动到该数据方面。](https://datascience.stackexchange.com/questions/47666/can-feedback-loops-occur-in-machine-learning-that-cause-the-model-to-become-less/47668)

一个很好的例子就是新闻回音室。机器学习算法看到你喜欢与某个观点相关的新闻/视频，你看更多这样的视频，模型就变得更加确信你的选择。因此，它建议更多的内容与相似的观点。正如[卫报所说的](https://www.theguardian.com/science/blog/2017/dec/04/echo-chambers-are-dangerous-we-must-try-to-break-free-of-our-online-bubbles)我们必须努力打破网络泡沫。

![](img/371fbf994fd0e01d89febe3b1462afdc.png)

图片来源:克里斯托弗·沃莱特

如果你想了解更多关于回音室的内容，你可以点击[这里](https://en.wikipedia.org/wiki/Echo_chamber_(media))和[这里](http://theconversation.com/explainer-how-facebook-has-become-the-worlds-largest-echo-chamber-91024)。

多重共线性

> OLS 回归中多重共线性的影响是众所周知的:很可能出现高标准误差、过度敏感或无意义的回归系数以及低 t 统计量。这些效应使得对所研究的系数的解释几乎不可能。——小卡尔·帕特里克·克拉克，2013 年。

![](img/67f5ec46de47198b0cc3d387bbd04e8d.png)

我有许多相互关联的变量，我的模型散发着浓郁的桃花心木的味道

正如这篇 [Minitab 博客](https://blog.minitab.com/blog/understanding-statistics/handling-multicollinearity-in-regression-analysis)中所建议的，我们可以做许多事情来帮助应对多重共线性的挑战:

1.  我们可以(I)从模型中移除高度相关的预测因子。如果您有两个或更多具有高方差通胀因子(VIF)的因子，请从模型中移除其中一个。此外,( ii)考虑逐步回归或子集回归,( iii)使用我们对数据集的专业知识来挑选要包含在我们的模型中的变量。
2.  使用偏最小二乘回归(PLS)或主成分分析，这种回归方法将预测因子的数量减少到一个较小的不相关成分集。

> 在统计中，**方差膨胀因子** ( **VIF** )是一个多项模型的方差与一个单项模型的方差的商。——统计学学习导论第 8 版。

对于多层模型，Carl Patrick Clark Jr 博士在他的论文中提供了一种计算多重共线性的方法，他创建了一种称为多层方差膨胀因子(MVIF)的方法。

haos 理论与人工神经网络模拟水-大气过程

> 混沌理论——对混沌的研究——动力系统的状态(其中一个函数代表几何空间中一个点的时间相关性的系统),其看似随机的外观是无序和不规则的状态，由对初始条件高度敏感的确定性法则控制— [高级数学术语的权威词汇表](https://mathvault.ca/math-glossary/)

![](img/1b1e90073983d23b5a63ea18f6a068d5.png)

混沌理论蝴蝶:图片来源:Asteriaa

![](img/4edf1cb925e489913efe52c87a9330fc.png)

双杆摆视觉

Soojun Kim 等人。艾尔。，旨在评估水文气象过程中的混沌行为。混沌理论以前曾被统计学家提出来研究复杂系统中的这类非线性现象。使用人工神经网络来分析他们评估的关联维度，他们发现蓄水量(目标特征)的混沌特征很可能是水库系统混沌行为的特征，而不是输入变量(气温、流量、降雨量)本身。

![](img/56b223aa1e541f76ee15e6e02a8ae5d3.png)

(美国鱼类和野生动物管理局)

还有由极端值和异常值引起的非正态分布的问题。

## 总结一下:

我们已经看到，使用来自环境科学的大数据会面临诸多挑战，重要的不仅仅是数据的大小。在这个领域中，我们最关心的是底层数据源的准确性和多样性。此外，需要解决使用和共享数据以及创建和共享易于相互集成的模型的挑战。最后，复杂系统中的复杂性是非常复杂的，我们正在谈论动态系统的混沌、多个亚稳态，它们通常有反馈回路！哇哦。如果你热衷于这方面的研究，Blair G. S .等人的[路线图](https://www.frontiersin.org/articles/10.3389/fenvs.2019.00121/full#B41)提供了更多的内容，他们呼吁社区共同努力，为这一领域做出贡献。

![](img/e64d7cf4dcd876fd84888ae79d599f65.png)

在附近。

感谢阅读和免责声明，所有这些东西对我来说都是新的，我在学习中不断进步。我在参考书目的底部为你添加了一个有趣的碳足迹计算器:)

# 参考资料和活动:

1.  嘿，t，坦斯利，s，和巨魔，k。).(2009).*第四范式:数据密集型科学发现*。微软研究院。
2.  Clark P. C .，(2013) [多级模型中多重共线性的影响](https://corescholar.libraries.wright.edu/cgi/viewcontent.cgi?article=1879&context=etd_all)
3.  卡斯滕斯、曼杜萨、塞瓦托、弗罗德曼、古德温、李奔等人(2009 年)。地球科学家如何思考和学习？ *Eos Trans* 。90, 265–266.doi: 10.1029/2009EO310001
4.  金素君，金永洙，李宗洙，金洪洙。, (2015).水文气象过程中混沌行为的识别和评估。Hindawi 出版公司《气象学进展》2015 卷，文章编号 195940，12 页[http://dx.doi.org/10.1155/2015/195940](http://dx.doi.org/10.1155/2015/195940)
5.  [https://www . the guardian . com/science/blog/2017/dec/04/echo-chambers-is-dangerous-we-must-try-break-free-of-our-online-bubbles](https://www.theguardian.com/science/blog/2017/dec/04/echo-chambers-are-dangerous-we-must-try-to-break-free-of-our-online-bubbles)
6.  [https://en . Wikipedia . org/wiki/Echo _ chamber _(media)](https://en.wikipedia.org/wiki/Echo_chamber_(media))
7.  [http://the conversation . com/explainer-how-Facebook-已经成为世界上最大的回音室-91024](http://theconversation.com/explainer-how-facebook-has-become-the-worlds-largest-echo-chamber-91024)
8.  统计学习导论第 8 版。[http://faculty.marshall.usc.edu/gareth-james/ISL/](http://faculty.marshall.usc.edu/gareth-james/ISL/)
9.  詹姆斯，加雷斯；丹妮拉·威滕；哈斯蒂，特雷弗；罗伯特·蒂布拉尼(2017)。*统计学习导论*(第 8 版。).斯普林格科学+商业媒体纽约。[ISBN](https://en.wikipedia.org/wiki/International_Standard_Book_Number)[978–1–4614–7138–7](https://en.wikipedia.org/wiki/Special:BookSources/978-1-4614-7138-7)。
10.  高等数学术语的权威词汇表[https://mathvault.ca/math-glossary/](https://mathvault.ca/math-glossary/)
11.  艾伦·图灵研究所关于数据科学气候与环境的活动[https://www . Turing . AC . uk/events/Data-Sciences-climate-and-environment](https://www.turing.ac.uk/events/data-sciences-climate-and-environment)
12.  强烈推荐！一个快速直观的碳足迹计算器[！：](https://footprint.wwf.org.uk/?pc=ATC001002&gclsrc=aw.ds&ds_rl=1263542&ds_rl=1263542&gclid=CjwKCAiAob3vBRAUEiwAIbs5TstzQ0ou7A4AiOgUhOQ6ecBbKiNjXB3TTlu0O9awIwa8TYVTX6kazBoCexEQAvD_BwE&gclsrc=aw.ds#/)

如果你想更多地了解我，请到我的环境博客上来。