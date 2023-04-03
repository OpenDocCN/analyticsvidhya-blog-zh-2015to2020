# 超参数调谐-简要理论和你在手册中找不到的东西

> 原文：<https://medium.com/analytics-vidhya/hyperparameter-tuning-8ca311b16057?source=collection_archive---------14----------------------->

# 理论，实用知识，实例！

最初发布于 Portal do Thiaguera:

[](https://www.thiagocarmonunes.com.br/) [## 蒂亚格拉门户

### Thiagueraémeu portfolio de projetos 门户。

www.thiagocarmonunes.com.br](https://www.thiagocarmonunes.com.br/) 

# 参数和超参数:

# 参数:

概率模型是由称为参数的未知量来估计的。使用优化技术对这些进行调整，以便在训练样本中能够以最佳方式找到模式。简单地说，参数是由算法估计的，用户对它们几乎没有控制。

> 在简单的线性回归中，模型参数是β。

![](img/bdefa87b3ad67bc5be2627238b162b01.png)

font:[https://pt . slide share . net/vitor _ vasconcelos/regresso-linear-mltipla](https://pt.slideshare.net/vitor_vasconcelos/regresso-linear-mltipla)

**嘭！！！**:在统计学术语中，参数定义为总体特征。所以严格地说，当我们想讨论样本特征时，我们使用估计量这个术语。在这种情况下，这没什么区别，但值得注意的是。

# 超参数:

超参数是用于控制算法学习方式的信息集。他们的定义影响模型的参数，被视为一种学习的方式，从新的超参数改变。这组值影响模型的性能、稳定性和解释。

每个算法都需要一个特定的超参数网格，可以根据业务问题进行调整。

超参数改变模型学习的方式，在参数之后触发该训练算法以生成输出。

> 在决策树中，一个主要的超参数是树的深度和每个叶子中样本的数量。

![](img/a7d7145ba2004fef5fe9d21ac0854f8c.png)

font:[https://towards data science . com/scikit-learn-decision-trees-explained-803 f 3812290d](https://towardsdatascience.com/scikit-learn-decision-trees-explained-803f3812290d)

**警报+嘭！！！**:仅训练+测试数据交叉验证策略不应选择超参数。对此我们必须非常小心。

# 数据集策略:

当目标是调整和测试超参数配置时，数据排列必须设计成:

1.  训练:用超参数网格训练算法的数据集；
2.  验证:将应用性能度量来验证超参数的数据集；
3.  测试:有效验证模型性能的最后一组。

![](img/770df7bcbe03e8b70d975616a87bfafd.png)

font:[https://Harvard-iacs . github . io/2017-cs109 a/labs/lab 4/notebook/](https://harvard-iacs.github.io/2017-CS109A/labs/lab4/notebook/)

# 超参数选择:

为了便于选择，我们来谈谈 4 种技巧:

1.  商业知识；
2.  网格搜索；
3.  随机搜索；
4.  贝叶斯优化。

# 商业知识:

具有商业知识的超参数选择是最复杂的。它要求你能够抽象出你的模型，并拟合从以前的研究或商业知识中得到的假设和行为，以做出更合适的选择。在这个阶段，依靠支持选择超参数集的科学文章和出版物来更成功地解决问题也是很常见的。

**Obs！！！**:在我看来，这是启动优化算法的最佳方式。然而，它更复杂，需要数据科学家的大量经验和阅读。

# 网格搜索:

网格搜索结合了科学家建立的一系列超参数，并贯穿所有超参数来评估模型的性能。它的优点是，这是一个简单的技术，将通过所有的编程组合。最大的缺点是，它遍历参数空间的特定区域，无法理解空间的哪个运动或哪个区域对优化模型很重要。

![](img/33d457c68f57b361988d9317b94d4e62.png)

font:[https://www . data camp . com/community/tutorials/parameter-optimization-machine-learning-models](https://www.datacamp.com/community/tutorials/parameter-optimization-machine-learning-models)

# 随机搜索:

在随机搜索中，超参数是在它可以假定的值的范围内随机选择的。这种方法的优点是有更大的机会找到具有更合适的超参数的成本最小化空间的区域，因为每次迭代的选择是随机的。这种方法的缺点是超参数的组合超出了科学家的控制。

![](img/c1c0821f28a1c286d8d22307ff3ef9d4.png)

font:[https://www . data camp . com/community/tutorials/parameter-optimization-machine-learning-models](https://www.datacamp.com/community/tutorials/parameter-optimization-machine-learning-models)

# 网格与随机搜索:

![](img/6dddf04a7a8be1cebdc920952ff11bc3.png)

font:[https://community . alter yx . com/t5/Data-Science/Hyperparameter-Tuning-Black-Magic/ba-p/449289](https://community.alteryx.com/t5/Data-Science/Hyperparameter-Tuning-Black-Magic/ba-p/449289)

# 贝叶斯优化:

贝叶斯超参数优化是一种非常有效和有趣的方法来找到好的超参数。在这种方法中，朴素的解释方式是使用支持模型来寻找最佳超参数。

基于概率模型(通常为高斯过程)的超参数优化过程将用于从给定模型或测试超参数集的性能的后期分布中观察到的数据中寻找数据。

![](img/aba07727c9145bafb89ba6652b9e6fba.png)

font:[https://medium . com/analytics-vid hya/hyperparameter-search-Bayesian-optimization-14 be 6 fbb 0 e 09](/analytics-vidhya/hyperparameter-search-bayesian-optimization-14be6fbb0e09)

由于它是每次迭代的贝叶斯过程，所以与所使用的超参数相关的模型性能的分布被评估，并且生成新的概率分布。有了这种分布，就有可能对我们将使用的一组值做出更合适的选择，从而使我们的算法以最佳方式学习。

**总结**！！！:这是一个优化学习参数的模型。对于这个主题的更技术性的解释，我推荐下面的文章:

[贝叶斯优化技术](/analytics-vidhya/hyperparameter-search-bayesian-optimization-14be6fbb0e09)

![](img/2eb9b05329af7caae71ded00e9434130.png)

字体:【https://en.wikipedia.org/wiki/Bayesian_optimization 

# 实用知识。你在手册里找不到的

# 处理/培训时间:

训练一个模型，即使只有一次，也是一项非常昂贵的任务。如果我们仍然需要改变这个算法的超参数，我们必须找到对基础设施和业务领域的交付/可用性的需求可行的策略。小心！！！科学家在笔记本上训练的模型和植入的模型之间有巨大的差距。所以要考虑到这一点！

# -开发期间:

面对大量数据时，一个适当的策略是使用训练库中的样本来搜索最佳超参数。注意需要分层并理解每个特征的类别之间以及模型因变量之间所需的最小代表性。

**警惕！！！**:不要在训练集和测试集中没有代表所有变量类别的情况下进行采样。

# 向前看部署/实施

科学家与业余爱好者的不同之处在于他解释模型的能力，以及他对如何在可用的基础设施中实现所发现的模式的看法。因此，为了让我们更专业，总是需要思考一个模型将如何实现，从它的估计结构到所需的计算复杂度/容量。

合适的模型应该具有良好的性能和稳定性，并且易于部署/提供。

![](img/6c8d74db99197c9ada30308f10df3e62.png)

font:[https://www . researchgate . net/figure/marrieve-do-not-equal-foresight-Knowledge-of-outcome-bias-our-judgment-about-the _ fig 2 _ 245102691](https://www.researchgate.net/figure/Hindsight-does-not-equal-foresight-Knowledge-of-outcome-biases-our-judgment-about-the_fig2_245102691)

# 学习崩溃

是的，我的朋友们，即使是算法也深受其害！！!'*

优化技术无法收敛到最佳点的情况数不胜数，在小范围内变化，优化可能不会“优化”(不收敛)。基于梯度下降的技术，例如 BFGS，可能存在局部热点的主要问题。这种情况的替代方案是改变优化算法或使用降维技术来简化问题。

![](img/27f142a50446a849be178ee53d772c16.png)

font:[https://psicospecialidades . com . br % 2f processmento-auditivo-central-x-dificuldade-de-aprendizagem](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.psicoespecialidades.com.br%2Fprocessamento-auditivo-central-x-dificuldade-de-aprendizagem%2F&psig=AOvVaw2Atw-9TzkQ62wyQvxwqCXL&ust=1596143801976000&source=images&cd=vfe&ved=0CAMQjB1qFwoTCJDdyuWx8-oCFQAAAAAdAAAAABAD)

# 缺乏评估模式

在给定一组可用信息的情况下，不可能对数据做出精确的假设。当科学家开始意识到这一事实时，第一个策略是使用更激进的参数来交付 MVP(“最小可行产品”)。当使用非常激进的超参数时，除了可解释性的显著损失之外，还可能有模型残差的虚假减少。因为这种误差的减少可能不能有效地反映正确的数据模式。一个实际的例子是多项式回归。这种技术能够完美地拟合任何线性相关的数据。这种结构复杂性的增加，尽管减少了误差，但并不反映问题的线性性质。所以要注意这一点。

**提示！！！**:永远不要用一个复杂的等效性能模型来代替一个简单且可解释的结构，因为你对屏幕背后发生的事情知之甚少。**节俭(‘少即是多’)**

# 关键超参数:

绝大多数技术都有一小组超参数，负责大部分模型调整。例如，在决策树中，每个节点/叶中个体的深度和比例是关键，因此，这些超参数的正确选择可能已经足以进行良好的调整。

![](img/edd83a49bdd6a1b353b322a5b21a9839.png)

font:[https://julienbeaulieu . git book . io/wiki/sciences/machine-learning/decision-trees](https://julienbeaulieu.gitbook.io/wiki/sciences/machine-learning/decision-trees)

**提示！！！**:寻找算法中最重要的东西，并专注于它。没有必要知道所有的超参数，但有些是**必须正确使用算法。**

# 稳定性:

不要选择在交叉验证中产生不稳定结果的超参数。稳定性是最重要的，有助于维护部署后模型的健壮性。

**提示！！！**:一个做得很好的 K 折几乎“解决”了一切！。

![](img/bf44d18493c67a3f608594d489f204a3.png)

font:[https://sci kit-learn . org/stable/auto _ examples/model _ selection/plot _ nested _ cross _ validation _ iris . html](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

# 可复制性和可解释性

**别忘了种子！！！**

你必须诠释你的模型！！！

**别忘了种子！！！**

你必须解释你的模型！！！

**别忘了种子！！！**

***不可复制/可解释的模型是没有用的。***

![](img/24aacd3fe1b44d8938051541a0bb20e6.png)

font:[https://nymag . com/intelligence r/2017/02/roll-safe-the-guy-tapping-head-meme-explained . html](https://nymag.com/intelligencer/2017/02/roll-safe-the-guy-tapping-head-meme-explained.html)

# 决赛:

始终选择一组超参数，使误差最小化，同时缩小训练和测试基础之间的性能差距。当一个结构被观察时，它的行为是一种方式，而当它不在样本中时，它的行为是另一种方式。

![](img/e41809a863d0a75dc0c38264ff85d5ef.png)

font:[https://MC . ai/early-stopping-with-py torch-to-inhibit-your-model-from-over fitting/](https://mc.ai/early-stopping-with-pytorch-to-restrain-your-model-from-overfitting/)

# 动手:

基于树的算法超参数调整。访问链接:

*   [决策树](https://www.thiagocarmonunes.com.br/artigos/HyperParOptim/Tree/)
*   [LGBM](https://www.thiagocarmonunes.com.br/artigos/HyperParOptim/LGBM/)
*   [XGB](https://www.thiagocarmonunes.com.br/artigos/HyperParOptim/XGB/)

# 结论:

涵盖了超参数优化的基本概念以及如何使建模过程中这项极其重要的任务更加有效的技巧。

理论到位了，现在我们去实践吧。我将不同技术中的超参数优化的例子分开，以固定内容并查看它在实践中如何工作。

# 反馈:

感谢阅读。如果您希望通过[发送反馈，请联系](https://www.thiagocarmonunes.com.br/sobre/#contact)

# 参考资料:

[1]门户操作。Estimando os Parâ metros dos Modelos。[门户动作:Estimando OS parmetros dos Modelos](http://www.portalaction.com.br/confiabilidade/42-estimando-os-parametros-dos-modelos#:~:text=Os%20modelos%20probabil%C3%ADsticos%20apresentados%20na,por%20quantidades%20desconhecidas%2C%20denominadas%20par%C3%A2metros.&text=Essas%20quantidades%20conferem%20uma%20forma%20geral%20aos%20modelos%20probabil%C3%ADsticos.)

[2]维基百科超参数。[维基百科:超参数做链接](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))

[3]理解超参数及其优化技术。[了解超参数及其优化技术](https://mc.ai/understanding-hyperparameters-and-its-optimisation-techniques/)

![](img/29acc054a145a0abe9836cbb94a971d5.png)

font:[https://momo tattoo . com . br/2013/03/12/isso-e-tudo-pes soal-por-hoje/](https://momotattoo.com.br/2013/03/12/isso-e-tudo-pessoal-por-hoje/)