# 用 keras-rl 2020 实现 DQN、双 DQN 和决斗 DQN

> 原文：<https://medium.com/analytics-vidhya/implementation-of-dqn-double-dqn-and-dueling-dqn-with-keras-rl-2020-bc55339b21a9?source=collection_archive---------9----------------------->

> 查看完整的实现代码:
> 
> 点击此处！

# 双 Q 学习

我们刚刚建立的标准 Q 学习模型的另一个增强是双 Q 学习的想法，这是由 Hado van Hasselt (2010 年和 2015 年)提出的。这背后的直觉很简单。回想一下，到目前为止，我们使用贝尔曼方程来估计每个状态-动作对的目标值，并检查在给定状态下我们的预测有多离谱，如下所示:

然而，以这种方式估计最大预期未来回报会产生一个问题。您可能已经注意到，目标方程中的 max 运算符( *yt* )使用相同的 Q 值来评估给定的动作，这些 Q 值用于预测采样状态的给定动作。这引入了高估 Q 值的倾向，最终甚至失控。为了弥补这种可能性，Van Hasselt 等人(2016)实施了一个模型，将行动的选择与其评估分离开来。这是通过使用两个独立的神经网络实现的，每个神经网络都被参数化以估计整个方程的子集。第一个网络的任务是预测在给定状态下要采取的行动，而第二个网络用于生成目标，在迭代计算损失时，通过该目标来评估第一个网络的预测。尽管每次迭代的损失公式不变，但给定状态的目标标号现在可以用增广的双 DQN 方程来表示，如下所示:

正如我们所看到的，目标网络有自己的一组参数需要优化，(θ-)。这种从评估中分离出行动选择的方法已经被证明可以补偿天真的 DQN 所学到的过于乐观的表现。因此，我们能够更快地收敛我们的损失函数，同时实现更稳定的学习。

在实践中，目标网络的权重也可以是固定的，并且缓慢地/周期性地更新，以避免由于不良反馈回路(在目标和预测之间)而使模型不稳定。这种技术在另一篇 DeepMind 论文(Hunt，Pritzel，Heess 等人，2016 年)中得到推广，该方法被发现可以稳定训练过程。

【https://arxiv.org/pdf/1509.02971.pdf**Hunt、Pritzel、Heess 等人的 DeepMind 论文《深度强化学习的连续控制》，2016。**

您可以通过 keras-rl 模块实现双 DQN，使用的代码与我们之前训练太空入侵者代理时使用的代码相同，只需对定义 DQN 代理的部分稍加修改:

```
double_dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               policy=policy,
               memory=memory,
               processor=processor,
               nb_steps_warmup=50000,
               gamma=.99, 
               target_model_update=1e-2,
               train_interval=4,
               delta_clip=1.,
               enable_double_dqn=True,
              )
```

我们只需将 enable_double_dqn 的布尔值定义为 True，就万事大吉了！或者，您可能还希望试验预热步骤的数量(即，在模型开始学习之前)和目标模型更新的频率。

# 决斗网络架构

我们将实现的 Q-learning 架构的最后一个变体是决斗网络架构([https://arxiv.org/abs/1511.06581](https://arxiv.org/abs/1511.06581))。顾名思义，在这里，我们使用两个独立的状态值和状态-动作对值的估计器，形象地进行神经网络决斗。你应该还记得，在本章的前面，我们使用单个卷积流和密集连接层来估计状态-动作对的质量。然而，我们实际上可以将 Q 值函数分解为两个独立项的和。这种分离架构背后的原因是允许我们的模型分别学习可能有价值或可能没有价值的状态，而不必专门学习在每个状态下执行的每个动作的效果:

在上图的顶部，我们可以看到标准的 DQN 架构。在底部，我们可以看到决斗 DQN 架构如何分叉成两个独立的流，其中状态和状态动作值是单独估计的，没有任何额外的监督。因此，决斗 dqn 使用单独的估计器(即，密集连接层)来估计处于某个状态的值*【V(s)】*，以及在给定状态下执行一个动作优于另一个动作的优势*【A(s，A)】*。然后将这两项结合起来预测给定状态-动作对的 Q 值，确保我们的代理从长远来看选择最佳动作。虽然标准的 Q 函数 *Q(s，a)* 只允许我们估计给定状态的选择动作的值，但是我们现在可以分别测量状态的值和动作的相对优势。在执行一个动作没有以足够相关的方式改变环境的情况下，这样做是有帮助的。

值和优势函数都在下面的等式中给出:

DeepMind 的研究人员(王等，2016)在一个早期的赛车游戏(Atari Enduro)上测试了这样的架构，在这个游戏中，智能体被指示在一条有时可能会出现障碍的道路上行驶。研究人员注意到状态价值流如何学习关注道路和屏幕上的分数，而行动优势流只会在游戏屏幕上出现特定障碍时学习关注。自然地，只有当障碍物在其路径上时，代理执行动作(向左或向右移动)才变得重要。否则，向左或向右移动对代理没有重要性。另一方面，对于我们的代理人来说，保持对道路和分数的关注总是很重要的，这是由网络的状态价值流来完成的。因此，在他们的实验中，研究人员展示了这种架构如何能够导致更好的政策评估，特别是当一个代理面临许多具有类似后果的行为时。

我们可以使用 keras-rl 模块来实现决斗 dqn，解决我们之前看到的空间入侵者问题。我们需要做的就是重新定义我们的代理，如下所示:

```
dueling_dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               policy=policy,
               memory=memory,
               processor=processor,
               nb_steps_warmup=50000,
               gamma=.99, 
               target_model_update=10000,
               train_interval=4,
               delta_clip=1.,
               enable_dueling_network=True,
               dueling_type='avg'
              )
```

这里，我们只需将布尔变量 enable_dueling_network 参数定义为 True，并指定一个决斗类型。

> 查看完整的实现代码:
> 
> 点击此处！

[](/@madeshselvarani/implemtationactor-critic-with-keras-rl-2020-955f07fc8d34) [## 与 Keras-Rl 2020 一起实现演员-评论家

### 签出以实现:

medium.com](/@madeshselvarani/implemtationactor-critic-with-keras-rl-2020-955f07fc8d34) [](/@madeshselvarani/how-to-superimage-resolution-with-autoencoder-keras-2020-1dacfcbc23da) [## 如何使用 autoencoder 增强图像分辨率— keras 2020

### 什么是超图像分辨率？从一幅或多幅低分辨率图像中获取一幅或多幅高分辨率图像

medium.com](/@madeshselvarani/how-to-superimage-resolution-with-autoencoder-keras-2020-1dacfcbc23da) [](/@madeshselvarani/most-easiest-way-to-learn-multi-scale-transfer-learning-keras-2020-3fecf1d4acf3) [## 学习多尺度迁移学习的最简单方法 keras-2020

### Keras 框架将是整个代码的基本结构，转移学习。很少有深度学习模型有更多的…

medium.com](/@madeshselvarani/most-easiest-way-to-learn-multi-scale-transfer-learning-keras-2020-3fecf1d4acf3)