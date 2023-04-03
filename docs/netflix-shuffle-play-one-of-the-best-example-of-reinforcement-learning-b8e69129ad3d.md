# 网飞洗牌游戏:强化学习的最好例子之一。

> 原文：<https://medium.com/analytics-vidhya/netflix-shuffle-play-one-of-the-best-example-of-reinforcement-learning-b8e69129ad3d?source=collection_archive---------8----------------------->

正如你们大多数人可能已经看到的，网飞最近在智能电视屏幕上实现了“随机播放”，这将帮助你找到你可能喜欢的下一个系列。这是为了在这个网飞家庭观看时间处于高峰的 covid 时间帮助狂欢的观众，但仍然有许多用户在努力寻找好的电影/连续剧来狂欢观看。

这是强化学习的一个经典例子，人们可以看到它的实际应用。在这篇文章中，我将使用网飞洗牌游戏的例子来解释强化学习的概念，以便于理解。到目前为止，这是我所经历的关于强化学习的最好也是最简单的例子。

**什么是强化学习？**

强化学习(RL)是机器学习的一个领域，涉及软件代理应该如何在环境中采取行动，以最大化累积回报的概念。那么在网飞的背景下，什么是媒介，什么是环境？强化学习的不同实体是主体、环境、动作、内部状态和奖励函数

代理:代理是正在观看网飞并且正在使用随机播放来识别下一系列/节目的用户(可能是你)

环境:这是你的电视和你的网飞应用，它背后的所有算法都在迎合你的需求。

动作:动作是每次你点击“随机播放”，那是你在环境上执行的一个动作

奖励:奖励是什么，是你看一个新推荐的剧集被播放的时间量，是几秒或者几分钟的事情。你看的时间越多，奖励越多，所以算法的目标是最大化奖励

内部状态:内部状态与应用或环境不同，它是当前的推荐状态，即它现在向你推荐的是什么类型的类型，它向你展示的是什么系列，以及它收到的奖励是什么。我们稍后会详细讨论它

**强化学习是如何工作的？**

下图显示了强化学习在洗牌例子中是如何工作的。

![](img/87bd98784d2739c823bc3b6e1d885924.png)

图片来源(网飞和 CNET)

工作原理:

1)当观看网飞的用户点击随机播放时，这是一个动作，该动作被发送到环境(网飞 APP ),环境初始化内部状态

2)这向用户推荐了一个结果，并且该奖励被测量为用户观看电影所花费的时间量。

3)算法的目标是最大化回报，即用户在观看推荐的电影或连续剧上花费的时间。RL 算法用于识别和维护内部状态的一种常用方法是“马尔可夫决策过程”。

4)马尔可夫决策过程(MDP)是一种离散时间随机控制过程。它提供了一个数学框架，用于在结果部分随机、部分在决策者控制下的情况下对决策进行建模

![](img/fef88dbbf5685b2d2a32b0b85c90b26a.png)

来源(维基百科)

5)算法识别下一个推荐，例如:如果奖励非常少，如用户立即改变电影，这意味着模型采用的早期路径是错误的，这提示模型选择不同的类型，即不同的路径以到达具有潜在高观看时间的电影。

6)如果用户观看电影一段合理的时间(几分钟，意味着奖励比先前的场景更好),那么路径是部分正确的，并且识别出下一个可能的选项。这可以解释为体裁可能是正确的，但分性质的选择是错误的。持续的学习过程将有助于为用户识别正确的电影

**注:**以上评估纯属我个人观点，意图仅在于解释强化学习。显然，我们永远不知道网飞使用的方法。

**参考文献:**

[https://www . analyticsvidhya . com/blog/2017/01/introduction-to-enforcement-learning-implementation/](https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/)

[https://www . cs . Toronto . edu/~ zemel/documents/411/rl tutorial . pdf](https://www.cs.toronto.edu/~zemel/documents/411/rltutorial.pdf)

[https://wiki.pathmind.com/deep-reinforcement-learning](https://wiki.pathmind.com/deep-reinforcement-learning)

[https://en.wikipedia.org/wiki/Reinforcement_learning](https://en.wikipedia.org/wiki/Reinforcement_learning)

[https://en.wikipedia.org/wiki/Markov_decision_process](https://en.wikipedia.org/wiki/Markov_decision_process)

[https://deep sense . ai/what-is-reinforcement-learning-the-complete-guide/](https://deepsense.ai/what-is-reinforcement-learning-the-complete-guide/)