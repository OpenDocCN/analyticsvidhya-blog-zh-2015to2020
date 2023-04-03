# 重温蒙提·霍尔悖论——其背后的直觉和 R

> 原文：<https://medium.com/analytics-vidhya/the-monty-hall-paradox-reconsidered-the-intution-behind-it-and-a-simulation-in-r-4110c145d59c?source=collection_archive---------1----------------------->

![](img/92da5dd4963a9ece26db9e22d82cc741.png)

骑特斯拉的山羊；来源:https://www.youtube.com/watch?v=8i9nYbCUIO4

在过去的几年里，我对一个真正反直觉的悖论进行了多次讨论:蒙蒂霍尔问题。这是如此反直觉，以至于有些人不相信你——即使你向他们展示了数学证明。

昨天再次讨论后，我决定把我认为最直观的解释放到一篇博客文章中，并在 R 中添加一个模拟，清楚地显示哪个解决方案是正确的。

# 蒙蒂·霍尔问题

你是一个游戏节目的参与者。

游戏节目主持人给你三扇门，你必须选择一扇。其中一扇门后是一辆特斯拉，另外两扇门后是山羊。

当你选择一扇门后，游戏节目主持人会打开另一扇门，门后有一只山羊。然后他问你:“你愿意重新考虑，走另一扇门吗？”。

你应该坚持你最初的选择还是应该改变？

大多数人会想，没关系(把游戏节目主持人潜在的心理诡计排除在外):毕竟，你现在有两个门可以选择，赢得特斯拉的概率应该是 50%，对吗？

不对！如果你换了门，你有 66.66 %的机会获胜，相比之下，如果你坚持最初的选择，你有 33.33 %的机会获胜。但是为什么呢？

# 直觉

即使你向人们展示模拟结果(见下文),他们仍然不敢相信。还剩两扇门。概率应该是 50%。

不幸的是，大多数解释也不直观。所以这是我给出一个更好的答案的机会(我还没有在任何地方读到过)。

**从游戏节目主持人的角度考虑游戏，而不是你的角度！你是游戏节目主持人，你不想失去这辆车。**

这里是*你的*规则:参赛者选择一扇门，然后你必须打开一扇没有奖品的门。那么对方要么换，要么不换。

在第一次选择对手之前，你有 1/3 的机会输。但是你必须打开一扇门。现在有以下选项:

1.  要么是对手拿着奖品挑的门(33.33%的发生概率)。然后你打开另外两扇门中的任何一扇门——哪一扇门并不重要。
2.  或者对手挑了一个有山羊的门(66.66%发生几率)。然后你要用山羊打开另一扇门。**所以你基本上放弃了正确的解决方案——>你没有打开的第三扇门。**

现在让我们来看看对手的两个策略:

假设*对手不切换*。你真幸运。他赢的概率还是 1/3，因为你开门对他来说改变不了什么。

但是现在考虑他们交换:你有 1/3 的机会，他们犯了一个错误，因为他们从一开始就有正确的门。但是有 2/3 的可能性是他们最初选错了门，但这意味着你必须放弃正确的门！

用更简单的话来说:如果对手的策略是总是切换，那么你必须在三分之二的时间里放弃正确的解决方案。作为游戏节目主持人，你只能希望他们从一开始就选择正确的门(这种情况只发生 33.33%。那么你赢了。)

# 模拟一下

让我们用一个模拟来证明它实际上是正确的:交换产生 2/3 的获胜机会，而不是 50%。

首先，我们需要一个游戏机制的函数。这很简单。在第一步中，你随机选择三个门中的一个。然后，其中一个门(不包含特斯拉)被移除。该函数带一个参数 *switch —* 表示是否切换*。*

```
# This is a function to simulate the Monty-Hall-Gameshow:
gameshow <- function(switch = FALSE){

  #Step 1: Guess

  gates <- c(1:3)
  answer <- sample(gates, 1, replace = TRUE)
  guess <- sample(gates, 1, replace = TRUE)

  #Step 2: Open First Gate
  if(guess != answer){
    removed_gates <- c(guess, answer)
  }
  if(guess == answer){
    removed_gates <- c(answer, sample(gates[-answer], 1))
  }

  # Step 3: If strategy is "switch" --> new guess
  if(switch == TRUE){
    guess <- removed_gates[which(removed_gates != guess)]
  }
  # Step 3: Check if correct
  return(guess == answer)

}
```

然后我们再写一个函数模拟游戏 N 次。

```
##### Simulate the Gameshow with switch or stay strategy for N iterationssimulate_gameshow<- function(switch=FALSE, N = 10000){

  gameshow_results <- 0

  for(i in 1:N){
    if(gameshow(switch) == TRUE){
      gameshow_results <- gameshow_results + 1
    }
  }

  print(paste("your win percentage was ",gameshow_results/N*100, "%" ))
}
```

然后我们可以模拟 10，000 次，每一次都是你改变策略和坚持最初选择的策略:

```
> simulate_gameshow(**switch=FALSE**, N=10000)
[1] **"your win percentage was  33.58 %"**> simulate_gameshow(**switch=TRUE**, N=10000)
[1] **"your win percentage was  66.52 %"**
```

事实上:转换是将你的胜算提高到三分之二的最佳策略！