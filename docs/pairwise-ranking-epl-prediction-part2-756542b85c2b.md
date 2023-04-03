# 成对排序:EPL 预测——第二部分

> 原文：<https://medium.com/analytics-vidhya/pairwise-ranking-epl-prediction-part2-756542b85c2b?source=collection_archive---------28----------------------->

# 概观

成对排名在搜索排名优化、产品比较等领域有多种应用。与传统的累积比较相比，我们可以说 A 比 B/C/D 好 n%,等等..还通过相应的相关性对每个变量的影响进行优先排序。该方法背后的理论在本系列的上一篇文章中有详细介绍:[https://medium . com/@ David acad 10/pairwise-ranking-EPL-prediction-3ce 75575958](/@davidacad10/pairwise-ranking-epl-prediction-3ce755575958)

![](img/7e1f0918fe574c4c5842ea9d2a5ece27.png)

图片作者:thinkersnewsng

假设，我们现在对如何在幕后应用成对排名有了一个简单的想法，我们将有一个有趣的应用，我们试图预测 2020-21 赛季结束时英超联赛的积分榜。我们用来给球队排名的参数将是最近 5 场比赛的结果。当应用于 2019-20 赛季的验证时，看到它与真正的积分榜有多接近是非常有趣的。

# 诉讼案件

我们用来给球队排名的参数是每支球队在过去 5 场比赛中的比分。胜一场奖励 3，平一场，输 0，就像在 EPL 一样。所以一个团队的最高分是 15 分，最低分是 0 分。我们的目标是在赛季结束时预测最终排名，在赛季开始时甚至没有一场比赛。[我们可以尝试更实时的比赛模型，通过比赛预测稍后:D]。

正如在每个数据科学问题中一样，重要的部分是准备好数据。有相当多的好消息来源。我已经从 football.com 收集了 2004 年的比赛结果，并整理成一个文件。您可以在以下位置访问回购:

[https://github . com/David acad 10/EPL _ 分析/tree/master/Pairwise _ 排名。](https://github.com/davidacad10/EPL_Analytics/tree/master/Pairwise_Ranking.)

让我们加载数据并开始:

```
library(rio)
library(dplyr)
library(xgboost)
library(knitr)rm(list = ls())
data=import("Pairwise_Ranking_Data.xlsx")
data=data%>%
     mutate(Season_Start_Date=as.Date(Season_Start_Date)
          ,Season_End_Date=as.Date(Season_End_Date)
          ,Prev_SSD=as.Date(Prev_SSD)
          ,Prev_SED1=as.Date(Prev_SED1))
```

数据框具有来自 2004 赛季的 H2H 结果，评级如前所述。在赛季的第一周，H2H 是上赛季的。因此，对于每个赛季 N，排名将是球队在赛季 N+1 的最终排名。例如，在 2018-19 赛季结束时，切尔西队领先曼城队 9 分，领先利物浦队 6 分，但他们在接下来的 2019-20 赛季中获得了第三名。因此，对于团队 Ti，在赛季 N 结束时所有团队 T 的 H2H 将是用于预测赛季 N+1 中排名的参数。

让我们把数据分成训练和测试。在 2012-13 赛季之后，EPL 发生了变化，所有的球队都变得非常有竞争力。所以我们可以选取 2013-14 赛季到 2018-19 赛季的训练数据。我们可以测试为 2019-20 赛季创建的模型，因为许多原因(新冠肺炎，利物浦夺冠等)，2019-20 赛季本身可以被称为异常赛季。:D)。

```
train=data%>%filter(Prev_SSD<=as.Date('2018-08-10'))%>%
      filter(Prev_SSD>=as.Date('2013-08-01'))

test1=data%>%filter(Prev_SSD>as.Date('2018-08-10'))##Getting the previous records available for the promoted teams
##And adding into the 17 other teams to have the whole 20 team ##recordspromoted=c("Norwich","Sheffield United","Aston Villa")test2=data%>%filter(Team1%in%promoted)%>%
   ungroup()%>%
   group_by(Team1)%>%
   arrange(desc(Season_Start_Date))%>%
   mutate(rpp=row_number())%>%
   filter(rpp==1)%>%
   select(-rpp)test=bind_rows(test1,test2)

train_data=train%>%ungroup()%>%select(Arsenal:Wolves)
test_data=test%>%ungroup()%>%select(Arsenal:Wolves)
target=as.numeric(as.character(train$Rank))set.seed(1000)
xgbTrain <- xgb.DMatrix(as.matrix(train_data), label = train$Rank)
xgbTest <- xgb.DMatrix(as.matrix(test_data))##No extra parameter tuning being applied except ntress=5000 and
#early stopping round as 20 
params <- list(booster = 'gbtree',
                objective = 'rank:pairwise')

##Metric used is NDCG as described in tutorial
rankModel <- xgb.train(params, xgbTrain, 5000, watchlist = list(tr = xgbTrain), eval_metric = 'ndcg',early_stopping_rounds = 20)## Stopping. Best iteration:
## [55] tr-ndcg:0.983104##Predicting The Model on Test Data for 2019-20pred=(predict(rankModel, xgbTest,reshape = TRUE))

test$Pred=pred
test_pred=test%>%
   ungroup()%>%
   select(Team1,Prev_SSD1,Rank,Pred)%>%
   arrange(Pred)%>%
   mutate(Pred_Rank=row_number(),
          Probability=1/(1+exp(Pred)))%>%
   select(Prev_SSD1,Team1,Rank,Pred_Rank,Pred,Probability)%>%
rename(Actual_Rank=Rank,Title_Probability=Probability,Season_Start=Prev_SSD1)##Print the prediction frame
view(test_pred)
```

**2019–20 赛季预测**

*瞧*。我们能够正确地获得冠军和亚军。请注意，在所有的训练数据中，利物浦排名第一，但只有曼城、切尔西和莱斯特。该模型能够通过观察利物浦在单挑中的进步来发现他们能够超越的模式。我们 3 点面对托特纳姆，他们获得了第六名(有趣的是吧！).尽管这是一个淡季，上赛季他们甚至进入了前 4 名和 UCL 决赛，但他们经历了有史以来最好的一个赛季。考虑到这个离群值，曼联和切尔西的下一个排名是正确的。

lambda rank 方法为最高评级提供了比最低评级更多的普遍性。虽然，除了狼队(由于数据不多)，倒数 5 名中有 4 名处于降级战中，其中 2 名确实降级了。

现在我们已经看到了上赛季预测的准确性，让我们看看赛季末会是什么情况。

```
##Load the 2020–21 prediction data
newtest=import(“Pairwise_Ranking_Data_Pred_2020_data.xlsx”)
rel=c(‘Norwich’,’Watford’,’Bournemouth’)newtest=newtest%>%filter(!Team1%in%rel)
newtest2=newtest%>%select(-Team1)newtest_xgb <- xgb.DMatrix(as.matrix(newtest2))
pred=(predict(rankModel, newtest_xgb,reshape = TRUE))newtest$Pred=pred
newtest_pred=newtest%>%
             ungroup()%>%
             select(Team1,Pred)%>%
             arrange(Pred)%>%
 mutate(Prev_SSD1=as.character(‘2020–09–12’),Pred_Rank=row_number(),
 Probability=1/(1+exp(Pred)))%>%
 select(Prev_SSD1,Team1,Pred_Rank,Pred,Probability)%>%
 rename(Title_Probability=Probability,Season_Start=Prev_SSD1)##Print the 2020-21 season prediction
view(newtest_pred)
```

**2020–21 赛季预测**

注意到这个预测没有利兹，因为他们是 2002 年后第一次在 EPL。有 19 支球队参赛，切尔西有望夺冠，利物浦第二，曼联第三，曼城第四。随着切尔西的新签约，他们事实上可以瞄准它。虽然对于一个被足球困扰的人来说，很难接受曼城只获得第四名，而曼城或者利物浦没有赢得联赛冠军。但是数字不会说谎。祈祷:D

对于保级之战，预计将在富勒姆、西布朗、水晶宫和阿斯顿维拉之间展开。注意到这 4 人中有 3 人是在过去两个赛季中晋级的。

# 结论

该预测实际上可能存在一些偏差:

*   考虑中的参数是最近两个赛季的有效对抗。这还没有考虑到转会对球队变化的影响
*   更多的优先权被提供来正确地得到最高的等级超过最低的等级
*   像利兹这样的新加入 EPL 的球队没有相关的数据，因为我们认为 EPL 是一对一的，所以不能加入

既然我们已经完成了一个很酷的成对排序应用程序，我相信我们在应用程序方面也做得很好。事实上，除了 xgboost 之外，还有许多其他的包可供使用，如果您感兴趣的话，可以选择它们。如果你想集思广益，看看这篇文章

[https://medium . com/@ David acad 10/pairwise-ranking-EPL-prediction-3ce 755575958](/@davidacad10/pairwise-ranking-epl-prediction-3ce755575958)