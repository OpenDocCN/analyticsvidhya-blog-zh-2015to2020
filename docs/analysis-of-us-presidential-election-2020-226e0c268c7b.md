# 2020 年美国总统大选分析

> 原文：<https://medium.com/analytics-vidhya/analysis-of-us-presidential-election-2020-226e0c268c7b?source=collection_archive---------20----------------------->

哈什塔·加尔格

2020 年美国总统大选充满了戏剧、激动、困惑和争议。尽管特朗普继续拒绝承认，但全世界都在屏息注视着乔·拜登被宣布获胜。

![](img/8d55c5350cb79bb6f1cf733b55405dde.png)

这里提出的分析试图理解不同年龄、种族和教育水平的人对两位选举候选人唐纳德·特朗普和乔·拜登的感受。

我们在这里试图回答的问题是:

1.  选民的感觉和他们的人口统计数据如年龄、性别、种族和教育有什么有趣的模式吗？
2.  选民对不同种族和族裔的感觉会影响他们对特朗普/拜登的感觉吗？

## **数据**

本分析的数据来自美国国家选举研究网站(ANES)。ANES 通过对投票、公众意见和政治参与等一系列主题进行在线调查，产生高质量的数据。

我们分析的第一步是将数据读入 r 数据帧。数据集可以从[这里](https://electionstudies.org/data-center/2020-exploratory-testing-survey/)下载。

```
data_2020 <- read.csv(“C:/Users~/anes_pilot_2020ets_csv.csv”, na.strings=c(“”,” “,”NA”))
```

在该数据集中，受访者对这些调查中不同人/问题的感受由感受温度计描述，数值范围为 0-100。在这篇文章中，我们将使用缩写 ft 来表示感情温度计。

data_2020 中共有 3080 行，365 个预测变量。但是我们只对感觉温度计栏和个人资料感兴趣，比如年龄、种族、教育和性别。现在让我们创建一个新的数据框，其中包含我们感兴趣的所有列。

```
col_2020 <- c("fttrump1", "ftobama1", "ftbiden1","ftblack", "ftwhite", "fthisp", "ftfeminists", "age", "sex", "race7", "educ")
short_2020 <- data_2020[,col_2020]
```

快跑！complete.cases 命令告诉我们数据集中没有缺失值。但在 short_2020 中，ft 列的某些值具有无效值 999。让我们去掉这些行。

```
short_2020 <- short_2020 %>% filter_all(all_vars(. !=999 ))
```

## 探索性分析和可视化

让我们画一些有见地的图，并尝试更好地理解我们的数据。

```
ggplot(short_2020) +
  geom_histogram(aes(x=fttrump1,fill = "green", alpha = 0.3)) +
  geom_histogram(aes(x=ftbiden1,fill = "blue", alpha = 0.3), position = "identity") + 
  scale_fill_manual(name="FT", values=c("green","blue"),labels=c("Trump","Biden"))  +
  ggtitle( "Feelings Thermometer") 
```

![](img/a8ef582eb118ee5d247d3b8f419e1639.png)

这是特朗普和拜登的感情分布图。在这些直方图中，我们可以看到，对拜登有最热烈(100 度)感情的受访者比特朗普多，但对拜登和特朗普都有冷淡(0 度)感情的人也有很大一部分。比特朗普更多的人对拜登有 0 的感觉。但总体而言，拜登的感觉比特朗普更好。

让我们试着画出所有数据变量之间的相关图。

```
library(corrplot)
corrplot(cor(short_2020), method = "circle", type = "upper")
```

![](img/a0168b2a85a0748df59d3f9560b70e55.png)

这个图给了我们一些有趣的见解，比如:

1.  对特朗普和拜登的感觉与对特朗普和奥巴马的感觉之间存在负相关。这意味着投票认为对特朗普有热情的人往往对奥巴马和拜登有冷淡的感觉，反之亦然。
2.  对奥巴马和拜登的感觉是正相关的。这意味着随着受访者对奥巴马的感觉增加，他们对特朗普的感觉也增加。
3.  对女权主义者的感觉和对特朗普的感觉之间有一个小的负相关。暗示支持女权主义者的人一般不支持特朗普，反之亦然。

这些见解可以通过对特朗普和拜登投票率高的人绘制数据来进一步证实。为此，我们要先做一个对特朗普的好感度≥ 75 度的人群子集。然后我们画出这些人对不同种族的感觉，如黑人、白人、西班牙人和女权主义者。我们对拜登也做了同样的事情，得到了下面的情节。

```
#subset of people who voted > 74 degrees for Trump
trump_high_2020 <- short_2020[which(short_2020$fttrump1 > 74),]
#Draw frequency plots for ftblack, ftwhite, fthisp and ftfeminists
p1 <- ggplot(trump_high_2020)+ geom_freqpoly(aes(x= ftblack,color= "ftblack"), size = 1.5) + geom_freqpoly(aes(x = ftwhite, color = "ftwhite"), size = 1.5) + geom_freqpoly(aes(x = fthisp, color ="fthisp" ), size = 1.5) +geom_freqpoly(aes(x = ftfeminists, color = "ftfeminists"), size = 1.5)
# Add labels and colors 
trump <- p1 + scale_color_manual(name = "FT", values = c(ftblack = "black", ftwhite = "yellow", fthisp = "green", ftfeminists = "red")) +labs(title = "People who voted high for Trump", x = "Feelings Thermometer")biden_high_2020 <- short_2020[which(short_2020$ftbiden1 > 74),]
p1 <- ggplot(biden_high_2020)+ geom_freqpoly(aes(x= ftblack,color= "ftblack"), size = 1.5) + geom_freqpoly(aes(x = ftwhite, color = "ftwhite"), size = 1.5) + geom_freqpoly(aes(x = fthisp, color ="fthisp" ), size = 1.5) +geom_freqpoly(aes(x = ftfeminists, color = "ftfeminists"), size = 1.5)
biden <- p1 + scale_color_manual(name = "FT", values = c(ftblack = "black", ftwhite = "yellow", fthisp = "green", ftfeminists = "red")) + labs(title = "People who voted high for Biden", x = "Feelings Thermometer")#plot the 2 graphs together
grid.arrange(trump, biden, ncol = 2)
```

![](img/a15eec561b0fb17f581c82a84d98ea4b.png)

上述情节表明，对特朗普感到温暖的人通常对白人和黑人都有温暖的感觉。他们对西班牙裔有着复杂的感情，对女权主义者普遍冷淡。另一方面，对拜登投了高票的人，大多对黑人、白人和拉美裔种族有好感，对女权主义者有复杂的感情。但总的来说，拜登的支持者比特朗普的支持者更支持女权主义者。

现在让我们试着找出不同种族的人对特朗普的看法。可以使用下面的代码来绘制关于种族变量的 fttrump 图。

```
#convert race to a categorical variable
short_2020$race7 <- as.factor(short_2020$race7)
#getting rid of races with very few data points
less_races <- short_2020 %>% filter(short_2020$race7 != 7 & short_2020$race7 != 9)
#draw the plot
p <- ggplot(less_races, aes(x = fttrump1, fill = race7))+
  geom_density(alpha = 0.4) + labs(title = "Feeling Thermometer by Race")
#fix the legend
p + scale_fill_discrete(name = "Race", labels = c("White","Black", "Asian","Mixed","Hispanic","American Indian"))
```

我们得到如下的情节:

![](img/63fd6df1626dc054820e35a3d549f557.png)

这个情节显示了不同种族对特朗普的感受的一般模式。这表明，黑人社区往往对特朗普有更冷的感觉，而白人则有相对温暖的感觉。其他种族有着复杂的感情，但总的来说，来自亚洲、混血和西班牙裔种族的人对特朗普的感觉更冷而不是温暖。

为拜登绘制了一个类似的感觉温度计图:

![](img/6fadf2c5038acadd65662f8a989e08cf.png)

该图显示大量白人对拜登的好感度为 0 度，而大量黑人和拉美裔对他的好感度普遍较高。亚裔和混血儿对拜登有着复杂的感情，但总的趋势是温暖多于寒冷。

现在，让我们试着看看是否能找出受访者的教育模式以及他们对两位候选人的感受。为此，我们先把他们按教育程度分组，找出他们的刻薄情绪。然后将这些数据绘制在图表上。

```
#Group by education and find mean
educ_ft <- short_2020 %>%  group_by(`educ`, educ)%>%   summarise_at(vars("fttrump1", "ftbiden1"), mean)
#convert mean to factor
educ_ft$educ <- as.factor(educ_ft$educ)
#Plot the data
p1 <- ggplot(educ_ft)+geom_point(aes(x= educ, y = fttrump1), shape = 19, size = 4, color = "red") + geom_point(aes(x=educ, y = ftbiden1), shape = 15, size = 4, color = "green") 
#Display education levels on the plot
p2 <- p1+  scale_x_discrete(labels = c("<= 12th grade", "High School Diploma", "No degree", "Associate degree", "Bachelor's degree","Master's Degree", "Professional Degree", "Doctorate")) 
#display titles and axes labels
p2 + theme(axis.text.x=element_text( size=11, angle=30, vjust=.8, hjust=0.8))+labs(title = "Mean Feelings by Education") + xlab ("Education")+ ylab("Mean feelings")
```

![](img/1e4e5f315ac20113e693d98feaec8801.png)

这张图表显示，拥有硕士、专业学位和博士学位等较高教育水平的受访者对拜登更有好感。另一方面，特朗普更多地受到那些一直学习到 12 年级并拥有高中文凭/副学位的人的支持。

现在，我们想按性别绘制英国《金融时报》的图表。因为我们希望在同一个情节中，对特朗普和拜登的感情都与性相对应，所以我们使用了库 reshape 中的融化功能。你可以在这里阅读更多关于这个功能的信息[。](https://www.r-bloggers.com/2012/04/melt/)

```
short_2020$sex <- as.factor(short_2020$sex)
library(reshape)
#melt the data
m_2020 <- melt(short_2020, id.vars = 'sex', measure.vars = c('fttrump1', 'ftbiden1'))
#plot the boxplots
p1 <- ggplot(m_2020) + geom_boxplot(aes(x=sex, y=value, fill=variable))
#fix axes and titles of the plot
p1 + scale_x_discrete(labels = c("Male", "Female"))+ labs(title = "Feelings by sex", fill = "Feelings Thermometer" )
```

![](img/ccc328df35280858cd1e94f472b6ef1c.png)

女性对特朗普的平均感觉比男性对特朗普的平均感觉低很多。对拜登来说，男性和女性的平均感受大致相同(接近 50 度)。

最后，让我们绘制 ft 与年龄的关系图，看看我们是否能找到任何模式。一个简单的按年龄划分的感情温度计散点图没有给出任何有用的结果。因此，年龄被划分为 10 岁的年龄组，并根据他们的感受总和进行标绘。

```
#create age range
short_2020 <- short_2020%>%mutate(AgeRange = cut(age, breaks = c(18,30,40,50,60,70,80,90)))
#plot feelings thermometer by age
p1 <- ggplot(short_2020, aes(x=AgeRange, y = fttrump1, fill = “green”, alpha = 0.3)) +geom_bar(stat = “identity”) 
p2 <- p1 + geom_bar(aes(x = AgeRange, y = ftbiden1, fill = “blue”, alpha = 0.3), stat = “identity”)
#fix the labels
p2 + scale_fill_manual(name=”FT”, values=c(“green”,”blue”),labels=c(“Trump”,”Biden”))+ ggtitle( “Feelings Thermometer by Age”)
```

![](img/a1d0536013dbbd72bc1111d767aef22a.png)

这一情节表明，比拜登更年轻的人往往对特朗普更有好感。60-70 岁和 80-90 岁年龄段的人更喜欢拜登，而不是特朗普。

## 结论

在绘制了所有图表并仔细分析了给定数据后，我们得出了以下结论:

1.  特朗普有很多白人支持者，但黑人支持者不多。另一方面，拜登有大量来自黑人的支持者，同样多的白人不支持他。
2.  随着教育水平的提高，人们支持拜登的机会也增加了。
3.  相对年轻的选民更喜欢特朗普而不是拜登，特朗普的女性支持者少于男性。
4.  对特朗普感到温暖的人也对白人和黑人社区有温暖的感觉，但对女权主义者没那么多感觉。

在本文的下一部分，我们将尝试分析从 2016 年到 2020 年的四年中，选民对特朗普的感受是如何变化的。