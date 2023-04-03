# 2018-2019 年 NBA 球员统计数据——用 Python 编写！

> 原文：<https://medium.com/analytics-vidhya/a-data-dive-into-2018-2019-nba-player-stats-in-python-d7a1c051ac5c?source=collection_archive---------0----------------------->

![](img/50aefd21e24ea3dfd1b9878ad87a0c1b.png)

主要的体育联盟是挖掘大量数据的成熟领域——在本文中，我们只触及表面。

当我开始思考我的第一篇数据分析博文的主题时，可能性似乎是无限的。然而，随着所有关于高知名度篮球运动员转移到新球队(如科怀·伦纳德和拉塞尔·威斯布鲁克，仅举几例)的新闻，很快就清楚了，使用 NBA 统计数据的数据分析项目将是一个很好的起点！

**NBA 统计数据来源**

在我寻找高质量数据进行分析的过程中，我遇到了一个非常有用的网站，叫做 Basketball Reference(Basketball-Reference . com)。篮球参考提供了一个关于各种篮球统计数据的公共可用数据的大缓存。为了将这个数据分析项目缩小到一些小而有趣的东西，我将我的分析集中在一个表格上，其中包括每个球员在整个 2018-2019 赛季的统计数据:

![](img/1d8b55d2d7572786d257d0a26440031f.png)

链接:[https://www . basketball-reference . com/联赛/NBA_2019_totals.html](https://www.basketball-reference.com/leagues/NBA_2019_totals.html)

该表有 530 行(当隐藏赛季中转换球队的球员的部分行时)，每行代表与单个球员相关的数据和统计。例如，在第 8 行 Rk=8(上图)与拉马库斯·阿尔德里奇相关联，并包括奥尔德里奇在 2018-2019 赛季提供的各种整体赛季统计数据，如他的出场次数(G)，上场时间(MP)，三分球命中率(3P%)，进攻/防守/总篮板数(ORB/DRB/TRB)，以及总得分(PTS)。

**现在我们已经得到了数据，该怎么办？**

使用存储在该表中的数据，我们可以做几件事情。一种选择是做一些探索性的数据分析，深入数据，看看是否有任何独特的模式或趋势出现。另一种选择是提出具体的问题(或测试假设)，然后通过客观地检查数据告诉我们的内容来尝试回答这些问题。为了对数据有一个感觉，我们将从第一个选项开始，然后一旦我们对数据感到满意，我们将开始问一些具体的，可验证的问题，这些问题可以使用我们的数据集来回答，例如:是否有可能预测球员的[位置](https://en.wikipedia.org/wiki/Basketball_positions)(即[控卫](https://en.wikipedia.org/wiki/Point_guard)(PG)[得分后卫](https://en.wikipedia.org/wiki/Shooting_guard)(SG)[小前锋](https://en.wikipedia.org/wiki/Small_forward)(SF)[大前锋](https://en.wikipedia.org/wiki/Power_forward_(basketball)) (PF)， 或者 [center](https://en.wikipedia.org/wiki/Center_(basketball)) (C))只使用该表中两列的数据，(例如，只查看球员整个赛季的总篮板和助攻)？ 在我们开始之前，让我们先熟悉一下这个表格中的数据。

**探索数据**

我们可以通过将表加载到 Python 环境中(作为. csv 文件，可从上面的 Basketball Reference 链接下载)来开始研究数据。在此之前，我们需要导入必要的 Python 库:

```
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

现在我们已经导入了库，我们可以阅读。包含我们的数据的 csv 文件，并使用以下代码随机打乱表中的行:

```
df=pd.read_csv("2018-2019_Overall_NBA_Stats_by_Player.csv")
df=df.sample(frac=1)
df = df[df.MP > 96]
```

请注意，上面的最后一行过滤掉了我们数据中所有上场时间少于 96 分钟的球员(因为一个球员上场时间少于 2 场比赛会过度扭曲我们的数据)。

所以我们的数据被加载并准备好进行一些初步分析。一个好的起点是查看一些与我们的数据相关的描述性统计数据。我们可以使用下面的代码行轻松查看 dataframe 对象的这些描述性统计信息(注意，我首先从原始 dataframe 中选择了一个列子集，以使输出更易于阅读):

```
df_small=df[["Pos","Age","TRB","AST","STL","BLK","PTS"]]
df_small.describe()
```

![](img/4d3c53889fb80cb771d0093f7070900d.png)

通过查看这个输出，我们可以看到 bat 的一些优点。例如，NBA 球员的平均年龄几乎是 26 岁，标准差大约是 4 岁。有趣的是，TRB、AST 和其他变量的最大值往往是其平均值的 10 倍 5X(例如，拥有最多盖帽的球员几乎是所有球员盖帽平均值的 10 倍)。这些值可能表明这些变量中的长尾分布，这可能至少部分是由往往远高于平均值的“超级明星”球员造成的。

理解我们的数据的一个障碍是我们的表有超过 10 列的整数或实值数据。为了更好地可视化三维或多维数据，我们可以使用 *seaborn* Python 库。具体来说，如果我们对数据使用 *seaborn* 包的 *pairplot* 函数，我们可以看到所有列对排列的数据的 2D 切片。同样，我选择了原始列的子集，以使输出更容易看到:

```
sns.pairplot(df_small, hue='Pos', size=2.5)
```

![](img/97dba4a0401682f9b16c2a246e8789a0.png)

非常酷！但是这个图表矩阵告诉了我们关于 NBA 球员的什么呢？首先，看看对角线，我们可以看到许多分布看起来是正尾分布，可能类似于伽玛或幂律分布。实际上，这些分布可能代表了 NBA 球员中“天赋”的分布，这可以通过更深入的分析来进一步探索。很多变量看起来也是正相关的，比如分和助攻。这个特点可能是这样一个事实的自然延伸，即一些球员与其他球员相比，总体上更好(或更差)。

我们怎样才能给这个数据展示增添一点趣味呢？在代表球员位置的数据上叠加一个颜色层怎么样？首先，让我们将“PF-SF”和“SG-PF”这样的混合位置转换为 5 个主要篮球位置之一，因此“Pos”列中的每个变量只是 5 个主要位置之一:

```
df=df.replace("PF-SF","PF")
df=df.replace("SG-PF","SG")
df=df.replace("SG-SF","SG")
df=df.replace("C-PF","C")
df=df.replace("PF-C","PF")
df=df.replace("SF-SG","SF")df_small=df[["Pos","Age","TRB","AST","STL","BLK","PTS"]]
```

太好了！现在让我们绘制它:

```
sns.pairplot(df_small, hue='Pos', size=2)
```

![](img/2564e822dd6d2e72a19a67d1481d92fd.png)

为篮球位置添加分类图层

好的，所以这里肯定有一些图案是可以视觉识别的。例如，看看 AST 与 TRB 的对比，我们可以看到 PG 球员(紫色)往往 TRB 较低，但 AST 较高，而 C 球员(红色)往往 TRB 较高，但 AST 较低。此外，对角线让我们对每个变量的分布有了更多的了解，但按篮球运动员的位置细分。因此，看左上角的对角线，我们可以看到 C 球员在 TRB 指标上比所有其他球员类别有更长的尾巴，而 PG 球员在 AST 指标上比所有其他球员类别有更长的尾巴。关于这种可视化的另一个显著特征是，PTS 分布在所有 5 个玩家位置上看起来*基本相同*。

那么，这些数据是否与关于不同篮球运动员位置表现的传统观念相一致呢？维基百科说“控卫(PG)…通常是球队最好的控球者和传球者。因此，他们经常在助攻方面领先他们的球队，并且能够为自己和队友创造投篮机会。”维基百科还说，“中锋(C)…通常在底线附近或靠近篮筐的地方打球…他们通常擅长抢篮板、抢球和掩护进攻。”让我们看看是否可以使用我们的数据来确认/测试这些断言！

在我们继续之前，先说几点——我们目前使用的数据是*标记的数据*,可能会在*监督学习*算法中使用。尽管我们已经知道当前数据集中位置的分类，但我们仍然可以尝试使用*无监督学习*技术建立一个预测模型。其中一种技术是流行的 K-means 聚类算法，它是使用 *sklearn* 库在 Python 中实现的。我们还必须为我们的模型选择正确的*特性*作为输入。

在查看了上面的一组图表之后，我们可以挑选一组列作为任务中的特征向量*。一个看起来特别有趣的图表是 AST 对 TRB 的图表。让我们从这里开始，看看我们的聚类算法是否能够正确地识别与每个球员位置相关联的聚类。此外，让我们将 AST 和 TRB 变量标准化，以更好地控制上场时间的变化，例如计算 AST/MP(每分钟助攻数)和 TRB/MP(每分钟总篮板数)。下面的代码片段将完成这些任务，并绘制出结果:*

```
df["TRB/MP"]=df["TRB"]/df["MP"]
df["AST/MP"]=df["AST"]/df["MP"]fig, ax = plt.subplots()x_var="AST/MP"
y_var="TRB/MP"colors = {'SG':'blue', 'PF':'red', 'PG':'green', 'C':'purple', 'SF':'orange'}ax.scatter(df[x_var], df[y_var], c=df['Pos'].apply(lambda x: colors[x]), s = 10)# set a title and labels
ax.set_title('NBA Dataset')
ax.set_xlabel(x_var)
ax.set_ylabel(y_var)
```

![](img/4ae0296dce3503c832bc344aa6643709.png)

每分钟助攻数与每分钟篮板总数的对比图

这里似乎有一些由我们的球员位置类别定义的聚类，所以让我们将 K-means 算法(为简单起见，假设 4 个聚类)应用于我们的(标准化)数据，并看看它将我们带到哪里！

```
dfn = df[["AST/MP","TRB/MP"]]kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 500, n_init = 10, random_state = 0)y_kmeans = kmeans.fit_predict(dfn)print(kmeans.cluster_centers_)
```

最后一行的输出如下:

```
[[0.07171785 0.33468039]
 [0.17600423 0.1335057 ]
 [0.06933687 0.12144473]
 [0.06577468 0.21234039]]
```

这 4 对点代表由算法识别的每个聚类的中心点(在 AST/MP 与 TRB/MP 空间中)。为了让它看起来更有趣，让我们把它形象化:

```
d0=dfn[y_kmeans == 0]
d1=dfn[y_kmeans == 1]
d2=dfn[y_kmeans == 2]
d3=dfn[y_kmeans == 3]
d4=dfn[y_kmeans == 4]Visualizing the clusters
plt.scatter(d0[x_var], d0[y_var], s = 10, c = 'blue', label = 'D0')
plt.scatter(d1[x_var], d1[y_var], s = 10, c = 'green', label = 'D1')
plt.scatter(d2[x_var], d2[y_var], s = 10, c = 'red', label = 'D2')
plt.scatter(d3[x_var], d3[y_var], s = 10, c = 'purple', label = 'D3')#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
```

![](img/0c350a7e54795c7c8a4c55d402541603.png)

不错！看起来我们的算法识别出了与实际玩家位置聚类至少在一定程度上接近的聚类(中心用黄色的大圆点表示)。我们能做得更好吗？让我们尝试一种称为 K-最近邻(KNN)的不同算法，它是一种*无监督学习*算法，因为该算法实际上使用篮球位置标签来基于最近邻点的数量对点进行分类:

```
neighb = KNeighborsClassifier(n_neighbors=4)
y_neighb = cluster.fit_predict(dfn, df["Pos"])d0=dfn[y_neighb == 0]
d1=dfn[y_neighb == 1]
d2=dfn[y_neighb == 2]
d3=dfn[y_neighb == 3]#Visualising the clusters
plt.scatter(d0[x_var], d0[y_var], s = 10, c = 'blue', label = 'D0')
plt.scatter(d1[x_var], d1[y_var], s = 10, c = 'green', label = 'D1')
plt.scatter(d2[x_var], d2[y_var], s = 10, c = 'red', label = 'D2')
plt.scatter(d3[x_var], d3[y_var], s = 10, c = 'purple', label = 'D3')
```

![](img/e70cd0447c1ce3d1100fbc7aeceff8ee.png)

显然，KNN 提出了类似的集群，但每个集群的边界不同。哪种方法比较好？很难判断，因为实际数据中的聚类有很大的重叠。不过，我们可以肯定的是，这两种分类算法仅仅基于两个变量就能够在一定程度上准确地识别聚类:AST/MP 和 TRB/MP。这是一个令人印象深刻的壮举！我们也可以证实我们最初的怀疑，球员位置等级如何与这两个变量相关联。

例如，查看 K 均值算法的聚类中心…

```
[[0.07171785 0.33468039]
 [0.17600423 0.1335057 ]
 [0.06933687 0.12144473]
 [0.06577468 0.21234039]]
```

看起来好像与 PG 球员相关联的集群位于[AST/MP = 0.176，TRB/MP = 0.133]，而与 C 球员相关联的集群位于[AST/MP = 0.071，TRB/MP = 0.334]。因此，我们可以说，我们的两种分类方法都符合传统智慧，即控卫(PG)球员“经常在助攻方面领先于他们的球队”，而中锋(C)球员“通常擅长抢篮板。”

其他三个位置 SG，PF，SG 呢？嗯，我们的分类方法似乎已经在一定程度上准确地将 SG 识别为它自己的簇(在上图的左下角)。但是 PF 和 SG 球员之间的重叠，至少在 AST/MP 与 TRB/MP 空间中，对于这两个组的分类来说太极端了。

所以让我们快速回顾一下我们发现中一些有趣的部分。根据我们的 K-means 算法，被归类为 C 位置的球员应该每三分钟抢一个篮板，而 SG 和 PG 位置每分钟抢的篮板几乎是 C 位置的三分之一！此外，考虑到我们数据中的重叠聚类，在我们的 AST/MP 与 TRB/MP 数据中似乎没有任何有意义的区别，使我们能够以任何显著的准确性水平确定一个球员是 PF 球员还是 SG 球员。

我们也可以看看哪些球员“看”或“出现”在比赛中，就像一个与实际比赛位置不同的位置。在浏览了 KNN 算法完成的分类后，我发现了以下有趣的瑰宝:

*   **马克加索尔(真位= C；打球喜欢= PG)** —虽然加索尔可以说是一个伟大的球员，但他在 2018-2019 赛季的篮板低于标准，至少与其他中锋(C)球员相比(只有 0.25 TRB/MP)。这一显著低于标准的 TRB/MP 值是我们的 KNN 分类方法将他归入“PG”类而不是“C”类的原因。
*   扬尼斯·阿德托昆博(真实位置= PF，打球方式= C) —人们称这家伙为“希腊怪胎”是有原因的维基百科称，“在 2016-2017 年，他在所有五个主要统计类别中领先雄鹿队，并成为 NBA 历史上第一个在总得分、篮板、助攻、抢断和盖帽方面进入前 20 名的常规赛球员。”詹尼斯每分钟的总篮板数(几乎 0.4 个 TRB/场均)明显高于很多中锋，比如赛尔吉·伊巴卡和卡尔-安东尼-唐斯。
*   **德雷蒙德·格林&乔·英格尔斯(真实位置= PF 打球就像= PG)**——乔·英格尔斯的维基百科页面直截了当地指出，乔“主要打小前锋和得分后卫的位置，*，但他足够多才多艺，可以打控球后卫，因为他经常打前锋的角色*关于德雷蒙德·格林，维基百科是这样说的:“格林……被认为是 NBA 新兴趋势的领导者之一，这种趋势是*多才多艺的前场球员能够打和防守多个位置*，为队友创造机会，并分隔地板。”
*   **拉塞尔·维斯特布鲁克(真实位置= PG 像詹尼斯一样，拉塞尔·维斯特布鲁克是一个真正的异数。你看到上面图表中最右边的孤独点了吗？那是拉塞尔·威斯布鲁克——他是独一无二的！他每分钟的助攻数在联盟领先(几乎 0.3 AST/MP)，但每分钟的篮板总数也与一些中锋持平(大约 0.3 TRB/MP)。同样值得注意的是，KNN 算法将拉塞尔归类为 C 级球员(不正确的分类)，而 K 均值算法将拉塞尔归类为 PG 级球员(正确的分类)。在即将到来的 2019-2020 赛季，看看拉塞尔的助攻天赋和詹姆斯·哈登的投篮/得分天赋之间的协同效应会是多么有趣！**

Github 上的 Python 代码链接:[https://github.com/vincent86-git/NBA_Analysis](https://github.com/vincent86-git/NBA_Analysis)