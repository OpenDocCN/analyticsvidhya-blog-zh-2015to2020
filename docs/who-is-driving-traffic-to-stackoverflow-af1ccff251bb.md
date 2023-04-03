# 谁在推动 StackOverflow 的流量？

> 原文：<https://medium.com/analytics-vidhya/who-is-driving-traffic-to-stackoverflow-af1ccff251bb?source=collection_archive---------16----------------------->

# 2020 年哪些数据库需求量大？什么对开发者选择工作起着很大的作用？

**或 StackOverflow 年开发者调查分析**

![](img/14d6d35dc221d836637164e0d31fabbb.png)

[Denys Nevozhai](https://unsplash.com/@dnevozhai?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

十年来，StackOverflow 一直在进行年度开发者调查。调查结果已经公之于众，并从许多不同的角度进行了分析，以揭示开发人员概况和行业趋势。

作为一名开发人员，我经常使用 StackOverflow，并且总是发现它非常有帮助。作为一个用户，我对 StackOverflow 中的社区合作程度感到惊讶。我有一些具体的问题，我想通过分析截至目前(2020 年 10 月)的最新开发者调查结果来找到答案。这项调查是在 2020 年 2 月进行的，就在全世界进入新冠肺炎隔离区之前。

【2020 年 StackOverflow 开发者调查

StackOverflow 开发者调查 2020 是一项包含近 60 个问题的大型调查。64461 名受访者填写了调查问卷。在没有特定问题的情况下接近一个数据集，你可能会发现自己陷入了一场巨大而混乱的努力中。在阅读了几篇关于前几年 StackOverflow 调查的中型帖子后，我思考了我最感兴趣的是从这个数据集中发现什么。

我是一名数据专业人员，我想知道目前哪些数据库在开发人员中需求量很大。我还想画一个 StackOverflow 频繁用户的侧写。最后，我想知道哪些因素在开发人员的工作决策中起着重要作用。

我将尝试从调查结果中找到以下问题的答案。

我的分析过程包括三个部分:

1.数据探索—我探索数据集及其属性。

2.数据准备——清理、整理和转换数据集。

3.数据分析——我进行分析并找到研究问题的答案。

**2020 年最流行的数据库是什么？**

![](img/e2c1b69f9b5eee0610766995d2cd7299.png)

作者图片

MysQL，PostgreSQL，微软 SQL Server 领先。MySQL 和 PostgreSQL 都是开源的，是维护良好的数据库系统之一。SQLite 是一个轻量级数据库系统，主要用于轻量级 web 应用程序，接近顶级。主要用于大数据的 NoSQL 数据库系统 MongoDB 正在崛起。

![](img/8cf2347857b1fc13f80a0e85010b01ad.png)

作者图片

我还调查了开发者明年想要什么样的数据库。看来 PostgreSQL 和 MongoDB 的流行还会继续。Redis、SQLite 和 Elasticsearch 明年将加入更高的行列。

StackOverflow 频繁用户的特征是什么？

![](img/86b6d2a7a20c6d71effa32ece88c7160.png)

作者图片

开发者以不同的频率访问 StackOverflow。看起来大多数被调查的开发者经常使用这个网站。

![](img/613da895b469d49b5878694b0957ff69.png)

作者图片

调查结果显示男女之间存在巨大的性别差距。这是回答性别问题的人的分布。

![](img/0c3953426ae5f869ed46bdcec5dd5af8.png)

作者图片

我调查了 StackOverflow 访问频率的性别分布。不出所料，男性比女性更多地使用 StackOverflow。

![](img/8f60dfdec6ff4732f2f1cd7f45af73f0.png)

作者图片

为了更好地分析，我放大了每天或几乎每天多次访问 StackOverflow 的受访者。我称他们为 StackOverflow 频繁用户。

看起来有十年编码经验的开发者都是 StackOverflow 明星用户。我们可以得出结论，拥有四至十五年编码经验的开发人员访问站点最多。相反，拥有超过 15 年经验的开发者倾向于较少使用这个网站。

![](img/fdbbfe27db3da6c9827dce4e9ac3ab7c.png)

作者图片

我调查了一般有编码经验的开发人员和专业编码的开发人员在 StackOverflow 的使用上是否有什么不同。事实证明，拥有一到五年专业编码经验的开发人员使用该网站最多。这可能表明，开发人员在开始将编码作为一项工作之前，往往会有几年的编码经验。与上面的图表类似，更多年的专业编码意味着较少使用 StackOverflow。

![](img/bd333caac24fd5c6c8314a6b3902939c.png)

作者图片

StackOverflow 经常使用的用户都比较年轻。25 岁到 32 岁的开发者贡献了网站的大部分流量。

![](img/6bce37b17c1f4dc6517cd1b51d8cb63b.png)

作者图片

StackOverflow 频繁用户的职业是开发人员。这并不奇怪，因为我们已经从之前的图表中发现了这一点。

![](img/d26b0a16342ef76824fe70011cfdb7ae.png)

作者图片

StackOverflow 的经常用户来自美国、印度、德国和英国。在这种趋势中找不到中国是一种有趣的洞察力。

![](img/f250245524537f843037e02076850a0a.png)

作者图片

StackOverflow 的频繁用户是他们开发人员职业之上的业余编码爱好者。

![](img/94af9b59a34af312d00999b14b9a11f5.png)

作者图片

大多数 StackOverflow 的常客不是在找工作，而是对新的机会持开放态度。

**2020 年，对开发者来说，最重要的工作因素是什么？**

![](img/614e5270903298efc2d8ca4dbe3d9ff1.png)

作者图片

大多数开发人员基于技术语言、框架和其他技术做出关于工作的决策。这并不奇怪，因为许多其他专业人士可能也会这样做。大多数开发人员认为办公环境和公司文化、弹性时间或灵活的日程安排以及发展机会在工作考虑中更重要。当开发人员考虑工作时，远程工作选项是一个上升的因素。

**结论**

![](img/f22ce94a758092ec5d66b30b71c8b8ea.png)

托拜厄斯·菲舍尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

MySQL、PostgreSQL 和 MS SQL Server 是最流行的数据库。MongoDB 和 Redis 在未来一年的需求会更高。

![](img/50db6366a53b7bd9edf64146b111ef54.png)

StackOverflow 频繁用户是一名年轻男性，其职业是开发人员，具有大约五年的专业编码经验，也喜欢将编码作为一种爱好，并且通常已经编码大约十年，来自北美、印度或欧洲，有一份稳定的工作，但对新的机会持开放态度。

![](img/12da1a2cbd9fde38fd12c95f9b69f48a.png)

照片由[珍·西奥多](https://unsplash.com/@jentheodore?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

当考虑一项工作时，开发人员关注他们将要使用的技术语言、框架和技术。他们还看重办公环境、公司文化、弹性时间、弹性日程、职业发展和远程工作选择。

对调查结果的数据分析很有教育意义，因为我必须找到清理和整理数据的方法来进行分析。发现有趣的见解也令人兴奋。在即将到来的调查中，我有兴趣比较后大流行时代开发人员的职业和生活选择发生了什么变化。我的代码可以在 [GitHub](https://github.com/DiloromA/StackOverflow-Developer-Survey-2020-Analysis) 上找到。