# 大数据…但它是什么？

> 原文：<https://medium.com/analytics-vidhya/big-data-but-what-is-it-85e2cc36db89?source=collection_archive---------20----------------------->

![](img/d8d8c2e8c16a175514734aa400b11fcb.png)

要谈论大数据，我们首先必须谈论数据。随着技术的快速进步，数据以各种形状和大小出现。有结构化数据，非结构化数据，轻微结构化数据也称为半结构化数据。由于其行和列结构，结构化数据很容易管理、搜索和过滤，并且需要较少的存储空间。而非结构化数据很复杂，有许多不同形式的数据库文件，难以管理，需要更多存储。

![](img/4bac4abf8193bacd95a56101113478b2.png)

结构化数据与非结构化数据:它们是什么，为什么要关注它们？来自 Lawtomated

但这些数据通常分为两类:传统数据和大数据。

## 传统数据

无论是在数据科学领域还是在数据科学领域之外，大多数人都熟悉传统数据。传统数据是通常以固定格式维护的结构化数据。xlsx，。sql，。csv，。json 等)，易于操作和清理。传统的数据系统可能很大(千兆字节或太字节(= 1000 千兆字节),通常用于处理和了解自己的组织、公司或客户状况。

## 大数据

在过去 10 年中，围绕大数据的讨论越来越多，并慢慢成为主流。[据预测，全球数据空间将超过 175 吉字节](https://www.seagate.com/files/www-content/our-story/trends/files/idc-seagate-dataage-whitepaper.pdf) (1 吉字节= 10 亿字节)。

![](img/4ba78dac38fd5089cd6ceebf5bd48040.png)

资料来源:来自 IDC 全球数据圈的《2025 年数据时代》

作为进入数据科学领域的人，我们也应该知道什么是大数据，以及它与传统数据有何不同。

# 大数据的三个 v

![](img/f672738aa0c43a7d1922a4478ca36839.png)

大数据三巨头

这三个“V”是成交量、速度和多样性。这些“V”被广泛接受来组成大数据，并由道格·兰尼创造。

## 音量:

大数据…很大。范围从 Pb(1 Pb = 1，000，000 GB)到 Zettabytes (1 Zettabyte = 1，000，000 Pb)。但这不足以称之为大数据。与传统数据相比，大数据相对较新，大数据的出现是由于技术进步，这使得数据收集变得如此容易。

## 速度:

数据的速度就是数据的增长。大数据需要高速度。很容易看出脸书每天 4pb 的数据如何变成大数据(4pb x365 天= 1，460 Pb/年)…

如前所述，由于技术进步和证明我们拥有 MRI，大数据正在上升。[现在核磁共振成像的速度快了 7 倍，](https://news.berkeley.edu/2011/01/05/functionalmri/)这导致图像的时间间隔更短，这意味着医生和数据科学家需要分析更多的数据并得出合理的结论。

## 品种:

多样性是指数据库文件的多样性。脸书是大数据公司的典范。你有没有做过“你是哪个(x)人？”或者那些小测验或者那些“性格测试”？你点击并回答，然后提交“结果”，你得到的反馈要么会让你产生某种情绪。但在幕后，你所有的答案都被收集和存储，很可能是在一个表中，一个结构化的数据。这是大数据中非常非常非常小的一部分。脸书用户还公开或私下张贴和发送照片、音频、视频、信息，这些都是脸书需要存储在他们服务器上的数据。[因此，每天接收的 4 Pb 新数据是非结构化或半结构化的。](https://research.fb.com/blog/2014/10/facebook-s-top-open-data-problems/)

![](img/55ad2236d85a0fadbcfe06e5186b4310.png)

GIPHY 的 gif

# 更多的 V。

上面的三个“V”被广泛接受，但仍有人谈论哪些其他类别的数据必须被视为大数据。

[IBM 认为准确性](https://www.ibmbigdatahub.com/infographic/extracting-business-value-4-vs-big-data)，数据的质量/确定性也是一个因素，使其成为四个“V”。如果它是无意义的数据，那么它只是占用了其他有意义数据的空间。但是一个人怎么知道什么数据是无意义的呢？这就是为什么公司更喜欢拥有领域知识的候选人。

[SAS(统计分析系统)研究所认为大数据有五个定义](https://www.sas.com/en_us/insights/big-data/what-is-big-data.html)。SAS 认为准确性和可变性是影响大数据的因素。多样性和可变性是不同的。多样性着眼于不同的数据库文件，而可变性着眼于数据的“流动”。

Lightsondata 的创始人 George Firican 认为有 10 个“V”来描述大数据。你可以在这里阅读关于 10 个“V”的[。](https://tdwi.org/articles/2017/02/08/10-vs-of-big-data.aspx#:~:text=Variability%20in%20big%20data's%20context,of%20inconsistencies%20in%20the%20data.&text=Variability%20can%20also%20refer%20to,is%20loaded%20into%20your%20database.)

然而，这位创造了大数据“三个 V”的人，道格·兰尼在接受来自 KDnuggets 的格雷戈里·皮亚特斯基的采访时说道:

> “是的，其他人提出了其他 V，如准确性，但这些不是衡量大的标准，因此不是大数据的真正定义特征。然而，对于大多数数据来说，它们是重要的考虑因素。事实上，我和一些同事提出了 12 个 Vs 的数据，可用于确保数据的各个方面得到适当的管理和利用。”——道格·兰尼

![](img/2939a35a37a9447b2d6580acc764608f.png)

Gif 来自 GIFSFORUM.com

那么什么是大数据呢？最终，大数据是庞大的、非结构化的，并以疯狂的速度增长。至于其他 V，检查你的行业重点和你信任自己的领域知识。*然而，不管“V”是什么，数据需要添加/带来* ***值*** *到数据收集的大计划中。*价值在于更好地了解客户活动或行为以提高客户满意度/忠诚度，展示减少公司支出、提高效率的方法等。

如果你仍然不确定什么是什么以及 V 的区别，那么看看这篇文章

## 想了解更多关于访问大数据的信息吗？

Hadoop 和 Spark 是众多大数据领先工具中的两个。查看[什么是 Hadoop？](/better-programming/what-is-hadoop-b90591ffae89)作者[欧耶托克·鸢·艾曼纽](https://medium.com/u/7f00eb9ee674?source=post_page-----85e2cc36db89--------------------------------)或[阿帕奇·斯帕克的高层概述](/better-programming/high-level-overview-of-apache-spark-c225a0a162e9)作者[埃里克·吉鲁阿德](https://medium.com/u/824bf28da451?source=post_page-----85e2cc36db89--------------------------------)。

## 关于数据科学的可选趣味读物:

如果你有脸书，想了解更多关于它的统计数据，[点击这里](https://kinsta.com/blog/facebook-statistics/)，如果你想了解更多关于脸书的数据结构(大的和小的)，[点击这里](https://research.fb.com/blog/2014/10/facebook-s-top-open-data-problems/)。

David Reinsel、John Gantz 和 John Rydning 的《世界的数字化》(非常有趣，关于大众的潜在未来)