# 大数据带 R！！

> 原文：<https://medium.com/analytics-vidhya/big-data-with-r-3cea0549cfba?source=collection_archive---------20----------------------->

时不时的，我们总会面对和听到 R 在大数据上的迟缓。这里我们谈论的是 TB 或 Pb，这是 R 的最大限制之一，数据应该适合 RAM。

为了避免这种情况，我们使用了内存不足的处理概念，即分块处理，而不是一次全部处理。我们使用两个不同的包，如下所示。

```
#install.packages("ff")
library(ff)#install.packages("ffbase")
library(ffbase)
```

1.  ff 包基本上是将数据分块，并以编码的原始平面文件的形式存储在硬盘上，并且让您可以更快地访问这些函数。ff 数据帧的数据结构还提供了到 RAM 中分区的数据集的映射。数据块如何工作的示例，假设一个 2GB 的文件，读取文件中的数据大约需要 460 秒，其中有 1 个 515 KB 大小的 ff 数据帧和 28 个 50 MB 的 ff 数据文件，因此为 1.37GB
2.  为了执行基本的合并，寻找重复和丢失的值，创建子集，等等，我们使用 ffbase 包。我们还可以直接用 ff 对象执行聚类、回归和分类。

让我们为上述操作寻找一些 R 代码

```
# Uploading from flatfiles system("mkdir ffdf")
options(fftempdir = "./ffdf")system.time(fli.ff <- read.table.ffdf(file="flights.txt", sep=",", VERBOSE=TRUE, header=TRUE, colClasses=NA))system.time(airln.ff <- read.csv.ffdf(file="airline.csv", 
VERBOSE=TRUE, header=TRUE,colClasses=NA))# Merging the datasetsflights.data.ff = merge.ffdf(fli.ff, airln.ff, by="Airline_id")
```

子集化

```
# Subsetsubset.ffdf(flights.data.ff, CANCELLED == 1, select = c(Flight_date, Airline_id, Ori_city,Ori_state, Dest_city, Dest_state, Cancellation))
```

描述统计学

```
# Descriptive statisticsmean(flights.data.ff$DISTANCE)
quantile(flights.data.ff$DISTANCE)
range(flights.data.ff$DISTANCE)
```

使用 biglm 进行回归(数据集:加州大学欧文分校在[http://archive.ics.uci.edu/ml/index.html](http://archive.ics.uci.edu/ml/index.html)的慢性肾病数据集)

```
# Regression requires installation of biglm packagelibrary(ffbase)
library(biglm)model1 = bigglm.ffdf(class ~ age + bp + bgr + bu + rbcc + wbcc + hemo, data = ckd.ff, family=binomial(link = "logit"), na.action = na.exclude)model1
summary(model1)
#Refining of the model can be done according to the significance level obtained in model1
```

具有 biglm 和 bigmemory 的线性回归

```
# Regression with big memory and biglm packagelibrary(biglm)ckd.mat = read.big.matrix("ckd.csv", header = TRUE, sep = ",", type = "double",backingfile = "ckd.bin", descriptorfile = "ckd.desc")regression  = bigglm.big.matrix(class~ bgr + hemo + age, data = ckd.mat, fc = c("bgr", "hemo"))summary(regression) 
```

此外，当你深入一点时，我们刚刚谈到了存储，但当我们需要处理或分析数据时，我们需要知道并行计算。解释这一点的最简单方式是 youtube 视频，并计算视频中随机颜色出现的时间，因此在这种情况下，并行计算开始发挥作用，mapper 拆分输入，并进一步简化为键值对。

因此，我们使用 H20 作为 r 中并行和大数据的快速和可伸缩平台。

我希望这篇文章对您在 r 中使用大数据有所帮助。感谢您阅读这篇文章。

“数据是新的科学，大数据掌握着答案”——帕特·基尔辛格

![](img/44e4339e79493b8db0ea4074c17c6ba5.png)

informationweek.com[图片作者](https://www.informationweek.com/big-data/big-data-analytics/big-data-maturity-youre-not-as-mature-as-you-think/d/d-id/1331383)