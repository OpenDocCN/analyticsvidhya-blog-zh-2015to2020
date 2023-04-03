# 更新 Ubuntu 中的 R:该做的和不该做的

> 原文：<https://medium.com/analytics-vidhya/updating-r-in-ubuntu-dos-and-don-ts-8e2f45081fb?source=collection_archive---------39----------------------->

![](img/bfc0cfe74b870df620bbc63f5faecab9.png)

照片由[马太·亨利](https://stocksnap.io/author/200)从 [StockSnap](https://stocksnap.io) 拍摄

昨天是我终于鼓起勇气在电脑上更新 R 的日子。很明显，当我使用代号为“某人依靠”的旧 R 版本 3.4.4 时，更新就要到了。我想把它更新到最新版本 4.0.0，“植树节”。就像任何正常人可能做的那样，我在互联网上搜索如何更新 r。令我失望的是，我在各种网站上找到的建议没有一个对我有用。就在那时，我决定发挥一点创造力，走一条自己的路。显然这就是麻烦的开始。

***注意:这不是更新 r 的标准方式，我强烈建议您遵循本*** [***文章***](https://cran.r-project.org/bin/linux/ubuntu/) ***中所述的标准程序。如果这不起作用，欢迎你尝试。***

首先，我在我的系统中添加了适当的公钥。

```
sudo apt-key adv — keyserver keyserver.ubuntu.com — recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
```

然后我指定了应该下载包的远程位置。

```
sudo add-apt-repository 'deb [https://cloud.r-project.org/bin/linux/ubuntu](https://cloud.r-project.org/bin/linux/ubuntu) bionic-cran40/'
```

然后我想更新来源

```
sudo apt update
```

下一步是尝试安装新的 R 版本

```
sudo apt-get install r-base-dev
```

这就是麻烦开始的地方。由于某种原因，安装返回了错误。这些错误根本不是人能看得懂的。在 4、5 次不成功的尝试后，我放弃了安装它。

然后我尝试了一种不同的方法。首先，我卸载了我目前的 R 版本 3.4.4。

```
sudo apt-get remove r-base-core
```

….然后尝试安装新的 R 版本。

```
sudo apt install r-base-core
```

这次成功了，安装了新的 R 版本。我通过在终端中键入以下命令来确保安装了正确的版本。

```
R --version
```

令我感到有趣的是，它返回了以下输出。

```
R version 4.0.0 (2020-04-24) -- "Arbor Day"
Copyright (C) 2020 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)
```

似乎一切都很好，对吗？真的不是。

# **在新版本上重新安装新包**

新版本一安装，我就想在新的更新上安装我以前安装的包。我认为将旧版本的包文件夹中的所有包文件复制到新版本的包文件夹中是可行的。我把旧版本文件夹里的所有东西都复制到了新版本的文件夹里。

```
cp -a /home/batman/R/x86_64-pc-linux-gnu-library/3.4/. /home/batman/R/x86_64-pc-linux-gnu-library/4.0
```

然后我试着从交互式 R 控制台加载一个随机的包。它返回一个错误，要求我重新安装每个软件包。由于手动重装数百个软件包是一项巨大的(如果不是不可能的话)任务，我在谷歌上寻找解决方案。然后我就找到了这个方法，把所有的包都更新了，而不是重装。

```
update.packages(ask = FALSE)
```

虽然，它没有重新安装每一个软件包。我不得不自己手动安装一些软件包。

这就是我第一次更新 R 的经历，我想分享它，以防万一，如果你有任何错误，坚持用标准方式更新，它会帮助你。