# 在 Ubuntu 上安装 Scala/SBT:-第 3 部分

> 原文：<https://medium.com/analytics-vidhya/installing-scala-sbt-on-ubuntu-part-3-fd08fec70946?source=collection_archive---------1----------------------->

***日积月累:-***

Scala 是一种受 Java 影响的通用、面向对象的编程语言。事实上，Scala 项目可以像 Java 项目一样编码。由于 Scala 在数据处理和分布式编程方面有很高的需求，所以它是一种您可能希望放在工具箱中的语言。但是如果你的开发平台是 Linux。然后你需要在最新的 Ubuntu 机器上安装 Scala，最新的是 Ubuntu 18.4，

在 Ubuntu 上安装 Scala 之前，你需要确认你的 Ubuntu 机器上是否安装了 Java。如果没有，那么你需要先安装 java。

安装 Java 最简单的选择是使用 Ubuntu 打包的版本。具体来说，这将安装 OpenJDK 8，这是最新的推荐版本。

**第一步:安装 Java。**

首先，通过在终端中键入以下命令来更新包索引:

> *sudo apt-get 更新*

输入你的密码后，它会更新一些东西。
现在，您可以使用以下命令安装 JDK:

> sudo apt-get 安装 openjdk-8-jdk

在你的系统中安装了 java 之后。你可以用下面的命令检查版本。

> javac 版本

**第二步:在 Ubuntu 上安装 Scala**

接下来我们将安装 Scala。为此，下载必要的。deb 文件，命令为:

> sudo wget[https://downloads . light bend . com/Scala/2 . 13 . 0/Scala-2 . 13 . 0 . deb](https://downloads.lightbend.com/scala/2.13.0/scala-2.13.0.deb)

**注:-**

查看 Scala 档案页面，确保你下载的是最新版本的 Scala。用斯卡拉。下载 deb 文件，用命令安装它:

> sudo dpkg -i scala-2.11.12

一旦在你的系统中安装了 Scala，你可以用下面的命令检查 Scala 的版本:

> scala 版本

要编写 Scala 代码，你只需输入

终端中的“scala”并用你的第一个代码**“Hello Word”**进行检查

> Println("Hello Word ")

**第三步:从 Ubuntu 卸载 Scala:-**

如果你想从系统中卸载 Scala，你可以运行下面的命令。

> sudo apt-get 删除 scala-2.11.12.deb

**第四步:在 Ubuntu 上安装 SBT**

现在是时候安装 sbt 了。首先使用以下命令添加必要的存储库:

> sudo apt-key adv—key server " deb[https://dl.bintray.com/sbt/debian](https://dl.bintray.com/sbt/debian)/" | sudo tee-a/etc/apt/sources . list . d/SBT . list

使用以下命令为安装添加公钥:

> sudo apt-key adv—key server hkp://key server . Ubuntu . com/pks/lookup？op = get & search = 0 x2 ee 0 ea 64 e 40 a 89 b 84 B2 df 73499 e82a 75642 AC 823

使用以下命令更新 apt:

> sudo apt-get 更新

最后，使用以下命令安装 sbt:

> sudo apt-get 安装软件

安装完成后，您需要使用下面的命令创建一个目录，在其中构建您的项目。

mkdir **sbt_project**

cd **sbt_project**

一旦到达位置**“sbt _ project”**，键入 SBT 并按回车键

***SBT*** *现在已经准备好了，等待你来构建你的第一个项目…*