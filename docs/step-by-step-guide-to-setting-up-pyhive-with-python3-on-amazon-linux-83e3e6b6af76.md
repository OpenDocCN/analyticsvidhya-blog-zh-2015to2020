# 在 Amazon Linux 上用 python3 设置 PyHive 的分步指南

> 原文：<https://medium.com/analytics-vidhya/step-by-step-guide-to-setting-up-pyhive-with-python3-on-amazon-linux-83e3e6b6af76?source=collection_archive---------5----------------------->

![](img/00e64f1af5e260c8a7d1d00996af06e9.png)

格伦·卡斯滕斯-彼得斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

一个不眠之夜和数百次谷歌搜索之后，我想出了如何设置我的新 ec2 实例来成功连接 hive。

我正在处理 AWS 数据管道，并使用运行 Amazon Linux 的 AWS 提供的现有映像创建一个新的 ec2 实例。(亚马逊 Linux AMI 2018 . 03 . 0 . 2018 08 11 x86 _ 64 HVM EBS)

[亚马逊 Linux](http://aws.amazon.com/amazon-linux-ami/) 是从[红帽企业 Linux (RHEL)](https://en.wikipedia.org/wiki/Red_Hat_Enterprise_Linux) 和 [CentOS](https://en.wikipedia.org/wiki/CentOS) 演化而来的发行版。它不支持`apt-get`，但有`yum`可满足所有安装要求。

PyHive 是针对 [Presto](http://prestodb.io/) 和 [Hive](http://hive.apache.org/) 的 Python [DB-API](http://www.python.org/dev/peps/pep-0249/) 和 [SQLAlchemy](http://www.sqlalchemy.org/) 接口的集合。它有助于从 Python 查询您的数据库。

众所周知，让`PyHive`和`python3`一起工作是非常困难的。我无法在一个地方找到所有的步骤，因此决定写这篇博客，这样就没有人会因为如此琐碎的事情而度过一个不眠之夜。

按顺序运行下面的命令似乎效果很好。

# 说明

`**Line 4**`
因为这是一个新的 EC2 实例，所以建议从我们正在使用的 AMI 镜像中更新所有可用的包。

`**Line 5**`
在实例上安装 python3.6

`**Line 7 and Line 8**`
安装`gcc`和`gcc-c++`，它们为构建我们使用`c`和`c++`构建的 python 包提供了基础设施。

`**Line 10**`
安装 python36-dev 包。这不是开箱即用的，需要单独安装。

如果您不安装这些，当您试图在`line 17`中安装`sasl`时，您可能会得到以下错误跟踪

安装`python36-devel.x86_64`包会带来许多头文件，包括缺失的`pyconfig.h`文件，这是构建 python 扩展所必需的。此实例上可用于 yum 安装的`python-dev`包是`python36-devel.x86_64`。

如果需要，您可以通过以下方式找到适用于您系统的`python-dev`软件包:

`> yum search python3 | grep devel`

我的系统上的输出是:

您可以使用`Line 10`中的命令安装相关的开发包。

`**Line 11**`
安装`sasl-dev`包。这个包专门带来了顺利安装`sasl`包所需的所有头文件。如果没有安装，您将得到以下错误跟踪

使用下面的命令快速搜索，我们可以找到要安装的相关`sasl-dev`包

`> yum search sasl | grep devel`

输出是需要安装的`cyrus-sasl-devel.x86_64: Files needed for developing applications with Cyrus`。我们可以通过运行`Line 11`中的命令来安装它。

`**Line 13**`
Boto 是针对 Python 的亚马逊 Web 服务(AWS) SDK。它使 Python 开发人员能够创建、配置和管理 AWS 服务，如 EC2 和 S3。因此我们安装它。

`**Line 15 to Line 18**`
我们安装 pyhive 顺利运行所需的所有其他相关包。

现在，您可以在 yout python 脚本中非常轻松地连接到您的 hive 服务器。

就是这样！:D

我希望这能帮助你，并节省你的谷歌时间。

请不要犹豫在评论部分改正任何错误。我真的很想学习和提高。