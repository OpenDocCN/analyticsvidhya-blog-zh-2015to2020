# 使用 ROS 开始学习机器人技术

> 原文：<https://medium.com/analytics-vidhya/getting-started-with-robotics-fb1ebe5bc630?source=collection_archive---------16----------------------->

![](img/b9ee5693a979d97a195e61d1c4a7c569.png)

如果你对机器人感兴趣，不知道如何开始，那么这个教程是给你的。

**什么是 ROS？**

机器人操作系统(ROS)是一个中间件或元操作系统(我不知道确切的操作系统),由斯坦福大学的博士们创建，他们发现需要一个平台来自动化机器人技术中涉及的大多数单调乏味的任务。这个团体叫做柳树车库，它仍然存在。中间件不是一个操作系统，而是运行在一个操作系统上，只要这个系统有 ROS 运行，它就可以进行任何平台无关的开发。这意味着它可以在任何设备上运行，不需要修改任何代码。它使系统集成变得非常容易。Ros 是开源的，所以已经对社区做出了很多贡献，所以你不必每次都要从头开始测试你的机器人。现在足够的背景知识。我们来装 ROS 吧？

请注意，要安装的 ROS 版本(通常称为 ROS 发行版)取决于您系统中使用的 Ubuntu 版本。下面是与一起使用的相应的 ROS 发行版

ROS Indigo — Ubuntu 14.04

ROS Kinetic(带 Gazebo 7) — Ubuntu 16.04

ROS Melodic(带露台 9) — Ubuntu 18.04 LTS(推荐)

ROS Noetic-Ubuntu 20.04 LTS

LTS =长期支持

Gazebo 是 ROS 的模拟环境。

建议您不要为初学者使用最新版本，因为它正处于开发阶段，您可能会面临一些新的错误。也不建议使用过时版本的 ROS，因为可能无法获得对该版本的支持。

在这篇文章中，我将向你展示如何安装 ROS Kinetic 和 Melodic，所以我将在这里介绍 Ubuntu 16.04 和 18.04 用户。

另外，请注意，如果您的系统中没有安装 Ubuntu，那么您有几个选择

1.切换到 Ubuntu OS 作为主要和唯一的操作系统。这可能有点难，但当你掌握了窍门后，你会爱上它的。

2.使用 Ubuntu 操作系统双重启动——当然，对于不想放弃 windows 的用户或游戏玩家来说，这可能是一个不错的选择。虽然，它需要一些核心的硬件规格来实现它。参见[这篇文章](https://www.geeksforgeeks.org/creating-a-dual-boot-system-with-linux-and-windows/)

3.在 windows 操作系统中安装虚拟机，这也是所有选项中的一个方便选项。参见[这篇文章](https://www.geeksforgeeks.org/how-to-install-ubuntu-on-virtualbox/)

**设置来源列表**

> ' sudo sh-c ' echo " deb[http://packages.ros.org/ros/ubuntu](http://packages.ros.org/ros/ubuntu)$(LSB _ release-sc)main ">/etc/apt/sources . list . d/ROS-latest . list '

**添加键**

> sudo apt-key adv—key server ' hkp://key server . Ubuntu . com:80 '—recv-key C1 cf 6 e 31 e 6 bade 8868 b 172 B4 f 42 ed 6 fbab 17 c 654

如果以上未导入，请参考 [ROS 官网](http://wiki.ros.org/kinetic/Installation/Ubuntu)获取新密钥。

**更新套餐列表**

> sudo apt-get 更新

**安装 ROS Kinetic 全桌面版**

> sudo apt-get 安装 ROS-kinetic-desktop-完整

运筹学

**安装 ROS Melodic 全桌面版**

> 安装 ros-melodic-desktop-full

**初始化 Ros 依赖关系**

Ros dep 使得为你要编译的源代码安装所有的系统依赖项变得非常容易，并且对于在 Ros 中运行一些核心组件也是必不可少的。所以让我们安装 rosdep。

> sudo rosdep 初始化

如果上面的命令不起作用，那么请试试这个

> sudo apt-get 安装 python-pip
> 
> sudo pip install -U rosdep
> 
> sudo rosdep 初始化
> 
> rosdep 更新

**设置 ROS 环境**

让 bashrc 文件知道 Ros 安装的位置

对于 Ubuntu 16.04

> printf " source/opt/ROS/kinetic/setup . bash " > > ~/。bashrc

对于 Ubuntu 18.04

> printf " source/opt/ROS/melodic/setup . bash " > > ~/。bashrc

对于两个版本的源 bashrc，应用在文件中完成的上述修改。

> 来源~/。bashrc

**为 ROS 安装 Python 包**

ROS 可以在 C++和 python 上工作。这使得新手加入社区变得非常容易。在这个系列教程中，我们将使用 python，它对初学者很友好，所以让我们看看如何为 ROS 安装 python 包和 python 柳絮工具。柳絮工具使我们能够使我们的柳絮工作空间，这将包含您的 ROS 项目的所有文件和文件夹。这使得管理需求不同的各种其他项目变得更加容易，也大大减少了版本冲突。

> sudo apt-get 安装 python-rosinstall
> 
> sudo 安装 python-柳絮-工具

**其他重要的 ROS 包**

为 ROS 转换包，这有助于我们在模拟中了解机器人不同部分的坐标。允许用户在任何期望的时间点在任何两个坐标框架之间变换点、向量等，并且还可以随时跟踪多个坐标框架。除此之外，我们需要安装一些更重要的包，这些包在未来的开发中肯定会用到。

对于 16.04

> sudo apt-get 安装 ros-kinetic-tf-*
> 
> sudo apt-get 安装 ROS-kinetic-PCL-msgs ROS-kinetic-MAV-msgs ROS-kinetic-MAV ROS-kinetic-octo map-* ROS-kinetic-geographic-msgs lib geographic-dev

对于 18.04

> sudo apt-get 安装 ros-melodic-tf-*
> 
> sudo apt-get 安装 ROS-melodic-PCL-msgs ROS-melodic-MAV-msgs ROS-melodic-MAV ROS-melodic-octo map-* ROS-melodic-geographic-msgs lib geographic-dev

**安装后开始使用**

**创建柳絮工作空间**

一个柔荑工作空间把你的包含文件、文件夹和依赖关系的 ROS 项目与其他对 ROS 不重要的文件夹分开

在终端中你想要保存所有 ROS 项目的文件夹中，执行以下命令。

创建柳絮工作空间

> mkdir 柳絮 _ws

进入工作空间

> cd 柳絮 _ws

正在创建源目录。-p 表示这是一个包。

> mkdir -p src

进入源文件夹

> cd src

初始化柳絮工作空间

> 柳絮 _ 初始化 _ 工作空间

打印的位置。bashrc 文件中的柳絮工作空间的 bash 文件。

> printf " source ~/柳絮 _ws/devel/setup.bash" >> ~/。bashrc
> 
> CD ~/柳絮 _ws

在我们的柳絮工作空间中构建包

> 柳絮 _ 制作

这使得 2 个新的文件夹建立和 devel 文件夹。

build —调用 CMake 和 Make 的文件夹

devel —包含在目标位置生成的所有文件。

现在到了乏味的部分。对于每个终端，您需要获取 setup.bash 文件，但您不必担心，因为我已经在 bashrc 文件中包含了 source 命令，所以每次调用它时，setup.bash 文件都会自动执行。我从一开始就罩着你。

> source ~/柳絮 _ws/devel/setup.bash

**检查 ROS**

在 Ubuntu 16.04 上，你应该有 ROS Kinetic，而在 Ubuntu 18.04 上，你应该有 ROS Melodic

现在，我们来检查一下 ROS 的版本。

> 罗斯版本-d

可以去[官方 Ros 文档](http://wiki.ros.org/Documentation)进一步研究。现在你的系统已经设置好了，你已经跨越了最初的障碍，这是任何旅程中最困难的部分。你可以看到你在这个领域的学习曲线呈指数增长，所以恭喜你！。如果你对此有任何疑问，请在下面的评论区提出。我将在这个学习机器人的系列中添加更多的教程。如果你想继续关注，你可以**star this Github**[**repository**](https://github.com/amancodeblast/RoboticsD)获取更多与**计算机视觉、SLAM、3 D 感知、导航等相关的机器人技术内容**。我会不断更新它以获取更多的资源，如果你想得到通知，记得启动资源库并在 GitHub 上关注我。