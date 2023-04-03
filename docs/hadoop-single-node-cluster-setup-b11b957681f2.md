# Hadoop 单节点集群设置|伪分布式模式

> 原文：<https://medium.com/analytics-vidhya/hadoop-single-node-cluster-setup-b11b957681f2?source=collection_archive---------1----------------------->

![](img/fe22dff9e293eced11d1b3b433a2e608.png)

> Hadoop 可以以 3 种不同的模式安装:独立模式、伪分布式模式和完全分布式模式。
> 
> **独立模式**是 Hadoop 运行的默认模式。独立模式主要用于在你不真正使用 [HDFS](http://hdfstutorial.com/) 的地方进行调试。
> **伪分布式模式**也被称为**单节点集群**，其中 NameNode 和 DataNode 将驻留在同一台机器上。
> **全分布式模式**是 Hadoop 的**生产模式，其中将运行多个节点。**

> 这个草稿可以帮助你创建自己的**定制 hadoop 伪模式集群。** *本设置使用的环境为* ***ubuntu 18.04，hadoop 版本为 3.1.2*** *。*

# 先决条件

## 创建新用户[可选]

> **注意:**如果您想为新用户安装 hadoop，请遵循此步骤，否则跳到下一部分。

打开一个新的终端 *(Ctrl+Alt+T)* &键入以下命令。

首先，创建一个组 *hadoop*

```
*sudo addgroup hadoop*
```

并在同一个 *hadoop* 组内添加一个新用户 *hdfsuser* 。

```
*sudo adduser --ingroup hadoop hdfsuser*
```

> **注意:**如果您想将现有用户添加到组中，则使用以下命令

```
*usermod -a -G hadoop username*
```

现在给 *hdfsuser* 必要的根权限来安装文件。根用户权限可以通过更新 *sudoers* 文件来提供。

在终端中运行以下命令，打开 *sudoers* 文件

```
*sudo visudo*
```

添加或编辑以下行

```
*hdfsuser ALL=(ALL:ALL) ALL*
```

现在，保存更改 *(Ctrl+O &按 enter)* 并关闭编辑器 *(Ctrl+X)。*

因此，现在让我们切换到我们新创建的用户进行进一步的安装。

```
*su - hdfsuser*
```

## Java 安装

Hadoop 是使用 Java 构建的，运行 MapReduce 代码需要 Java。对于最新的 hadoop 安装，Java 版本应为 Java 8 或更高版本，即 Java 1.8+。如果您的系统中已经运行了 java，那么通过在终端中运行以下命令来检查您是否拥有所需的版本

```
*java -version*
```

如果您有所需的版本，请跳到下一步。

> **注意:**如果你也计划安装 hive，那么最好选择 java 8，因为新版本不再有 URLClassLoader。

您可以从您的 OS 软件包管理器或 oracle 官方网站*(*[https://www . Oracle . com/java/technologies/javase/javase-JDK 8-downloads . html](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html)*)安装 Java。*

**安装使用资质 *(java 8)***

```
*sudo apt-get update* *sudo apt install openjdk-8-jre-headless* *openjdk-8-jdk*
```

要验证您的安装，请在终端中运行以下命令

```
*java -version*
```

## 设置 SSH 密钥

Hadoop 核心使用 Shell (SSH)在从属节点上启动服务器进程。它要求主机和所有从机以及从机之间的无密码 SSH 连接。

我们需要一个无密码的 SSH，因为当集群运行时，通信过于频繁。作业跟踪器应该能够快速发送任务到任务跟踪器。

> **注意:**不要跳过这一步，除非你已经有了一个无密码的 SSH 设置。这一步对于启动 hadoop 服务是必不可少的，比如资源管理器&节点管理器。

**安装所需的软件包**

在终端中运行以下命令

```
*sudo apt-get install ssh
sudo apt-get install sshd*
```

**生成密钥……**

```
*ssh-keygen -t rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod og-wx ~/.ssh/authorized_keys*
```

现在，我们已经成功地生成了 ssh 密钥，并将密钥值复制到 authorized_keys。

通过在终端中运行以下命令来验证安全连接。

```
*ssh localhost*
```

> **注意:**如果它不要求输入密码并让您登录，则配置成功，否则移除生成的密钥并再次执行步骤。
> 
> 不要忘记从本地主机退出(在终端中键入 exit 并按 enter 键)

***现在我们的 hadoop 安装先决条件已经成功完成。***

# Hadoop 3.x 安装

现在。让我们从*(*[*【https://hadoop.apache.org/releases.html*](https://hadoop.apache.org/releases.html)*)，*下载最新的稳定版本，开始 hadoop 的安装过程，对于旧版本请访问*(*[*【https://archive.apache.org/dist/hadoop/common/*](https://archive.apache.org/dist/hadoop/common/)*)。*

要下载您选择的版本，请使用以下命令。*(根据个人喜好更改目录和下载链接)。*

```
*cd /usr/local
sudo wget* [*http://archive.apache.org/dist/hadoop/common/hadoop-3.1.2/hadoop-3.1.2.tar.gz*](http://archive.apache.org/dist/hadoop/common/hadoop-3.1.2/hadoop-3.1.2.tar.gz)
```

在相同的位置提取 hadoop 文件。

```
*sudo tar xvzf hadoop-3.1.2.tar.gz*
```

重命名提取的文件夹

```
*sudo mv hadoop-3.1.2 hadoop*
```

## 在伪分布式模式下设置 Hadoop

现在，让我们将 hadoop 的所有权提供给我们的 *hdfsuser【如果不想更改所有权，请跳过】*

```
*sudo chown -R hdfsuser:hadoop /usr/local/hadoop*
```

将 *hadoop* 文件夹的模式改为读取、写入&执行的工作模式。

```
*sudo chmod -R 777 /usr/local/hadoop*
```

## **禁用 IPv6**

IPv6 网络目前不支持 Apache Hadoop。它只在 IPv4 堆栈上测试和开发过。Hadoop 需要 IPv4 才能工作，只有 IPv4 客户端才能与集群对话。

看一看 [HADOOP-3437](https://issues.apache.org/jira/browse/HADOOP-3437) 和 [HADOOP-6056](https://issues.apache.org/jira/browse/HADOOP-6056) 来理解为什么必须禁用 IPv6 才能让 HADOOP 工作。

您可以通过在终端中运行以下命令来检查 IPv6 配置的状态

```
*cat /proc/sys/net/ipv6/conf/all/disable_ipv6*
```

如果结果不是 1，则按照以下步骤禁用 IPv6

```
*sudo nano /etc/sysctl.conf*
```

现在将以下几行添加到文件的末尾

```
*# Disable ipv6
net.ipv6.conf.all.disable_ipv6=1
net.ipv6.conf.default_ipv6=1
net.ipv6.conf.lo.disable_ipv6=1*
```

保存文件并退出

> 如果 ipv6 仍然没有被禁用，那么问题将是/etc/sysctl.conf 没有被激活。要解决这个问题，可以运行以下命令来激活 conf

```
*sudo sysctl -p*
```

## 添加 Hadoop 环境变量

将 hadoop 路径添加到环境中是必要的，否则您将不得不移动到 hadoop 目录来运行命令。

运行以下命令打开 *bashrc* 文件

```
*sudo nano ~/.bashrc*
```

将以下几行添加到 *bashrc* 文件的末尾

```
# HADOOP ENVIRONMENTexport HADOOP_HOME=/usr/local/hadoop
export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop
export HADOOP_MAPRED_HOME=/usr/local/hadoop
export HADOOP_COMMON_HOME=/usr/local/hadoop
export HADOOP_HDFS_HOME=/usr/local/hadoop
export YARN_HOME=/usr/local/hadoop
export PATH=$PATH:/usr/local/hadoop/bin
export PATH=$PATH:/usr/local/hadoop/sbin**#** HADOOP NATIVE PATH
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS=-Djava.library.path=$HADOOP_PREFIX/lib
```

现在，通过运行以下命令加载 hadoop 环境变量

```
*source ~/.bashrc*
```

## 正在配置 Hadoop …

将工作目录更改为 hadoop 配置位置

```
*cd /usr/local/hadoop/etc/hadoop/*
```

*   **hadoop-env.sh** 通过运行以下命令打开 hadoop-env 文件

```
*sudo nano hadoop-env.sh*
```

将以下配置添加到文件*的末尾(根据您的设置更改 java 路径和用户名)*

```
*export HADOOP_OPTS=-Djava.net.preferIPv4Stack=true
export JAVA_HOME=/usr
export HADOOP_HOME_WARN_SUPPRESS="TRUE"
export HADOOP_ROOT_LOGGER="WARN,DRFA"* export HDFS_NAMENODE_USER="hdfsuser"
export HDFS_DATANODE_USER="hdfsuser"
export HDFS_SECONDARYNAMENODE_USER="hdfsuser"
export YARN_RESOURCEMANAGER_USER="hdfsuser"
export YARN_NODEMANAGER_USER="hdfsuser"
```

*   **yarn-site.xml** 通过运行以下命令打开 yarn-site 文件

```
*sudo nano yarn-site.xml*
```

在配置标记( <configuration></configuration> )之间添加以下配置

```
*<property>
<name>yarn.nodemanager.aux-services</name>
<value>mapreduce_shuffle</value>
</property>
<property>
<name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
<value>org.apache.hadoop.mapred.ShuffleHandler</value>
</property>*
```

*   **hdfs-site.xml** 通过运行以下命令打开 hdfs-site 文件

```
*sudo nano hdfs-site.xml*
```

在配置标记( <configuration></configuration> )之间添加以下配置

```
*<property>
<name>dfs.replication</name>
<value>1</value>
</property>
<property>
<name>dfs.namenode.name.dir</name>
<value>/usr/local/hadoop/yarn_data/hdfs/namenode</value>
</property>
<property>
<name>dfs.datanode.data.dir</name>
<value>/usr/local/hadoop/yarn_data/hdfs/datanode</value>
</property>* <property>
<name>dfs.namenode.http-address</name>
<value>localhost:50070</value>
</property>
```

*   **core-site.xml** 

```
*sudo nano core-site.xml*
```

在配置标记( <configuration></configuration> )之间添加以下配置

```
*<property>
<name>hadoop.tmp.dir</name>
<value>/bigdata/hadoop/tmp</value>
</property>
<property>
<name>fs.default.name</name>
<value>hdfs://localhost:9000</value>
</property>*
```

*   **mapred-site.xml** 运行以下命令打开核心站点文件

```
*sudo nano mapred-site.xml*
```

在配置标记( <configuration></configuration> )之间添加以下配置

```
*<property>
<name>mapred.framework.name</name>
<value>yarn</value>
</property>
<property>
<name>mapreduce.jobhistory.address</name>
<value>localhost:10020</value>
</property>*
```

***全部完成…现在让我们为 hadoop 创建一些目录来保存数据***

## 正在创建目录…

让我们为 dfs 创建一个临时目录，如 core-site.xml 文件中提到的那样

```
*sudo mkdir -p /bigdata/hadoop/tmp
sudo chown -R hdfsuser:hadoop /bigdata/hadoop/tmp
sudo chmod -R 777 /bigdata/hadoop/tmp*
```

> **注意:**如果不想改变所有权，跳过 chown 命令。在接下来的步骤中也要记住这一点。

正如我们在 yarn-site.xml 文件中提到的，现在创建保存数据的目录

```
*sudo mkdir -p /usr/local/hadoop/yarn_data/hdfs/namenode
sudo mkdir -p /usr/local/hadoop/yarn_data/hdfs/datanode
sudo chmod -R 777 /usr/local/hadoop/yarn_data/hdfs/namenode
sudo chmod -R 777 /usr/local/hadoop/yarn_data/hdfs/datanode
sudo chown -R hdfsuser:hadoop /usr/local/hadoop/yarn_data/hdfs/namenode
sudo chown -R hdfsuser:hadoop /usr/local/hadoop/yarn_data/hdfs/datanode*
```

好的，目前为止还不错。我们已经完成了所有必要的配置，现在让我们启动资源管理器和节点管理器。

## 正在完成…

在启动 hadoop 核心服务之前，我们需要通过格式化 namenode 来清理集群。每当您更改 namenode 或 datanode 配置时，不要忘记这样做。

```
*hdfs namenode -format*
```

现在，我们可以通过运行以下命令来启动所有 hadoop 服务

```
*start-dfs.sh
start-yarn.sh*
```

> **注意:**你也可以使用 **start-all.sh** 来启动所有的服务

**临时演员……**

> 您可以通过导航到以下 url 来检查您的 namenode 是否已启动并正在运行。

```
*http://localhost:50070*
```

> 要访问 ResourceManager，请导航到 ResourceManager web UI，网址为

```
*http://localhost:8088*
```

> 要检查 HDFS 是否正在运行，您可以使用 Java 进程状态工具。

```
jps
```

这给出了 java 中正在运行的进程的列表。

如果安装成功，您应该会看到列出了这些服务，

```
ResourceManager
DataNode
SecondaryNameNode
NodeManager
NameNode
```

**注意:**如果您发现服务中没有列出 namenode，那么请确保您已经格式化了 namenode。
如果 datanode 丢失，则可能是由于用户对 datanode 目录没有足够的权限，因此请将目录更改为用户具有读&写权限的位置，或者使用前面描述的 chown 方法。

> 要停止所有 hadoop 核心服务，请尝试以下任一方法

```
*stop-dfs.sh
stop-yarn.sh*
```

运筹学

```
*stop-all.sh*
```