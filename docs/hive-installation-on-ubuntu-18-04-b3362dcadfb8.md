# ubuntu 18.04 | MySQL Metastore 上的配置单元安装

> 原文：<https://medium.com/analytics-vidhya/hive-installation-on-ubuntu-18-04-b3362dcadfb8?source=collection_archive---------0----------------------->

![](img/3fd260db2770bd18e986bfee77d1fdab.png)

> Apache Hive 是一个数据仓库基础设施，它有助于查询和管理驻留在分布式存储系统中的大型数据集。它是基于 Hadoop 开发的。Hive 有自己的类似 SQL 的查询语言，叫做**Hive QL**(Hive Query Language)。
> 
> Hive 查询语言类似于 SQL，它支持子查询。使用 Hive 查询语言，可以跨 Hive 表进行 MapReduce 连接。

> 既然，Hive 是建立在 Hadoop 之上的， **Java** 和 **Hadoop** 需要安装在你的系统中。如果您的系统上没有配置 Hadoop，可以按照[https://medium . com/@ adarshms 1997/Hadoop-single-node-cluster-setup-b11b 957681 f 2？sk = 6 BC D2 f 534d 70 f 8d 7 c 34 e 86241 f 43 f 16 e](/@adarshms1997/hadoop-single-node-cluster-setup-b11b957681f2?sk=6bcd2f534d70f8d7c34e86241f43f16e)
> 
> 在安装配置单元之前，请确保您的 Hadoop 安装正常，并且 Hadoop 的所有核心服务都已启动并运行。
> *本次设置使用的环境为* ***ubuntu 18.04，hive 版本为 3.1.2*** *。*

***注意:*** *更喜欢 java 8，因为较新版本不再有运行 hive 所需的 URLClassLoader。*

# Hive 3.x 安装

现在。让我们从*(*[【http://apachemirror.wuchna.com/hive/hive-3.1.2/】](http://apachemirror.wuchna.com/hive/hive-3.1.2/)*)，*下载最新的 stables 版本，开始 hive 的安装过程，旧版本请访问*(**)。*

要下载您选择的版本，请使用以下命令。*(根据个人喜好更改目录和下载链接)。*

```
*cd /usr/local
sudo wget* [http://apachemirror.wuchna.com/hive/hive-3.1.2/apache-hive-3.1.2-bin.tar.gz](http://apachemirror.wuchna.com/hive/hive-3.1.2/apache-hive-3.1.2-bin.tar.gz)
```

提取相同位置的配置单元文件。

```
*sudo tar xvzf* apache-hive-3.1.2-bin.tar.gz
```

重命名提取的文件夹

```
*sudo mv* apache-hive-3.1.2-bin *hive*
```

## 添加配置单元环境变量

将配置单元路径添加到环境中是必要的，否则您将不得不移动到配置单元目录来运行命令。

通过运行打开 *bashrc* 文件

```
*sudo nano ~/.bashrc*
```

将以下几行添加到 *bashrc* 文件的末尾

```
# Set HIVE_HOME 
export HIVE_HOME=/usr/local/hive 
export PATH=$PATH:$HIVE_HOME/bin
```

现在，通过运行以下命令加载配置单元环境变量

```
*source ~/.bashrc*
```

**创建目录…** 现在我们需要在 **HDFS** 中创建 **Hive** 目录。

```
*hdfs dfs -mkdir /bigdata/tmp*
```

现在，为了让 hive 保存表或其他杂项数据，我们需要创建另一个目录。

```
*hdfs dfs -mkdir -p /bigdata/hive/warehouse*
```

**添加权限…**

```
*hdfs dfs -chmod g+w /bigdata/tmp
hdfs dfs -chmod g+w /bigdata/hive/warehouse*
```

## 正在配置配置单元…

将工作目录更改为配置单元配置位置

```
*cd /usr/local/hive/conf*
```

*   **hive-env.sh** 通过运行以下命令打开 hive-env 文件

```
*sudo nano hive-env.sh*
```

将以下配置添加到文件的末尾*(根据您的设置更改路径)*

```
# Set HADOOP_HOME to point to a specific hadoop install directory
HADOOP_HOME=/usr/local/hadoop# Hive Configuration Directory can be controlled by: 
export HIVE_CONF_DIR=/usr/local/hive/conf# Java Home 
export JAVA_HOME=/usr
```

## Metastore 配置

Metastore 是 Apache Hive 元数据的中央存储库。它在关系数据库中存储 Hive 表的元数据(如它们的模式和位置)和*分区* 。

所有 Hive 实现都需要一个 metastore 服务，它在其中存储元数据。默认情况下，Hive 使用内置的 Derby SQL 服务器。也可以选择 ***MySQL、Postgres、Oracle、MS SQL Server*** 作为 Hive Metastore。

对于这种配置，我们将使用 MySQL。Metastore 配置需要在 *hive-site.xml* 文件中指定。

> 首先，让我们使用 aptitude 安装最新的 mysql 版本。如果系统安装了 mysql，可以跳过这一步。

```
*sudo apt-get update
sudo apt-get install mysql-server*
```

> 如果安全安装实用程序在安装完成后没有自动启动，请输入以下命令:

```
*sudo mysql_secure_installation utility*
```

> 该实用程序提示您定义 mysql root 密码和其他与安全相关的选项，包括删除对 root 用户的远程访问和设置 root 密码。

```
*sudo systemctl start mysql*
```

> 这个命令启动 mysql 服务

```
*sudo systemctl enable mysql*
```

> 该命令确保数据库服务器在重新启动后启动

现在成功安装 mysql 服务器后，我们需要安装 ***mysql java 连接器。*** 运行以下命令安装连接器。

```
*sudo apt-get install libmysql-java*
```

为了让 hive 访问 mysql 连接器，需要在 hive lib 文件夹中为连接器创建一个软链接，或者将 jar 文件复制到 hive lib 文件夹中。

```
*sudo ln -s /usr/share/java/mysql-connector-java.jar $HIVE_HOME/lib/mysql-connector-java.jar*
```

使用位于**$ hive _ HOME/scripts/metastore/upgrade/MySQL**目录中的**Hive-schema-3 . 1 . 0 . MySQL . SQL**文件(或与您安装的 Hive 版本相对应的文件)创建初始数据库模式。

> 登录到 mysql shell

```
*mysql -u root -p*
```

> 为 metastore 创建数据库

```
*CREATE DATABASE metastore;**USE metastore;**SOURCE /usr/local/hive/scripts/metastore/upgrade/mysql/hive-schema-3.1.0.mysql.sql;*
```

> 为了让 Hive 访问 metastore，需要创建一个 MySQL 用户帐户。防止该用户帐户创建或更改 metastore 数据库模式中的表非常重要。**(别忘了引号)**

```
*CREATE USER 'hiveuser'@'%' IDENTIFIED BY 'hivepassword';**GRANT all on *.* to 'hiveuser'@localhost identified by 'hivepassword';**flush privileges;*
```

> 现在我们已经在 mysql 中创建了 metastore 和 hive 用户。让我们在 *hive-site.xml.* 中定义 metastore 配置

*   **hive-site.xml** 通过运行以下命令打开 hive-site 文件

```
*cd /usr/local/hive/conf
sudo nano hive-site.xml*
```

添加以下配置

```
<configuration> 
   <property> 
      <name>javax.jdo.option.ConnectionURL</name> 
      <value>jdbc:mysql://localhost/metastore?createDatabaseIfNotExist=true</value> 
      <description>metadata is stored in a MySQL server</description> 
   </property> 
   <property> 
      <name>javax.jdo.option.ConnectionDriverName</name> 
      <value>com.mysql.jdbc.Driver</value> 
      <description>MySQL JDBC driver class</description> 
   </property> 
   <property> 
      <name>javax.jdo.option.ConnectionUserName</name> 
      <value>hiveuser</value> 
      <description>user name for connecting to mysql server</description> 
   </property> 
   <property> 
      <name>javax.jdo.option.ConnectionPassword</name> 
      <value>hivepassword</value> 
      <description>hivepassword for connecting to mysql server</description> 
   </property>
   <property> 
        <name>hive.metastore.warehouse.dir</name> 
        <value>/bigdata/hive/warehouse</value> 
        <description>location of default database for the warehouse</description> 
    </property> 
    <property> 
        <name>hive.metastore.uris</name> 
        <value>thrift://localhost:9083</value> 
        <description>Thrift URI for the remote metastore.</description> 
    </property> 
    <property> 
        <name>hive.server2.enable.doAs</name> 
        <value>false</value> 
    </property>
</configuration>
```

一切就绪…现在让我们进入蜂巢控制台。在您的终端中键入以下命令，然后点击 enter

```
*hive*
```

如果您看到任何与找不到 jdbc 驱动程序相关的错误，请检查您是否已成功将驱动程序链接或复制到 hive lib 文件夹。

## 正在启动配置单元 metastore…

现在，让我们启动 metastore 服务，在您的终端中运行以下命令，

```
*hive --service metastore*
```

如果您遇到任何错误，请验证您的 metastore 配置。
检查您的 *hive-site.xml* 配置，并检查在 mysql 中配置的 metastore db 和 user。对于与 thrift 相关的错误，请检查您的 *hive.metastore.uris* 属性中使用的端口的可用性。

> **注意:**如果您想将 hive metastore 作为系统服务运行，请执行以下步骤。

首先，让我们创建一个服务文件来启动我们的 hive metastore

```
*sudo nano /etc/systemd/system/hive-meta.service*
```

将下列行添加到服务文件中

```
[Unit] 
Description=Hive metastore 
After=network.target 

[Service] 
User=ubuntu 
Group=www-data 
ExecStart=/usr/local/hive/bin/hive --service metastore 

[Install] 
WantedBy=multi-user.target
```

现在让我们通过运行以下命令来启动我们的 hive-meta 服务

```
*sudo systemctl start hive-meta*
```

> 如果需要在系统引导时启动配置单元 metastore，请使用以下命令创建一个符号链接

```
*sudo systemctl enable hive-meta*
```

## **临时演员…**

如果您能够正确无误地进入配置单元控制台，您可以通过执行以下步骤来验证您的 metastore 配置。

在配置单元中创建一个表。

```
*create table test(id int, name string);*
```

现在从配置单元控制台退出*(键入 exit 并点击 enter)*

让我们看看这个表是否成功地添加到了我们的 metastore 中。 ***使用 hive 用户凭证登录 mysql 控制台。***

```
*mysql -u root -p*
```

更改数据库和视图表。

```
*use metastore;**select * from TBLS;*
```

如果您看到 TBLS 表中列出的 ***测试*** 表，那么您的安装是成功的。如果没有，请检查 mysql 数据库和 hive metastore 的用户配置。