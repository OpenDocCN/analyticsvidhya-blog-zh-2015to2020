# 通过 Docker 在 Cloudera 集群中安装 Apache Kafka

> 原文：<https://medium.com/analytics-vidhya/installing-kafka-in-cloudera-cluster-through-docker-26ba92004c7e?source=collection_archive---------12----------------------->

## 在运行于 Docker 容器上的 Cloudera quickstart 中添加 Apache Kafka 服务的分步安装指南

![](img/25a3ba351df95b3a4e718fc6a865c601.png)

阿帕奇卡夫卡的知名度与日俱增。它由 LinkedIn 于 2010 年开发，Teach industries 正在将 Hadoop 生态系统服务用于各种流媒体和消息传递用例，其数量正在与日俱增。Kafka 如此受欢迎的原因。

*请参考这些文章*[*Docker*](/@ajpatel.bigdata/cloduera-quickstart-vm-using-docker-on-mac-2308acd196f2)*[*JDK*](/@vishnusharath/upgrade-docker-cdh-image-to-jdk1-8-to-use-spark-2-a-complete-tutorial-61a1b67b2597)*[*CDH*](/@ajpatel.bigdata/upgrading-cdh-and-cloudera-manager-from-5-7-to-5-16-in-docker-to-use-spark2-6223691e3685)*如果不满足必备条件，***

## **安装由 Apache Kafka 驱动的 CD**

> **环境:
> 1。Java 版本 8
> 2。Cloudera Manager 5.16
> 3。Cloudera 发行版 Hadoop 5.16**

**注意:我为这个博客运行 Cloudera Docker 容器，这里是内存和存储分配。**

**![](img/6cf5b17138a734387be15dbb3421d20e.png)****![](img/02f0123e18bcf789c1793bb7c9105285.png)**

## **选择您想要安装的 Kafka 版本**

**导航到包含可用 Kafka 版本列表的 URL**

 **[## CDK 由阿帕奇卡夫卡版本和包装信息

### 下表列出了访问每个 Kafka 工件所需的项目名称、groupId、artifactId 和版本…

docs.cloudera.com](https://docs.cloudera.com/documentation/kafka/latest/topics/kafka_packaging.html?source=post_page-----8245d8d0ebe5----------------------#concept_fzg_phl_br)** 

**复制要安装的包存储库 URL。**

**现在，回到 [http://localhost:7180/](http://localhost:7180/) 。(注意:由于我们重新启动 CM 服务，这将需要几分钟……)**

**用户名:管理员
密码:管理员**

**登录 Cloudera Manager Web UI。**

**![](img/f42732c1fa273e64a83d8c3d1e478e19.png)****![](img/939f3b169721ecd79e9167b4551ab424.png)**

**导航至主机→宗地。**

**![](img/86f2a228b810a2d25ed082aba4ccd66a.png)**

**将复制的 Parcel_URL 添加到宗地配置中。
注:我们选择安装 Kafka 4.1.0 版本。**

**保存更改。**

**![](img/b4bec03536b6bd8445c24811ee3a4b09.png)**

**等待几秒钟，添加的 Kafka 服务将出现在包裹列表中。**

**现在，下载→分发→激活它。**

**![](img/20db656f76038c1fd8ea5712c3904798.png)****![](img/facd86b42ff48345ad66352f4bd3523d.png)****![](img/d60b6c99ee1beaa331e9ab9d62597469.png)**

**验证 Kafka 包已激活，并且可以将 Kafka 服务添加到 Cloudera 集群。**

**![](img/6d34fd0121671a7268a27d7306e3ceb2.png)**

**导航至 Cloudera Manager 主页。**

## **将选定的 Kafka 服务添加到集群中**

**现在让我们将它添加到 Cloudera 集群中。点击“添加服务”。**

**![](img/5daa4581fd8a4bd99edc14b5e7397179.png)**

**选择 Kafka 选项并继续。**

**![](img/8c0e79dd7936f7151cee119c1f91d571.png)**

**为 Kafka Broker 和 Gateway 分配服务器实例并继续。**

**![](img/23277543898bcc1d1a863fa3d0eace7f.png)****![](img/e7d20b09031244101bcf4c3d7e700e3a.png)****![](img/69c9f9128e5d2ccf15e038e71d459ec9.png)**

**在此步骤中，保持这些配置不变并继续。**

**![](img/fbacf57ab80ff2afd996b7c6435b2486.png)**

**现在 Kafka 服务被添加到集群中。**

**如果您发现此错误，请不要担心。这是因为一些默认的配置。我们将在以下步骤中解决这些错误。**

**![](img/7fc67c4213e9cf0ec186633d48b27130.png)**

**让我们回到 CM 主页。**

**![](img/32dd92b38e91c707d91f06d9375f0d0a.png)**

**错误:服务将处于“已停止”模式，并显示严重警告消息。**

**![](img/46dd689579b4b78d844cd052592b8cde.png)**

**现在，让我们用所需的配置来修复这个错误。**

**![](img/f04da6964056669292f8a567269b50b2.png)**

```
**Java Heap Size of Broker = 256 MiB**
```

**![](img/4988a0978feb29fb09f224fdc222500c.png)**

```
**Advertised Host = quickstart.cloudera**
```

**![](img/fbb75dac53e05f8ef25b8d1057917798.png)**

```
**Inter Broker Protocol = PLAINTEXT**
```

**![](img/e1690bedae0ccc87228ad1f1e64818ae.png)**

**保存这些更改并重新启动 Kafka 服务。**

**![](img/9770a091450c4b464667098dbdc3249b.png)****![](img/1429fea254a67949b2568c67bdabcb14.png)****![](img/c81fb23d00539dcf404eb3e3dd7db964.png)**

**现在，验证 Kafka 服务是否处于“良好健康”状态。**

**![](img/82079ca18fd1efba257907591a221ab1.png)**

**恭喜你！现在，您可以使用 Kafka 服务了。**

****让我们创建一个 Kafka 应用程序:****

**首先，打开终端，创建一个卡夫卡主题。**

```
**kafka-topics — zookeeper quickstart.cloudera:2181 — create — topic first_topic — replication-factor 1 — partitions 3**
```

**![](img/3ab60c0e59862fbd0b89db9ba2b80b8a.png)**

**查看可用的 Kafka 主题列表。**

```
**kafka-topics — zookeeper quickstart.cloudera:2181 — list**
```

**![](img/ffb17212072b4496486c72c43ac13501.png)**

**启动 Kafka 控制台生产者:(Kafka-生产者终端)**

```
**kafka-console-producer — broker-list quickstart.cloudera:9092 — topic first_topic**
```

****启动卡夫卡控制台【消费者:】**
(卡夫卡-消费者-终端 1)**

```
**kafka-console-consumer — bootstrap-server quickstart.cloudera:9092 — topic first_topic — from-beginning — partition 0**
```

**(卡夫卡-消费者-2 号航站楼)**

```
**kafka-console-consumer — bootstrap-server quickstart.cloudera:9092 — topic first_topic — from-beginning — partition 1** 
```

**(卡夫卡-消费者-3 号航站楼)**

```
**kafka-console-consumer — bootstrap-server quickstart.cloudera:9092 — topic first_topic — from-beginning — partition 2**
```

**在我的下一篇文章中再见。关注我以获得更多关于数据工程的更新。干杯！！！⚔️**