# 阿帕奇卡夫卡与动物园管理员之旅(下)

> 原文：<https://medium.com/analytics-vidhya/journey-of-apache-kafka-zookeeper-administrator-part-2-38db736d3163?source=collection_archive---------9----------------------->

![](img/e9f52e6864a487961809e95d7c9dafe5.png)

【2019 年 6 月(续) (阿帕奇动物园管理员)

[在上一篇文章](/@116davinder/journey-of-apache-kafka-zookeeper-administrator-part-1-d84dfde205b)中，我已经解释了安装文件夹的结构，所以我必须实现它，这就是 **Ansible 在 Apache Zookeeper 上展示其魔力的地方。**

我已经写了一些可行的剧本，自动化了 Apache Zookeeper 管理的各个方面。

**举例:
1。基本设置
2。配置修改，如 Jvm /日志记录等。
3。OS + Zookeeper
4 的生产优化。升级集群
5。记录、监控&警报设置**

**GitHub 代码库:**[116 dav ender/zookeeper-cluster-ansi ble](https://github.com/116davinder/zookeeper-cluster-ansible)

[**常见的**](https://github.com/116davinder/zookeeper-cluster-ansible/tree/master/roles/common)它包含了 Apache Zookeeper 设置过程的基本任务，类似于
1。安装像 wget/tar/nc/net-tools 这样的包。
2。创建 zookeeper 用户和组。
3。创建所需的目录，如数据&日志。
4。像操作系统/网络/文件系统一样进行系统调优。
5。当前禁用系统防火墙或 IP 表。

[**Java**](https://github.com/116davinder/zookeeper-cluster-ansible/tree/master/roles/java)该角色将允许用户安装/升级不同版本的 Java，如 1.8 / 11/ 13 / 14 /等。

[**安装**](https://github.com/116davinder/zookeeper-cluster-ansible/tree/master/roles/install)1。它将把 Apache Zookeeper Tar.gz 从 Ansible 服务器上传到 Zookeeper 节点。
2。它会把 Tar.gz 解压到一个给定的位置，就像“/zookeeper”文件夹一样。
3。它将为 zookeeper 创建一个到 Apache Zookeeper 给定版本的符号链接。
4。它将为环境设置创建“/etc/profile.d/zookeeper.sh”。

[**配置**](https://github.com/116davinder/zookeeper-cluster-ansible/tree/master/roles/configure)这个角色实际上会为 Apache Zookeeper 创建/更新配置。
1。创建或更新**zoo . CFG**/**log4j . properties**/**Java . env .** 2。 **myid** 是根据 IP 地址为每个主机自动生成的。

[**服务**](https://github.com/116davinder/zookeeper-cluster-ansible/tree/master/roles/service)该角色将为 Apache Zookeeper 创建/更新 SystemD 文件。

[**service state**](https://github.com/116davinder/zookeeper-cluster-ansible/tree/master/roles/serviceState)该角色将允许用户重启/停止/启动服务。

[**端口检查**](https://github.com/116davinder/zookeeper-cluster-ansible/tree/master/roles/portCheck)该角色允许用户对给定端口进行基本检查，状态为开启或关闭。

[**Nri-Zookeeper**](https://github.com/116davinder/zookeeper-cluster-ansible/tree/master/roles/nri-zookeeper)该角色将允许用户为 Apache Zookeeper 安装基于新遗迹的监控设置。

# **动手吧:)**

当我在为 Apache Zookeeper 编写可翻译的剧本时，我意识到有几个问题应该首先解决。

**问题一:**如何为每个 zookeeper 节点生成可预测的 **myid** ？

```
**Method 1: # Using Jinja 2 / Native Ansible ( Preferred )** {% set id = hostvars[inventory_hostname]['ansible_default_ipv4']['address'].split('.')[3] | int | abs %}{{ id }}**Method 2: # Using Bash Shell** shell: "echo {{ ansible_ssh_host }} | cut -d . -f 4"
```

**问题二:**如何为 Zookeeper 设置 JVM 属性？

谷歌上的研究为我提供了几个选项，但没有一个是本地的，所以我最终检查了 Apache Zookeeper 文档，他们建议使用 **java.env** 但后来我想知道我必须在这个文件中使用什么格式，在谷歌上搜索了几次后，点击&重试了一次，一种格式对我有效。

```
export JVMFLAGS="-Xmx{{ zookeeperXmx }} -Xms{{ zookeeperXms }}"
```

**问题三:**如何设置生成**动物园。每台服务器的 cfg** ？

另一个问题是，我必须将每个动物园管理员节点地址添加到**动物园中。cfg** 及其 **myid** 。

```
**Method 1: # Using Jinja 2 / Native Ansible ( Preferred )**
{% for host in groups['clusterNodes'] %}
{% if not host | ipaddr %}
{% set ip = hostvars[host]['ansible_default_ipv4']['address'] %}
{% else %}
{% set ip = host %}
{% endif %}
{% set id = ip.split('.')[3] | int | abs %}
server.{{ id }}={{ ip }}:2888:3888
{% endfor %}**Method 2: # Using Bash Shell** shell: "echo server.$(echo {{ item }} | cut -d . -f 4)={{ item }}:2888:3888 >> {{ zookeeper_install_dir }}/conf/zoo.cfg"
loop:
  - "{{ groups['zookeeper'] }}"
```

**问题四:**优化
为什么要优化？优化什么？如何优化？
我不得不阅读很多关于红帽 7 为 Apache Zookeeper 优化的博客。最后，我能够收集操作系统/网络/ulimit 的不同调整。
**ansi ble Code:**[system tuning . yml](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/roles/common/tasks/systemTuning.yml)

**问题五:**如何监控日志？

这是最简单的一个，我必须使用 Splunk。我对阿帕奇卡夫卡和阿帕奇动物园管理员使用了一个索引。

```
[default]
host = $HOSTNAME

[monitor:///zookeeper/zookeeper-logs/*.out]
disabled = false
index = kafka
sourcetype = zookeeper
crcSalt = <SOURCE>
```

**问题六:**如何&用什么来监控 Apache Zookeeper？

这也是最简单的解决方案之一，我不得不使用新的遗物。
按照 [newrelic/nri-zookeeper](https://github.com/newrelic/nri-zookeeper) 的步骤操作即可。

在遵循新遗迹指南后，我发现我已经部署了超过 10 个不同的 Apache Zookeeper 集群，因此我将如何在新遗迹仪表板中区分这些集群，新遗迹有一个非常优雅的解决方案。更新 **zookeeper-config.yml** 以使用**标签**。现在，每个集群都将拥有自己唯一的环境名称。

```
integration_name: com.newrelic.zookeeperinstances:
  - name: {{ ansible_fqdn }}
    command: metrics
    arguments:
      host: localhost
      port: 2181
      cmd: nc
    labels:
      role: zookeeper
      env: {{ zookeeperEnvironment }}
```

一旦我能够发布 Apache Zookeeper 指标，然后我意识到系统指标(CPU /内存/磁盘)也是使用 **New Relic Infra Agent** 发布的，我必须在我的仪表板中使用这些指标，所以我对如何轻松找到这些指标做了更多的研究。幸运的是:)新遗迹基础设施代理也支持标签，所以我只需要为相同的标签更新它的配置(**/etc/New Relic-Infra . yml**)。

```
custom_attributes:
  label.env: {{ zookeeperEnvironment }}
```

**终于！Ansible Code 工作了:)** 因为上面提到的问题，我花了几天时间。“如果你坚持不懈，那么问题最终会消失”。

我对我的 Ansible 代码使用了一种灵活的方法，因为如果需要的话，我想单独更新 Apache Zookeeper 的每个进程/配置，如 JVM / Logging / Zoo.cfg /等。

**基础剧本**
[**cluster setup . yml**](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/clusterSetup.yml)**:**它会在给定的环境中安装 Apache Zookeeper。
[**clusternewrelicsetup . yml**](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/clusterNewRelicSetup.yml)**:**将安装新的遗迹监控设置。[**cluster upgrade . yml**](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/clusterUpgrade.yml)**:**它会将 Apache Zookeeper 升级到新版本。

**维护行动手册
注意*:** 以下行动手册将以滚动方式重启 Apache Zookeeper，以避免停机。[**cluster Java . yml**](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/clusterJava.yml)**:**它将安装/更新 Java 包。
[**cluster jvmconfigs . yml**](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/clusterJvmConfigs.yml)**:**会更新**Java . env .**
[**cluster logging . yml**](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/clusterLogging.yml)**:**会更新 **log4j.properties** 。
[**clusterremovenodes . yml**](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/clusterRemoveNodes.yml)**:**它将使节点退役。
[**clusterremoveoldversions . yml**](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/clusterRemoveOldVersions.yml)**:**它将删除旧版本的 configs 文件夹。
[**clustersystemupgrade . yml**](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/clusterSystemUpgrade.yml)**:**需要的话会升级 OS。
[**clusterrollingrestart . yml**](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/clusterRollingRestart.yml)**:**它会做一个 Apache Zookeeper 的滚动重启。

**手动步骤:(** 创建 New Relic Dashboard 是另一个挑战，因为在此之前我从未使用过它，这对我来说是一个学习曲线。

***需要记住的几件事是，New Relic Infra Agent 将指标发布到 New Relic Insights 中的不同数据库。***
**系统示例:**用于存储 CPU 指标。
**StorageSample:** 用于存储磁盘规格。
**NetworkSample:** 用于存储网络度量。
**ZookeeperSample:** 用于存储实际的 Zookeeper 指标。

使用 [New Relic API Explorer](https://docs.newrelic.com/docs/apis/rest-api-v2/api-explorer-v2/use-api-explorer) 导入下面的 dashboard JSON 代码。
**新遗迹仪表盘代号:**[New Relic-Dashboard-zookeeper . JSON](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/files/newrelic-dashboard/zookeeper.json)
**新遗迹仪表盘样本:**[Apache-Zookeeper.pdf](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/files/newrelic-dashboard/Apache%20Zookeeper.pdf)

[我的 GitHub 库](https://github.com/116davinder/zookeeper-cluster-ansible)也有其他剧本/角色，但我会在下一篇文章中介绍它们，因为这是我的故事，而这篇文章不适合它们。

阿帕奇卡夫卡之旅将在下篇开始！