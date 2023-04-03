# 阿帕奇卡夫卡与动物园管理员之旅(九)

> 原文：<https://medium.com/analytics-vidhya/journey-of-apache-kafka-zookeeper-administrator-part-9-614993e6ca85?source=collection_archive---------23----------------------->

2020 年 4 月(Python 作为救世主第二部分)

[在上一篇文章中，](/@116davinder/journey-of-apache-kafka-zookeeper-administrator-part-8-acdc030302ba)我解释了如何使用 python 来提取指标，对它们进行处理，并编写它们以供进一步处理。

![](img/da4c799b033121cce1fc55502c547ed6.png)

礼貌:cloudkarafka.com

让我们继续讨论**Kafka Mirror Maker v1**aka**MM1**Monitoring，因为我不得不移除 New Relic，而 Appdynamics 没有达到标准。我开始扩展脚本以支持 MM1。老实说，它已经支持了，MM1 只需要一些小的改动。

**监控脚本中 MM1 的调整列表**

*   用于清理的[域列表](https://github.com/116davinder/kafka-cluster-ansible/blob/master/roles/jmxMonitor/files/kafka-jmx-metric-collector-mm.py#L27)中的更改。
*   添加了另一个名为流程名称的标签。

脚本的其余部分与**系统指标**和**线程**非常相似。

```
#!/usr/bin/env python3# usage: python3 roles/jmxMonitor/files/kafka-jmx-metric-collector-mm.py localhost 9981 roles/jmxMonitor/files/kafka-mirror-input.txt /tmp/ vie-prod-kafka process-1# This script suppose to export all kafka metric from one node and write to file
# from where Splunk like tools can read it.from jmxquery import *
from socket import gethostname
from datetime import datetime
import json
import sys
import psutil
import threadingclass KafkaJmx:
    def __init__(self,kAddr,kPort,inputFile,logDir,env,processName):
        self.kAddr = kAddr
        self.kPort = kPort
        self.kJmxAddr = "service:jmx:rmi:///jndi/rmi://" + str(self.kAddr) + ":" + str(self.kPort) + "/jmxrmi"
        self.cTimeNow = datetime.now()
        self.jmxConnection = JMXConnection(self.kJmxAddr)
        self.inputFile = inputFile
        self.logDir = logDir
        self.env = env
        self.processName = processName
        self.domainNameList = ['java.lang','kafka.consumer','kafka.producer','kafka.tools']def getMetric(self):
        with open(self.inputFile) as file:
            for query in file:
                metrics = self.jmxConnection.query([JMXQuery(query.strip())], timeout=1000000)
                for metric in metrics:
                    domainName = metric.to_query_string().split(":")[0]
                    queryName = metric.to_query_string().split(":")[1]
                    queryValue = metric.value
                    _queryDict = {
                                "[@timestamp](http://twitter.com/timestamp)": str(self.cTimeNow),
                                "domainName": str(domainName),
                                "environment": str(self.env),
                                "processName": str(self.processName),
                                "queryName": str(queryName),
                                "queryValue":  queryValue
                                }
                    with open(self.logDir + domainName + ".log", 'a+') as logFile:
                       logFile.write("\n")
                       logFile.write(json.dumps(_queryDict))def cleanUpFiles(self):
        for domainName in self.domainNameList:
            open(self.logDir + domainName + ".log", 'w').close()def getStorageMetric(self):
        _sMM = psutil.disk_usage("/kafka")
        _sMetric = {
                    "[@timestamp](http://twitter.com/timestamp)": str(self.cTimeNow),
                    "domainName": "disk",
                    "environment": self.env,
                    "totalInGB": _sMM.total // (2**30),
                    "usedInGB":  _sMM.used // (2**30),
                    "freeInGB": _sMM.free // (2**30),
                    "usedPercent": _sMM.percent
                    }with open(self.logDir + "disk.log", 'w') as logFile:
            logFile.write(json.dumps(_sMetric))def getCpuMetric(self):
        _cMetric = {
                    "[@timestamp](http://twitter.com/timestamp)": str(self.cTimeNow),
                    "domainName": "cpu",
                    "environment": self.env,
                    "usedCpuPercent": psutil.cpu_percent()
                    }with open(self.logDir + "cpu.log", 'w') as logFile:
            logFile.write(json.dumps(_cMetric))def getMemoryMetric(self):
        _memStats = psutil.virtual_memory()
        _swapMemStats = psutil.swap_memory()
        _rMetric = {
                    "[@timestamp](http://twitter.com/timestamp)": str(self.cTimeNow),
                    "domainName": "memory",
                    "environment": self.env,
                    "totalMem": _memStats.total // (2**30),
                    "availableMem": _memStats.available // (2**30),
                    "percentUsedMem": _memStats.percent,
                    "usedMem": _memStats.used // (2**30),
                    "buffers": _memStats.buffers // (2**30),
                    "totalSwap": _swapMemStats.total // (2**30),
                    "usedSwap": _swapMemStats.used // (2**30),
                    "freeSwap": _swapMemStats.free // (2**30),
                    "percentUsedSwap": _swapMemStats.percent
                    }with open(self.logDir + "memory.log", 'w') as logFile:
            logFile.write(json.dumps(_rMetric))def main():
    hostname = sys.argv[1]
    port = sys.argv[2]
    inputFile = sys.argv[3]
    logDir = sys.argv[4]
    env = sys.argv[5]
    processName = sys.argv[6]z = KafkaJmx(hostname, port, inputFile, logDir,env,processName)
    z.cleanUpFiles()
    _metric_thread = threading.Thread(
        target=z.getMetric
    ).start()_cpu_metric_thread = threading.Thread(
        target=z.getCpuMetric
    ).start()_memory_metric_thread = threading.Thread(
        target=z.getMemoryMetric
    ).start()_storage_metric_thread = threading.Thread(
        target=z.getStorageMetric
    ).start()main()
```

在同一个 [jmxMonitor](https://github.com/116davinder/kafka-cluster-ansible/tree/master/roles/jmxMonitor) 角色中添加上述脚本

*   脚本文件:[JMX monitor/Kafka-JMX-metric-collector-mm . py](https://github.com/116davinder/kafka-cluster-ansible/blob/master/roles/jmxMonitor/files/kafka-jmx-metric-collector-mm.py)
*   输入文件:[JMX monitor/Kafka-mirror-input . txt](https://github.com/116davinder/kafka-cluster-ansible/blob/master/roles/jmxMonitor/files/kafka-mirror-input.txt)
*   可完成的任务:[JMX monitor/Kafka-mirror-maker . yml](https://github.com/116davinder/kafka-cluster-ansible/blob/master/roles/jmxMonitor/tasks/kafka-mirror-maker.yml)

Ansible Task 中唯一有趣的事情是生成应该被监控的动态名称+端口列表。

```
- name: kafka mirror metric collector cron
  cron:
    name: "kafka mirror metric collector cron task {{ item }}"
    minute: "*"
    hour: "*"
    weekday: "*"
    user: root
    job: 'find /bin/ -name "python3*m" -print0 -exec {} {{ kafkaInstallDir }}/jmxMonitor/kafka-jmx-metric-collector-mm.py {{ ansible_fqdn }} {{ kafkaMirrorMakerJmxInitialPort + item }} {{ kafkaInstallDir }}/jmxMonitor/kafka-mirror-input.txt {{ kafkaLogDir }}/Kafka-Mirror-Maker-Process-{{ item }}- {{ kafkaClusterName }} Kafka-Mirror-Maker-Process-{{ item }} \;'
  loop: "{{ range(1, kafkaMirrorMakerProcessCountPerNode + 1, 1) | list }}"
```

现在让我们利用 Splunk 来展示这些指标。

> **Splunk 仪表盘代码:**[Splunk-dashboards/Kafka-Mirror-Maker-v1 . XML](https://github.com/116davinder/kafka-cluster-ansible/blob/master/files/splunk-dashboards/apache-kafka-mirror-maker-v1.xml)
> **Splunk 仪表盘示例:**[Splunk-dashboards/Mirror _ Maker _ v1 . png](https://github.com/116davinder/kafka-cluster-ansible/blob/master/files/splunk-dashboards/Apache_Kafka_Mirror_Maker_v1_Splunk_7_2_9_1.png)

上面的仪表板可以支持我所有的不同环境，它有两个基本的过滤器
1。**环境名称**2
。**流程名称**

希望您会发现它简单且易于实现。

旅程将继续到下一篇文章(Zookeeper 的基于 Python 的监控)。