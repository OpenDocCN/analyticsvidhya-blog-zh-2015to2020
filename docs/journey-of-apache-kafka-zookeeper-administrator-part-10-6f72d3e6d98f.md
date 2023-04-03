# 阿帕奇卡夫卡与动物园管理员之旅(十)

> 原文：<https://medium.com/analytics-vidhya/journey-of-apache-kafka-zookeeper-administrator-part-10-6f72d3e6d98f?source=collection_archive---------21----------------------->

2020 年 4 月(Python 作为救世主第 3 部分)

[在上一篇文章](/@116davinder/journey-of-apache-kafka-zookeeper-administrator-part-9-614993e6ca85)中，我们讨论了使用我的自定义脚本进行 MM1 监控。

> 老实说，我从来没有读过 Apache Zookeeper 的发行说明和文档，因为它只是工作。

![](img/3249302a7c7518a11be53626fe7a1d96.png)

当我在为 Apache Kafka 定制脚本时，不知何故我看到 Zookeeper 3 . 5 . 0 版有一个叫做[管理服务器](https://zookeeper.apache.org/doc/r3.6.1/zookeeperAdmin.html#sc_adminserver)的东西。

> 管理服务器
> 
> **3 . 5 . 0 中的新特性:**AdminServer 是一个嵌入式 Jetty 服务器，它为四个字母的 word 命令提供了一个 HTTP 接口。默认情况下，服务器在端口 8080 上启动，通过转到 URL“/commands/[命令名]”发出命令，例如[http://localhost:8080/commands/stat。](http://localhost:8080/commands/stat.)命令响应作为 JSON 返回。与原始协议不同，命令不限于四个字母的名称，命令可以有多个名称；例如，“stmk”也可以称为“set_trace_mask”。要查看所有可用命令的列表，请将浏览器指向 URL/命令(例如，[http://localhost:8080/commands)。](http://localhost:8080/commands).)参见 [AdminServer 配置选项](https://zookeeper.apache.org/doc/r3.6.1/zookeeperAdmin.html#sc_adminserver_config)了解如何更改端口和 URL。

我对自己说，监视 Apache Zookeeper 变得超级容易。工藤的阿帕奇动物园管理员。

最后，我决定编写一个脚本来删除这些指标，并将它们保存在 zookeeper-logs 目录中，以便 Splunk 可以读取它们。

[脚本版本一:](https://github.com/116davinder/zooki/blob/af0b0ae3c910ccd88a59fc4891e66824e2a557cd/zooki.py)

```
#!/usr/bin/env python3# usage: python3 zooki.py /zookeeper /zookeeper/zookeeper-logs/metric.out# This script suppose to export all zookeeper metric from one node and write to file
# from where either splunk like tools can read it.from urllib import request
from socket import gethostname
from datetime import datetime
import json
import sysclass zooki:
    def __init__(self):
        self.zAddr = gethostname()
        self.zPort = 8080
        self.zHttpAddr = "http://" + self.zAddr + ":" + str(self.zPort) + "/commands/"
        self.cTimeNow = str(datetime.now())def getZMetric(self, commandPath):
        with request.urlopen( self.zHttpAddr + commandPath ) as f:
            _szMetric = json.loads(f.read().decode('utf-8'))
            _szMetric["[@timestamp](http://twitter.com/timestamp)"] = self.cTimeNow
        return json.dumps(_szMetric)def main():commandPaths = ['connections', 'dump', 'leader', 'monitor',
                    'observers', 'ruok', 'server_stats',
                    'voting_view', 'watch_summary',  'watches_by_path', 'zabstate']with open(sys.argv[2], "w") as zMetricFile:
        z = zooki()
        for c in commandPaths:
            zMetricFile.write("\n")
            zMetricFile.write(z.getZMetric(c))main()
```

还有一件小事要记住，我正在更新从 Zookeeper 废弃的 JSON，因为我必须在其中插入时间戳。

剧本很好。

**注:** [版本一脚本/库](https://github.com/116davinder/zooki)已存档。

一旦 Splunk 开始索引这些指标，我就意识到了问题所在。
1。 **Splunk 在指数指标上苦苦挣扎。
2。指标太多。**

> Splunk 对每一行都有限制，因此如果行大于 20 KB(可能有所不同),它就会跳过它，而您不知道为什么行/指标没有被索引。

为了解决上述问题，我创建/修改了脚本以减少工作量并使 Splunk 正常工作。

```
# usage: python3 zooki.py /zookeeper /zookeeper/zookeeper-logs/ dev-zookeeper# This script suppose to export all zookeeper metric from one node and write to file
# from where splunk/cloudwatch like tools can read it.from urllib import request
import shutil
from socket import gethostname
from datetime import datetime
import json
import sys
import os.pathclass zooki:
    def __init__(self):
        self.zAddr = gethostname()
        self.zPort = 8080
        self.zHttpAddr = "http://" + self.zAddr + ":" + str(self.zPort) + "/commands/"
        self.cTimeNow = str(datetime.now())def getStorageMetric(self):
        total, used, free = shutil.disk_usage(sys.argv[1])
        _sMetric = {
                    "[@timestamp](http://twitter.com/timestamp)": self.cTimeNow,
                    "command": "disk",
                    "environment": sys.argv[3],
                    "totalInGB": total // (2**30),
                    "usedInGB":  used // (2**30),
                    "freeInGB": free // (2**30)
                    }
        return json.dumps(_sMetric)def getZMetric(self, commandPath):
        with request.urlopen( self.zHttpAddr + commandPath ) as f:
            if f.status == 200:
                _zMetric = json.loads(f.read().decode('utf-8'))
                _zMetric["[@timestamp](http://twitter.com/timestamp)"] = self.cTimeNow
                _zMetric["environment"] = sys.argv[3]
            else:
                _zMetric = {}
        return json.dumps(_zMetric)# json retruned by monitor is too big to be handled by splunk indexer
# so separate function to reduce the json size
    def getMonitorMetric(self):
        with request.urlopen( self.zHttpAddr + "monitor" ) as f:
            if f.status == 200:
                _MM = json.loads(f.read().decode('utf-8'))
                _zMetric = {
                    "[@timestamp](http://twitter.com/timestamp)": self.cTimeNow,
                    "environment": sys.argv[3],
                    "command": _MM["command"],
                    "znode_count": _MM["znode_count"],
                    "watch_count": _MM["watch_count"],
                    "outstanding_requests": _MM["outstanding_requests"],
                    "open_file_descriptor_count": _MM["open_file_descriptor_count"],
                    "ephemerals_count": _MM["ephemerals_count"],
                    "max_latency": _MM["max_latency"],
                    "avg_latency": _MM["avg_latency"],
                    "synced_followers": _MM["synced_followers"] if _MM["server_state"] == "leader" else 0,
                    "pending_syncs": _MM["pending_syncs"] if _MM["server_state"] == "leader" else 0,
                    "version": _MM["version"],
                    "quorum_size": _MM["quorum_size"],
                    "uptime": _MM["uptime"]
                }
            else:
                _zMetric = {}
        return json.dumps(_zMetric)def main():commandPaths = ['connections', 'leader', 'watch_summary']z = zooki()
    with open(os.path.join(sys.argv[2], "disk.out"), "w") as zMetricFile:
        zMetricFile.write(z.getStorageMetric())for command in commandPaths:
        with open(os.path.join(sys.argv[2], command, ".out"), "w") as zMetricFile:
            zMetricFile.write(z.getZMetric(command))with open(os.path.join(sys.argv[2], "monitor.out"), "w") as zMetricFile:
        zMetricFile.write(z.getMonitorMetric())
main()
```

**脚本的主要变化:**

*   减少要运行的度量命令的数量。
*   为“monitor”命令创建一个单独的函数，以便从中剥离输出。
*   为不同的指标创建单独的输出文件，这样 Splunk 可以更快地索引它们，以后的查询也会更快一些。
*   添加了系统磁盘指标。
*   与 Ansible 代码集成。
*   插入环境标签和命令标签。

**在 Ansible 代码中创建了新的角色:**[116 davinder/zookeeper-cluster-ansi ble/roles/customMetricExporter](https://github.com/116davinder/zookeeper-cluster-ansible/tree/master/roles/customMetricExporter)

一个可行的任务应该做两件事

*   复制脚本。
*   用参数为上面的脚本创建 cron 任务。

```
---- name: copying zooki.py to {{ zookeeperInstallDir }}
  copy:
    src: zooki.py
    dest: "{{ zookeeperInstallDir }}"
    owner: "{{ zookeeperUser }}"
    group: "{{ zookeeperGroup }}"
    mode: 0777- name: cron for zookeeper metric collector ( zooki.py )
  cron:
    name: "zookeeper metric collector"
    minute: "*"
    hour: "*"
    weekday: "*"
    user: root
    job: 'find /bin/ -name "python3*m" -print0 -exec {} {{ zookeeperInstallDir }}/zooki.py {{ zookeeperInstallDir }} {{ zookeeperLogDir }}/  {{ zookeeperEnvironment }} \;'
```

在 Ansible Task 中还有一件有趣的事情需要注意，那就是如何找到 python3 二进制文件并运行它。我在不同的环境中见过不同的安装，它们有不同的二进制名称，这导致了默认命令“python3”的问题，所以我添加了小的 bash 命令来变魔术。

```
find /bin/ -name "python3*m" -print0 -exec {} zooki.py ......
```

我的脚本适用于所有环境，Splunk 也没有问题。

现在，是我动手使用 Splunk 仪表盘的时候了。

> 直到这个时候，我还在 Splunk 仪表盘上为我的用例做忍者。

**Splunk Dashboard Code**:[116 davinder/zookeeper-cluster-ansi ble/Splunk-dashboards/Apache-zookeeper . XML](https://github.com/116davinder/zookeeper-cluster-ansible/blob/master/files/splunk%20dashboards/apache-zookeeper.xml)
**Splunk Dashboard Sample:**不好意思，忘记上传到 GitHub:(。

这一旅程将在下一篇文章中继续(云与地形的乐趣)