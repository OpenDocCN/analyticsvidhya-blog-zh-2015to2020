# Apache Flink 系列 6—读取日志文件

> 原文：<https://medium.com/analytics-vidhya/apache-flink-series-6-reading-the-log-files-6e58ad7b542b?source=collection_archive---------5----------------------->

![](img/10e13ec967c87198da8962a0feef054c.png)

在本帖中，我们将查看日志文件(TaskManager 和 JobManager 的日志文件),并尝试了解 Flink 集群上发生了什么。

实际上这篇文章是关于创建样本 Flink 集群的第 3 步。然而，我只是认为阅读日志文件对理解任何工具/框架等的基础知识会有很大的帮助..因此我决定为此写一篇博文。

> 你可以参考我之前关于 [**创建样本 Apache Flink 集群 Part-1**](/@mehmetozanguven/apache-flink-series-5-create-sample-apache-flink-cluster-on-local-machine-part-1-5af20c5a5c8f) 的博客

在之前的博客中，我们可以用一个 JobManager 和一个 TaskManager 运行简单的 flink 集群。

让我们重新处理一下，只有一个任务槽(在 flink-conf.yaml 文件中将 numberOfTaskSlots 设置为 1):

```
taskmanager.numberOfTaskSlots: 1
```

现在使用`$ pathToFlink/bin/start-cluster.sh``运行您的集群

> 注意:如果您以前运行过 flink cluster，您可以从以前的设置中删除日志。否则，flink 会将新创建的日志附加到现有的日志中

# 作业管理器的日志文件

```
2020-02-28 00:03:44,009 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -  Starting StandaloneSessionClusterEntrypoint (Version: 1.10.0, Rev:aa4eb8f, Date:07.02.2020 @ 19:18:19 CET)
```

因为我们正在创建独立集群，Flink 也将为独立集群创建适当的类。如果您查看 Flink 源代码，您会看到 **ClusterEntrypoint 是一个抽象类，有一个名为 standalonessessionclusterentry point 的类扩展了它。**

```
(1)
2020-02-28 00:03:44,009 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -  OS current user: mehmetozanguven(2)
2020-02-28 00:03:44,010 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -  Current Hadoop/Kerberos user: <no hadoop dependency found>(3)
2020-02-28 00:03:44,010 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -  JVM: OpenJDK 64-Bit Server VM - AdoptOpenJDK - 11/11.0.6+10(4)
2020-02-28 00:03:44,010 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -  Maximum heap size: 1024 MiBytes(5)
2020-02-28 00:03:44,010 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -  JAVA_HOME: /home/mehmetozanguven/JavaJDKS/jdk-11.0.6+10
```

提供信息的日志(*环境，如代码修订、当前用户、Java 版本和 JVM 参数)*，如上所述，将在方法中打印:

```
org.apache.flink.runtime.util.EnvironmentInformation.logEnvironmentInfo(...)
```

例如 java_home 组件是这样读的

```
logEnvironmentInfo(...) {
    // ...
    String javaHome = System.*getenv*("JAVA_HOME");
    //...
}
```

以及所有关于系统、环境等的信息..将打印在这里:

```
logEnvironmentInfo(...){
    log.info("--------------------------------------------------------------------------------");
    log.info(" Starting " + componentName + " (Version: " + version + ", "
      + "Rev:" + rev.commitId + ", " + "Date:" + rev.commitDate + ")");
    log.info(" OS current user: " + System.*getProperty*("user.name"));
    log.info(" Current Hadoop/Kerberos user: " + *getHadoopUser*());
    log.info(" JVM: " + jvmVersion);
    log.info(" Maximum heap size: " + maxHeapMegabytes + " MiBytes");
    log.info(" JAVA_HOME: " + (javaHome == null ? "(not set)" : javaHome));
}
```

之后，我们会在作业管理器的日志文件中看到这些日志:

```
2020-02-28 00:03:44,010 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -  JVM Options:2020-02-28 00:03:44,010 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -     -Xms1024m2020-02-28 00:03:44,010 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -     -Xmx1024m2020-02-28 00:03:44,011 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -     -Dlog.file=/home/mehmetozanguven/Desktop/ApacheTools/flink-1.10.0/log/flink-mehmetozanguven-standalonesession-0-mehmetozanguven-ABRA-A5-V5.log2020-02-28 00:03:44,011 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -     -Dlog4j.configuration=file:/home/mehmetozanguven/Desktop/ApacheTools/flink-1.10.0/conf/log4j.properties2020-02-28 00:03:44,011 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -     -Dlogback.configurationFile=file:/home/mehmetozanguven/Desktop/ApacheTools/flink-1.10.0/conf/logback.xml
```

除了 Flink 读取启动时传递的 JVM 选项，您可能认为“我们只是通过。sh 文件谁传递了-Dlog.file/log4j 等..?"

这些系统属性由另一个。sh 文件名为`config.sh`

这个文件查看 flink-conf.yaml，如果它找到任何这些系统属性的键值参数，它将使用它，否则将使用默认值。

```
2020-02-28 00:03:44,011 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -  Program Arguments:2020-02-28 00:03:44,011 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -     --configDir2020-02-28 00:03:44,011 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -     /home/mehmetozanguven/Desktop/ApacheTools/flink-1.10.0/conf2020-02-28 00:03:44,011 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -     --executionMode
2020-02-28 00:03:44,011 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -     cluster2020-02-28 00:03:44,011 INFO  org.apache.flink.runtime.entrypoint.ClusterEntrypoint         -  Classpath: /home/mehmetozanguven/Desktop/ApacheTools/flink-1.10.0/lib/flink-table_2.11-1.10.0.jar:/home/mehmetozanguven/Desktop/ApacheTools/flink-1.10.0/lib/flink-table-blink_2.11-1.10.0.jar:/home/mehmetozanguven/Desktop/ApacheTools/flink-1.10.0/lib/log4j-1.2.17.jar:/home/mehmetozanguven/Desktop/ApacheTools/flink-1.10.0/lib/slf4j-log4j12-1.7.15.jar:/home/mehmetozanguven/Desktop/ApacheTools/flink-1.10.0/lib/flink-dist_2.11-1.10.0.jar:::
```

Flink 读取程序参数(命令行参数)并在找到时打印出来。然后 Flink 也会打印类路径来执行 java 类。这些信息也在 environment information . logenvironmentinfo(…)方法中打印出来

```
if (commandLineArgs == null || commandLineArgs.length == 0) {
   log.info(" Program Arguments: (none)");
}
else {
   log.info(" Program Arguments:");
   for (String s: commandLineArgs) {
      log.info("    " + s);
   }
}

log.info(" Classpath: " + System.*getProperty*("java.class.path"));

log.info("--------------------------------------------------------------------------------");
```

如前所述，这些参数是由 config.sh 文件设置的

```
2020-02-28 00:03:44,029 INFO  org.apache.flink.configuration.GlobalConfiguration            - Loading configuration property: jobmanager.rpc.address, localhost2020-02-28 00:03:44,029 INFO  org.apache.flink.configuration.GlobalConfiguration            - Loading configuration property: jobmanager.rpc.port, 61232020-02-28 00:03:44,029 INFO  org.apache.flink.configuration.GlobalConfiguration            - Loading configuration property: jobmanager.heap.size, 1024m2020-02-28 00:03:44,029 INFO  org.apache.flink.configuration.GlobalConfiguration            - Loading configuration property: taskmanager.memory.process.size, 1568m2020-02-28 00:03:44,029 INFO  org.apache.flink.configuration.GlobalConfiguration            - Loading configuration property: taskmanager.numberOfTaskSlots, 12020-02-28 00:03:44,029 INFO  org.apache.flink.configuration.GlobalConfiguration            - Loading configuration property: parallelism.default, 12020-02-28 00:03:44,030 INFO  org.apache.flink.configuration.GlobalConfiguration            - Loading configuration property: jobmanager.execution.failover-strategy, region
```

Flink 从 conf/flink-conf.yaml 中读取密钥并打印值。

这是一条重要的日志线，几乎位于底部:

```
2020-02-28 00:03:46,412 INFO  org.apache.flink.runtime.resourcemanager.StandaloneResourceManager  - Registering TaskManager with ResourceID c1438b06b128b7faf16b60e06f404f05 (akka.tcp://flink@127.0.1.1:36981/user/taskmanager_0) at ResourceManager
```

这一行基本上是说“id 为**c 1438 b 06 b 128 b 7 faf 16 b 60 e 06 f 404 f 05**的 TaskManager 将其自身注册到 ResourceManager。”

此后，JobManager 从 ResourceManager 请求一个任务槽来执行客户机的任务(实际上是流)。

# TaskManager 的日志文件

```
2020-02-28 00:03:44,843 INFO  org.apache.flink.runtime.taskexecutor.TaskManagerRunner       -  Starting TaskManager (Version: 1.10.0, Rev:aa4eb8f, Date:07.02.2020 @ 19:18:19 CET)
```

TaskManagerRunner 是独立模式下 TaskManager 的入口点。这个类构造了相关的组件，如内存管理器、网络、I/O 管理器等。

```
2020-02-28 00:03:45,846 INFO  org.apache.flink.runtime.taskexecutor.TaskManagerRunner       - Starting TaskManager with ResourceID: c1438b06b128b7faf16b60e06f404f05
```

TaskManager id(称为 ResourceId)在方法中生成:

```
public final class 
org.apache.flink.runtime.clusterframework.types.ResourceId ... {*/**
 * Generate a random resource id.
 *
 ** ***@return*** *A random resource id.
 */* public static ResourceID generate() {
   return new ResourceID(new AbstractID().toString());
}
}
```

让我们看一下日志语句，其中 TaskManager 向资源管理器注册自己:

```
2020-02-28 00:03:46,176 INFO  org.apache.flink.runtime.taskexecutor.TaskExecutor            - Connecting to ResourceManager akka.tcp://flink@localhost:6123/user/resourcemanager(00000000000000000000000000000000).2020-02-28 00:03:46,336 INFO  org.apache.flink.runtime.taskexecutor.TaskExecutor            - Resolved ResourceManager address, beginning registration2020-02-28 00:03:46,336 INFO  org.apache.flink.runtime.taskexecutor.TaskExecutor            - Registration at ResourceManager attempt 1 (timeout=100ms)2020-02-28 00:03:46,426 INFO  org.apache.flink.runtime.taskexecutor.TaskExecutor            - Successful registration at resource manager akka.tcp://flink@localhost:6123/user/resourcemanager under registration id b266a3eeb6c19ada5969142c1ffb2651.
```

日志清楚地表明“发生了什么事”，我猜想。

这是这篇文章的主题。希望我将继续第 2 部分，创建示例 Apache Flink 集群。

最后但同样重要的是，等待下一个帖子…