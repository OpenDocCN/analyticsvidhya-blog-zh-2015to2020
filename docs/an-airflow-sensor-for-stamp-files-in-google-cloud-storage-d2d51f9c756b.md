# 一种用于谷歌云存储中印章文件的气流传感器

> 原文：<https://medium.com/analytics-vidhya/an-airflow-sensor-for-stamp-files-in-google-cloud-storage-d2d51f9c756b?source=collection_archive---------13----------------------->

这篇博客最初发表在 [**大数据日报**](https://burntbit.com/an-airflow-sensor-for-stamp-files-in-google-cloud-storage/) ，也可以在 [Linkedin](https://www.linkedin.com/pulse/airflow-sensor-stamp-files-google-cloud-storage-boning-zhang) 上找到。源代码可以在 [Git](https://github.com/BoningZhang/Learning_Airflow/tree/master/common/sensors) 中找到。

Airflow 是一个用 python 编写的开源工作流控制平台。使用气流，我们可以定义任务依赖关系和调度我们的管道。不熟悉气流的可以看我其他的帖子或者这个很有帮助的[教程](/@dustinstansbury/understanding-apache-airflows-key-concepts-a96efed52b1a)。它很好地解释了气流中的关键概念。

气流的一个关键特征是传感器，它在等待事情发生。例如，如果我们的数据管道正在等待一些上游数据，那么我们可以使用传感器来检查其标记文件是否存在。我们的管道将在那里等待，直到传感器从它的标记文件中得到一个肯定的信号。假设我们需要在其分区`ds=2019-11-04`上处理一个 hive 表 A，那么一旦 HDFS 文件`hdfs://hive/A/ds=2019-11-04/_SUCCESS`被填充，它的下一个任务将被触发来进行处理。

以下是用于图章文件的气流传感器的实现。

Airflow 的 BaseSensorOperator 类中最重要的方法是 poke()函数，该函数将在一定的间隔内被调用，直到传感器找到 stamp 文件或达到其运行时间限制。

所以实现一个传感器就是实现它的`poke()`功能。在上面的例子中，我们首先在代码中生成一个 stamp 文件路径的数组作为`stamp_files`。我们的目标是使用`check_stamp_files_gcp()`检查`stamp_files`中的所有文件是否都存在。如果`check_stamp_files_gcp()`返回 true，那么我们在`poke()`函数中返回 true，否则`poke()`将返回 false，稍后将再次调用它来检查戳文件。

那么实现的关键就变成了检查图章文件，即`check_stamp_files_gcp()`。如果 HDFS 戳记文件位于本地 Hadoop 集群中，则可以使用子流程轻松实现 check_stamp_files()，如下所示:

```
ls_command = ["hadoop", "fs", "-ls"]ls_command += stamp_files[logging.info](http://logging.info/)("Running command: %s" %(" ".join(ls_command)))ret_code = subprocess.call(ls_command)==None
```

如果我们的表驻留在 Google 云存储中，我们需要调用 Google Cloud 的 API 来检查 stamp 文件。

在`check_stamp_files_gcp()`中，service_account 是 Google Cloud 项目中的一个 json 密钥文件，用来认证本地计算机，建立与 Google 云存储的连接。获取这个密钥文件的方法可以在 GCP 的[文档中找到。](https://cloud.google.com/storage/docs/reference/libraries)