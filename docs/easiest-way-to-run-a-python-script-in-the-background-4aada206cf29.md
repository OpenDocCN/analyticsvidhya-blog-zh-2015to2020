# 在后台运行 Python 脚本的最简单方法

> 原文：<https://medium.com/analytics-vidhya/easiest-way-to-run-a-python-script-in-the-background-4aada206cf29?source=collection_archive---------3----------------------->

运行 python 脚本在后台运行最简单的方法是使用 ***cronjob 特性(在 macOS 和 Linux 中)*** 。在 windows 中，我们可以使用 Windows 任务计划程序。

在 Mac 或 Linux 中，使用以下命令打开终端，检查后台正在运行哪些作业

> crontab -l

现在，要编辑列表，请键入以下命令

> crontab -e

然后，您可以通过给出时间细节来给出在特定时间运行的 python 脚本文件的路径。现在输入以下内容，按`CTRL+O and CTRL+X` 保存并退出。

目前我有两个任务计划在后台运行，这可以从下面的截图中看到

![](img/557d9fba42ffe796d57131bffd09f14b.png)

第一个作业将在该月 27 日第 11 个小时的第 15 分钟运行。该作业定期扫描现场办公室的详细联系信息，并将其存储在 csv 文件中。

第二个作业在每小时的第 15 分钟运行，检查各个地方的天气详细信息。

**cron job 的结构:**

![](img/a44bb84b0335967f8ec7c1dba56c001e.png)

**对于 Windows** ，搜索 ***任务调度*** 。

1.  打开程序。

![](img/946c379463a5fd6de6a60755ab322de3.png)

2.现在，在右侧面板上点击“*创建基本任务*

![](img/126836efd620092464d2a596141708ff.png)

给出任务名称及其描述，然后单击下一步

3.接下来选择您需要运行脚本的频率。

![](img/c4ce77cddebbe9481fb3eb9bf7e8a271.png)

如果要每天运行脚本，请选择“每天”

4.在下一个屏幕上，选择要运行脚本的日期和时间

![](img/36100f02e9a7c56f67de9005c72734db.png)

5.接下来，选择“启动程序”

![](img/d0fd7b7b09805f4df4df176c3a096804.png)

6.浏览要执行的脚本

![](img/ae9f169ba52fb44b3b8ea21485c091a1.png)

7.选择脚本并单击“完成”来计划任务

![](img/b3545d946cefa0f33126dc9a375e7e5d.png)

编码快乐！